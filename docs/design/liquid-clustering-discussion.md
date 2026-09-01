# [RFC / Discussion] Liquid Clustering in Lance

I'd like to gather feedback on adding **liquid clustering** to Lance — a Databricks-style layout
policy where you declare clustering keys once, normal writes inherit them, and an explicit
`OPTIMIZE`-style compaction incrementally re-sorts data into a multi-dimensional locality layout.

This is a proposal, not a finished design. Nothing described here is implemented yet; all type,
method, and config-key names below are suggestions I'd love to refine with maintainers before any
code lands. A more detailed RFC document is drafted alongside this post, but I wanted to open the
high-level shape for discussion first.

Related issues: #1434 (space-filling curve `cluster_by` write param), #1045 (data-skipping EPIC),
#6803 (recluster-as-compaction task), #952 (implicit partitioning, closed not-planned).

## The problem

Lance already has the pieces for data skipping, but nothing ties them into a *clustering* feature:

- **Read side exists.** The zonemap scalar index records per-zone min/max/null and prunes zones at
  scan time (`ZoneMapIndex` in `rust/lance-index/src/scalar/zonemap.rs`). On append, an existing
  zonemap with seeding enabled can already be seeded inline as data is written.
- **Write side is missing it.** `WriteParams` has no notion of clustering or sort order. Today any
  clustering must be arranged by the caller before data reaches the writer. In the Spark connector,
  `PARTITIONED BY` only approximates single-key clustering.
- **Compaction deliberately doesn't reorder.** `compact_files` / `plan_compaction` preserve insertion
  order on purpose — their positional old→new row mapping (index remap, stable-row-id rechunk) depends
  on it. There is no reorder-aware compaction entry point.
- **No general space-filling curve.** The only one is a 2D, bbox-specific Hilbert sorter for geometry
  (`HilbertSorter`); it is not a general multi-column encoder.

The hard part — **re-clustering that re-sorts by a multi-column key** — can't live in a connector,
because compaction is in the Lance core. So the bulk of this would be implemented in the Rust core,
with thin Python/Java/Spark surfaces on top.

### Why not just "Z-ORDER + partitioning"?

Legacy `ZORDER BY` + Hive-style partitioning makes you pick partition columns up front, suffers
small-file/skew problems, and needs full rewrites to change layout. Liquid clustering replaces both:
keys are declared once and can change *without* an immediate rewrite. Normal append/overwrite writes
inherit the active declaration; explicit clustering compaction brings older or externally-modified
fragments into the current layout over time.

## Proposed approach

Three cooperating pieces, all in the Rust core:

```
                 ┌─────────────────────────────────────────────┐
                 │ Table metadata (manifest.config)            │
                 │   lance.clustering.columns = ["a","b"]      │
                 │   lance.clustering.curve   = "hilbert"      │
                 │   lance.clustering.version = 1              │
                 │   lance.clustering.bits_per_dim = 16        │
                 └─────────────────────────────────────────────┘
                        ▲                    │
        declare/alter   │                    │ read by
       (UpdateConfig)   │                    ▼
   ┌───────────────┐    │   ┌──────────────────────────────────────┐
   │ Write builders│────┘   │ Compaction / recluster task          │
   │ explicit cols │        │  proposed ClusteringCompactionPlanner │
   │ or declaration│──────► │  + dedicated clustering task path     │
   └───────────────┘        └──────────────────────────────────────┘
        │                                   │
        │ produce clustered fragments,      │ re-sort selected fragments by
        │ optionally seed existing index    │ clustering key
        ▼                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Separately declared zonemap index → scan-time zone pruning   │
   └─────────────────────────────────────────────────────────────┘
```

**1. A declaration stored in `manifest.config`.** Four additive keys under a `lance.clustering.`
namespace: the ordered key `columns`, the `curve` (hilbert/zorder), a `version`, and `bits_per_dim`.
`config` is already an additive map mutated through the existing `UpdateConfig` op, so declaring or
altering the layout would be a cheap metadata-only commit. The namespace would be atomic (all-or-
nothing), validated against the schema on every manifest build, and gated by a reader+writer feature
flag so older implementations can't silently ignore the new invariants.

**2. A `SpaceFillingEncoder`.** It would map N key columns into a single fixed-size binary ordering
value; sorting by that value gives multi-dimensional locality. Roughly: normalize each column to
order-preserving unsigned "bits" (signed/unsigned int mapping, IEEE-754 bit-flip for floats, two
ordered values for bool, nulls → max coordinate), quantize to `bits_per_dim`, then either interleave
(Z-order) or apply the Hilbert transform. The first cut would accept only top-level numeric/boolean
columns; strings, temporal, decimal, nested, etc. rejected.

**3. A dedicated, reorder-aware clustering compaction.** Kept entirely separate from ordinary
compaction (see the key decisions below). A planner selects *under-clustered* fragments, a task
re-sorts their rows by the clustering key, and a commit step stamps the output with the current
clustering version.

## Key design decisions I'd like feedback on

**Keep clustering out of `WriteParams` and `CompactionMode`.** Rather than fold reordering into the
existing `compact_files` / `CompactionMode`, I propose *separate* entry points
(`plan_clustering_compaction`, `compact_files_with_clustering`, and a `CompactionRequest::Clustering`
discriminator that normalizes to `CompactionMode::Reencode`). Two reasons: `CompactionMode` is a
public enum, and ordinary compaction's positional row mapping assumes preserved order — a reordering
variant would either break that contract or risk cross-routing order-preserving vs reordering work in
distributed execution. Ordinary `compact_files` / `plan_compaction` would stay byte-for-byte unchanged.

**Per-fragment "clustered-at" version stamp.** To make reclustering incremental we need to know which
fragments are already clustered under the current spec. I propose a new scalar field on `DataFragment`
(`0` = unset) recording the clustering `version` that governed a fragment's write. A fragment is
"under-clustered" unless its stamp exactly equals the table's current `lance.clustering.version`.
Bumping the version (on any key/curve/bits change, or on its own) lazily marks all existing fragments
eligible for reclustering — so **changing keys never forces a full rewrite**; the table converges over
later `OPTIMIZE` runs while staying readable. In Rust the decoded stamps would live in a private
`Manifest` sidecar aligned with `Manifest::fragments`, not as public `Fragment` fields.

**Which paths sort, and which defer.** Append and overwrite would inherit the declaration and sort via
a proposed `cluster_sort_stream` (a `SortExec` over a hidden `FixedSizeBinary` column, spilling to
disk). Update and merge-insert would *not* sort — the fragments they change simply become
under-clustered and get picked up later. This keeps the hot write paths simple. Explicit columns on an
*undeclared* dataset would be a one-shot sort with no persistent stamp.

**Incremental planner with optional budgets.** The planner would group adjacent under-clustered
fragments up to `target_rows_per_fragment`, honoring optional `max_source_fragments` /
`max_source_rows` / `max_source_bytes` budgets (all `None` by default = unbounded). Budgets let a large
table converge over several bounded runs.

**Deliberate first-iteration restrictions.** Because reclustering reorders rows, it would initially
**reject** datasets using stable row ids or carrying a remappable secondary index (drop → recluster →
rebuild), plus `defer_index_remap=true`. Carrying row identity through the sort is a clear follow-up.

**Reuse zonemap for reads — no new pruning.** Declaring clustering would *not* auto-create a zonemap;
a separately declared zonemap with seeding just benefits from the tighter per-zone ranges that
clustering produces. Automatic zonemap declaration from the spec is future work.

## Proposed API surface (thin bindings, names TBD)

- **Rust:** `InsertBuilder::with_cluster_by_columns` / `FragmentCreateBuilder::with_cluster_by_columns`
  for explicit write columns; `Dataset::set_clustering` / `clustering_spec` / `clear_clustering` for the
  config-backed declaration; `ClusteringCompactionPlanner` + `plan_clustering_compaction` +
  `ClusteringCompactionTask::execute` + `commit_clustering_compaction` for distributed recluster, or the
  single-process `compact_files_with_clustering` helper.
- **Python:** `write_dataset(..., cluster_by=["a", "b"])`;
  `dataset.set_clustering(columns, *, curve, version, bits_per_dim)` / `dataset.clustering_spec()` /
  `dataset.clear_clustering()`; `dataset.optimize.compact_files(compaction_mode="cluster")`. The string
  `"cluster"` would be a binding-level request discriminator that routes to the dedicated clustering
  APIs — not a new Rust `CompactionMode`.
- **Java/JNI:** `WriteParams.Builder.withClusterBy(List<String>)`;
  `Dataset.setClustering / getClusteringSpec / clearClustering`; a `CompactionMode.CLUSTER` value that,
  like Python's `"cluster"`, is a binding-level discriminator routing to the dedicated core path.
- **Spark connector (out of this repo):** map SQL `CLUSTER BY (a, b)` to the persisted spec; `OPTIMIZE`
  triggers the incremental recluster; scan-time pruning stays as-is.

Validation and resolution would be centralized in Rust; bindings stay thin.

## Proposed phased delivery

- **Phase 0 — spec plumbing.** `ClusteringSpec` type, the `lance.clustering.*` config declaration, and
  `set_clustering` / `clustering_spec` / `clear_clustering`. No layout behavior change yet.
- **Phase 1 — encoder.** General multi-column Z-order + Hilbert `SpaceFillingEncoder` with tests (order
  preservation, Hilbert adjacency, null handling, mixed types, bit-budget validation).
- **Phase 2 — write-side clustering.** Builders accept explicit columns separately from `WriteParams`,
  resolve the spec on append/overwrite, sort via `cluster_sort_stream`, and stamp committed fragments.
  Update/merge-insert defer.
- **Phase 3 — incremental recluster.** The dedicated planner/task/commit path reorders under-clustered
  fragments by the clustering key, honoring optional source budgets, persisting the per-fragment stamp.
  Initially defers stable-row-id / remappable-index datasets and cross-fragment merge-in, and hardens
  the distributed task/result boundary over the phase.
- **Phase 4 — bindings & connectors.** Python + Java wrappers and the `"cluster"` / `CLUSTER`
  discriminators. Defers Spark integration and automatic zonemap declaration.
- **Phase 5 — docs & benchmarks.** Data-skipping recall vs unclustered baseline; write/optimize
  overhead; key-change convergence.

## Open questions (feedback especially welcome here)

1. **Curve & precision defaults.** Hilbert (better locality) vs Z-order (cheaper) as default? How to
   handle skewed / high-cardinality columns — per-column bit widths, whitening, rank normalization?
2. **Alternative quality metric.** Should a zonemap-overlap heuristic supplement the version stamp when
   deciding whether an already-current fragment would benefit from another rewrite?
3. **Merge scope during recluster.** How aggressively to pull in overlapping *already-clustered*
   fragments — the classic write-amplification vs clustering-quality knob.
4. **Zonemap `rows_per_zone` alignment.** Should the default zone size be coupled to the clustering
   config so zone granularity is fine enough to benefit?
5. **Concurrency.** Confirm recluster's conflict resolution with concurrent appends/updates matches
   existing compaction guarantees (it should, since it reuses that path).
6. **Distributed payload hardening.** Where and how strictly to validate task fragment descriptors
   against the read snapshot, reject empty tasks/results, and verify row-count preservation before
   committing worker output.

## Alternatives considered

- **Fold reordering into `CompactionMode` / `compact_files`.** Rejected — breaks the public enum and
  the positional row-mapping contract, and risks cross-routing in distributed execution.
- **Reuse the stable `unenforced-clustering-key:position` marker.** Rejected — that marker *asserts* an
  achieved ordering for query optimization, whereas liquid clustering is a *policy* fragments converge
  to. Overloading it conflates an assertion with a goal.
- **Dedicated top-level manifest fields for the declaration.** Rejected in favor of `config` (already
  additive, atomic, cheap to update) — though the per-fragment stamp does warrant a real protobuf field.
- **Extend the 2D geo `HilbertSorter`.** Rejected — bbox-specific and 2D; a general N-column encoder is
  a cleaner foundation and leaves geo code untouched.
- **Sort eagerly in update/merge-insert.** Rejected for v1 — complicates hot paths and their row-mapping
  guarantees; deferring via compaction keeps writes simple.

## What I'm looking for

Feedback on: (a) whether the config-based declaration + per-fragment version stamp is the right
metadata model; (b) the separate-API approach vs extending `CompactionMode`; (c) the encoder's initial
type restrictions and precision strategy; (d) the first-iteration restrictions (stable row ids /
remappable indices) and whether those blockers are acceptable to start; and (e) naming across the
Rust/Python/Java surfaces.

If the direction sounds reasonable, I'm happy to turn the detailed RFC into a phased set of PRs.
