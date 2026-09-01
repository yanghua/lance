# Proposal: Liquid Clustering in Lance

Status: Proposal / RFC (draft — no implementation yet)
Related issues: lance-format/lance#1434 (space-filling curve `cluster_by` write param),
lance-format/lance#1045 (EPIC: statistics and data skipping), lance-format/lance#952 (implicit
partitioning, closed not-planned), lance-format/lance#6803 (recluster task for row-id healing).

> This document proposes adding Databricks-style "liquid clustering" to Lance. The intent is to let
> a clustering layout be declared once, be inherited by normal `InsertBuilder` append and overwrite
> writes, and be maintained by an explicit clustering compaction. Update and merge-insert paths
> would not sort immediately; fragments whose layout they change would remain or become
> under-clustered and be handled by a later clustering compaction. Separately declared zonemap
> indices could then consume the resulting locality for data skipping.
>
> Nothing described here exists in the codebase yet. Every type, method, and config key named below
> is a *proposed* surface offered for discussion; names and shapes are subject to review before any
> implementation lands.

## 1. Background and motivation

Lance today has the read-side and write-side building blocks for data skipping, but not the layer
that would tie them into a clustering feature:

- **Read side (exists).** The zonemap scalar index records per-zone min/max/null over
  `rows_per_zone` rows and prunes zones at scan time (`ZoneMapIndex` in
  `rust/lance-index/src/scalar/zonemap.rs`). On append to a current-format dataset, an existing
  zonemap index with seeding enabled can already be seeded inline through `IndexSeedWriter`.
- **Write side (gap).** `WriteParams` has no notion of clustering or sort order. Any clustering must
  be arranged by the caller before data reaches the writer. In the Spark connector, `PARTITIONED BY`
  approximates single-key clustering: Spark shuffles/sorts by the key, and `LanceDataWriter` rolls a
  fresh fragment whenever the key changes.
- **Compaction (gap).** Ordinary `compact_files` and `plan_compaction` deliberately do *not* reorder
  data — they use the ordinary planner and try to preserve insertion order. There is no reorder-aware
  compaction entry point today.
- **Space-filling curves (gap).** The only space-filling curve implementation is a 2D Hilbert sorter
  specialized for geometry bounding boxes (`HilbertSorter` in
  `rust/lance-index/src/scalar/rtree/sort/hilbert_sort.rs`). It is not a general multi-column encoder.

The core of liquid clustering — **re-clustering that re-sorts data by a multi-column key** — cannot
be implemented in a connector (Spark/LanceDB) alone, because compaction lives in the Lance core. We
therefore propose to implement it primarily in the Rust core, with thin binding/connector surfaces
on top.

### Why not "just Z-ORDER + partitioning"

Legacy `ZORDER BY` + Hive-style partitioning requires the user to pick partition columns up front,
suffers small-file / skew problems, and needs full rewrites to change layout. Liquid clustering would
replace both: clustering keys are declared once and can change without an immediate rewrite. Normal
append and overwrite writes would inherit the active declaration, while an explicit clustering
compaction would bring older fragments, and fragments changed by other write paths, into the current
layout.

## 2. Goals and non-goals

### Goals

1. Declare clustering keys on a dataset (proposed `Dataset::set_clustering`), persisted in table
   metadata. SQL `CLUSTER BY (a, b, ...)` connector support is future work.
2. Cluster writes that explicitly request it, and normal append/overwrite writes that inherit the
   declaration, using a multi-column space-filling curve (Z-order/Hilbert), so each fragment/file is
   value-coherent across all key columns.
3. Re-cluster with an `OPTIMIZE`-driven task that picks up under-clustered data and merges it into
   the clustered layout. Optional source budgets should be able to bound the amount of data touched
   per run.
4. Change clustering keys without rewriting already-clustered data; new keys apply to future
   clustering passes.
5. Reuse the existing zonemap index for read-side pruning — no new pruning mechanism.
6. Keep Python/Java bindings and the Spark/LanceDB connectors as thin wrappers over Rust.

### Non-goals (for the first iteration)

- Replacing vector indexes or ANN clustering (IVF/PQ) — unrelated subsystem.
- Automatic, learned clustering-key selection (Databricks "CLUSTER BY AUTO"). Keys are explicit.
- Cross-fragment global sort guarantees. Clustering is statistical locality, not a total order.
- Changing any *stable* on-disk format contract. Clustering metadata must be additive.

## 3. Proposed design overview

We propose three cooperating pieces, all in the Rust core:

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

- **Clustering spec** would be stored in `Manifest::config` as a complete set of additive config
  keys. The `lance.clustering.` namespace would be atomic: no key in that namespace means the
  declaration is absent, while any key in it requires all four known keys to be present and valid in
  the resulting manifest. Partial or malformed declarations would be rejected. A clustering feature
  flag would gate readers and writers that predate this contract.
- **A `SpaceFillingEncoder`** would turn N key columns into a single 1-D ordering value using a
  space-filling curve. Sorting by this value gives multi-dimensional locality.
- **The normal insert write path** would optionally sort each incoming batch stream by the clustering
  value before fragment rolling. On append, the normal index-seed path could also seed a separately
  declared zonemap index when that index has seeding enabled. Update and merge-insert would not run
  this sort; changed or newly produced fragments without an authoritative clustering stamp would be
  selected by later clustering compaction.
- **A clustering-aware compaction planner** would select under-clustered fragments and rewrite them
  through a reorder-enabled path. Optional source budgets would make this incremental across runs;
  the proposed defaults impose no source-volume limit.

## 4. Clustering key encoder

We propose a new clustering module in the `lance-index` crate that does *not* reuse the geo
`HilbertSorter` (which is 2D and bbox-specific). A `SpaceFillingEncoder` would map N key columns into
a single fixed-size binary ordering value; a `ClusteringCurve` selector would choose Z-order or
Hilbert encoding.

The initial encoder would accept only top-level columns with these Arrow types: `Boolean`,
`Int8`/`Int16`/`Int32`/`Int64`, `UInt8`/`UInt16`/`UInt32`/`UInt64`, `Float32`, and `Float64`.
Strings, binary, dictionaries, temporal values, decimals, nested values, and other types would be
rejected.

Proposed encoding behavior:

1. **Normalize each key column to unsigned integer "bits."** Integers would use an order-preserving
   signed or unsigned mapping, floats the standard order-preserving IEEE-754 bit flip, and booleans
   two ordered values. Nulls would map to the maximum coordinate.
2. **Quantize** each column to a fixed bit width `b` (e.g. 16 or 20 bits/column), tunable.
3. **Interleave** (Z-order) or apply the Hilbert transform across dimensions to produce the ordering
   value.

We propose fixed-domain most-significant-bit truncation so the same value receives the same
coordinate in every input batch. This can collapse small ranges of wide integer and floating-point
types at low bit widths — an accepted trade-off of the fixed-domain approach. Null would map to the
maximum coordinate in its own dimension; a multi-dimensional curve would not promise row-level
`NULLS LAST`.

Precision/whitening (per-column bit width, handling skew) is an open question — see §10.

## 5. Format / metadata changes (additive, no stable-format break)

The complete desired layout would be stored in `manifest.config`. It should deliberately *not* reuse
the existing `lance-schema:unenforced-clustering-key:position` marker: that stable marker asserts an
already-achieved physical ordering for query-engine optimizations, while liquid clustering is a
policy to which existing fragments may only converge incrementally.

| Key | Type | Meaning |
|---|---|---|
| `lance.clustering.columns` | JSON array of strings | Ordered clustering-key columns |
| `lance.clustering.curve` | string (`hilbert` \| `zorder`) | Space-filling curve |
| `lance.clustering.version` | int | Layout version used to detect under-clustered fragments |
| `lance.clustering.bits_per_dim` | int | Per-column quantization bit width |

Rationale: `config` is already an additive `HashMap<String,String>` mutated through the existing
`UpdateConfig` operation (`rust/lance/src/dataset/metadata.rs`), so changing the desired layout would
be a cheap metadata-only commit. The complete declaration would be updated atomically. Any change to
columns, curve, or bit width would have to increase the layout version so existing fragments become
eligible for incremental reclustering. Increasing only the version would also be allowed and would
force all fragments with an older stamp to become eligible again. Clearing the declaration would
retain existing fragment stamps; re-enabling clustering would therefore need a version greater than
the maximum retained stamp. Every manifest build would validate the complete declaration against the
resulting schema, so a schema change could not leave an active clustering column missing, nested, or
of an unsupported type. Manifests carrying the declaration would set both a clustering reader and
writer feature flag, so older implementations do not open the dataset while ignoring the new
invariants.

**Per-fragment "clustered-at" marker.** To make re-clustering incremental we must know which
fragments are already clustered under the current spec. We propose a marker that records the active
declared clustering `version` that governed a fragment's write. A fragment would be
"under-clustered" unless its recorded version exactly equals the table's current
`lance.clustering.version`; an absent, older, or unexpected future stamp would be selected. A
one-shot sort on an undeclared dataset would therefore remain unstamped.

We propose to persist this stamp in a new scalar `DataFragment` protobuf field (`0` = unset). In
Rust, decoded stamps could be kept in a private `Manifest` sidecar aligned positionally with
`Manifest::fragments` rather than exposed as fields on the public `Fragment` type; manifest
serialization would write the sidecar values back to that field. A reader-and-writer feature flag
would fence older readers and writers whenever an active clustering declaration or any fragment
stamp is present.

Because the proposed field would be a stable scalar encoding whose zero value means "unset", active
clustering versions must be positive, and that zero sentinel could not later be reinterpreted as a
valid version. A future format needing explicit presence or version zero would require a new field
or encoding.

A future quality heuristic could additionally infer overlap from zonemap ranges, but it would not
replace the version stamp as the authoritative current-layout predicate.

## 6. Write path

We propose to keep clustering out of `WriteParams`. Normal dataset writes would accept explicit
column names through a new `InsertBuilder::with_cluster_by_columns`; multi-fragment writes could use
`FragmentCreateBuilder::with_cluster_by_columns`. The builders would resolve those columns to a
private `ClusteringSpec` passed separately to the internal writer.

Proposed write behavior:

1. **Resolve** the spec: `InsertBuilder` would resolve explicit columns, or inherit the dataset's
   declared spec when columns are omitted, for both append and overwrite. Overwrite would inherit the
   declaration because overwrite commits preserve dataset config. On a dataset with an active
   declaration, explicit columns must exactly match it. On a new or undeclared dataset, explicit
   columns would request a one-shot sort and would not declare or stamp a persistent layout. Low-level
   multi-fragment writes would have to pass explicit columns through `FragmentCreateBuilder`; omitting
   them would not inherit the declaration. Update and merge-insert would intentionally omit the
   clustering spec: they would preserve stamps only for fragments whose layout metadata is unchanged,
   while changed or new fragments would be left unstamped for a later clustering compaction.
   A note on transport: a public fragment-write API that returns only `Vec<Fragment>` cannot carry the
   private manifest-sidecar version to a later manual commit. We would need a hidden transport (e.g. a
   `write_fragments_with_clustering_version` variant) for distributed callers, and Python could instead
   write through an uncommitted-stream path, extract the reserved version from the resulting
   transaction, and carry it alongside exported fragment metadata. Any caller that separates fragment
   writing from commit must preserve the transaction's reserved clustering-version marker; otherwise the
   sorted fragments would be committed without a stamp.
2. **Sort** the write's batch stream by the encoded clustering value before fragments are written. We
   propose a `cluster_sort_stream` helper: a `SortExec` over a hidden `FixedSizeBinary` ordering
   column, spilling to disk for large inputs. Oversized input batches would be split and deep-copied
   before they reach the sort, because DataFusion cannot spill a single batch that already exceeds its
   memory reservation; a single row larger than the cap would be rejected. For the streaming writer
   this is a sort within the write unit; global ordering across concurrent writers would not be
   guaranteed (that is what re-clustering converges).
3. **Seed an existing zonemap when configured.** Declaring clustering would not automatically declare
   or build a zonemap. On append to an existing current-format dataset, the normal `IndexSeedWriter`
   path could seed an already-declared zonemap index when that index has seeding enabled. A clustering
   column without such an index would receive no automatic zonemap declaration or seed. Automatic
   deferred zonemap declaration from the clustering spec is future work.
4. **Stamp** each committed fragment governed by the active declaration with the current clustering
   version (§5). The write transaction would carry the resolved version internally until final
   fragment IDs are assigned; manifest construction would then reconcile the private sidecar. It would
   preserve a prior stamp only when the fragment's layout-identifying metadata is unchanged, and clear
   the stamp when a non-clustering rewrite changes that layout.

Connectors would keep their current role: Spark's `RequiresDistributionAndOrdering` can still
pre-sort at the engine for scale; the core sort would be the correctness backstop when the engine does
not.

## 7. Compaction / re-clustering

This is the part that must live in core because ordinary `compact_files` deliberately does not
reorder. We propose to keep the ordinary `compact_files` and `plan_compaction` entry points
order-preserving and to leave `CompactionMode` unchanged (the nonbreaking three-variant enum
`Reencode`, `TryBinaryCopy`, `ForceBinaryCopy`). Reordering would be exposed through *separate*,
newly introduced APIs instead of being folded into the existing ones.

**Dedicated clustering operation.** We propose a `ClusteringCompactionPlanner` and a
`plan_clustering_compaction` entry point that produce a `ClusteringCompactionPlan`. Its tasks would be
executed as `ClusteringCompactionTask` values, which resolve the declaration at the task's immutable
read version and re-sort rows through a private clustering rewrite strategy before writing fragments
via `cluster_sort_stream`. Each task would yield a `ClusteringRewriteResult`; a
`commit_clustering_compaction` step would validate and commit those results, then record the task's
clustering version after final fragment IDs are assigned. Clustering would require the existing
`CompactionMode::Reencode` behavior and would not use binary copy. Output would still respect the
writer semantics of `target_rows_per_fragment` / `max_bytes_per_file`: the row target would be passed
as `WriteParams::max_rows_per_file`, while the byte limit would remain the writer's existing soft
limit.

A single-process convenience entry point, `compact_files_with_clustering`, would wrap that same flow:
plan with `plan_clustering_compaction`, execute the dedicated tasks, and commit with
`commit_clustering_compaction`. Distributed callers would use those three stages directly.

**Incremental planner.** The dedicated `ClusteringCompactionPlanner` would select *under-clustered*
fragments — those whose per-fragment stamp does not equal the dataset's current
`ClusteringSpec::version` — group adjacent whole source fragments until the accumulated live rows
reach or exceed `target_rows_per_fragment`, and honor optional `max_source_fragments`,
`max_source_rows`, and `max_source_bytes` budgets. Because source fragments would not be split during
planning, a task could exceed `target_rows_per_fragment`; the option would be separately forwarded to
the output writer as `max_rows_per_file` during rewriting. All three source budgets would default to
`None`, so clustering compaction would be unbounded by source volume unless a caller or dataset config
sets at least one. When several are set, all would be hard upper bounds: planning stops before the
next eligible fragment would exceed any one. Rows would mean live rows; bytes would include source
data and overlay files but exclude separately stored Blob payloads. If the first eligible fragment
exceeds a configured budget, the plan would be empty and the budget must be raised. An
already-clustered fragment would break adjacency so it is left untouched, and a second run at the same
version would be a no-op.

**Distributed operation separation.** We propose that `ClusteringCompactionTask` and
`ClusteringRewriteResult` use dedicated, version-tagged serialized envelopes rather than the ordinary
`CompactionTask` and `RewriteResult` payloads. The tags would prevent accidental cross-routing between
order-preserving and row-reordering work. Task execution would check out the declared read version and
validate the clustering declaration and clustering-specific restrictions at that snapshot. The initial
design does not require comparing caller-provided `TaskData` fragment descriptors with the manifest's
fragment descriptors at that version; `commit_clustering_compaction` would require one read version and
clustering version across all results and check that version against the declaration before reserving
fragment IDs, but would not verify complete original-fragment identity or that output fragments
preserve the input live-row count. The serialized task/result boundary would therefore be a
trusted-worker boundary, not an attestation of the referenced files or rewrite contents. Distributed
callers would be required to provide valid, non-empty fragment descriptors corresponding to the stated
snapshot and preserve the tagged execution results unchanged. Hardening this boundary (descriptor
validation against the snapshot, rejecting empty tasks/results, row-count preservation checks) is
called out in §10 and would be part of the delivery plan rather than deferred indefinitely.

*Explicitly out of the first iteration:* pulling in overlapping already-clustered fragments so new data
merges into the right place (the planner would initially only recluster under-clustered fragments among
themselves), and per-task combined-key-range grouping. These refine clustering quality and are
follow-ups.

**Changing keys without full rewrite.** Bumping the clustering version while changing the columns or
other layout parameters would mark all existing fragments as under-clustered *lazily*; they would be
re-clustered by later `OPTIMIZE` runs. With a source budget configured, a large table could converge
over several bounded runs. With the default unbounded source budgets, one run could select all
eligible fragments. Old data would stay readable throughout.

**Interaction with existing compaction machinery.** Ordinary compaction preserves row order, so its
positional old→new row mapping (for index remap and stable-row-id rechunk) holds. Reclustering
*reorders* rows, which breaks that positional assumption. Rather than silently corrupt row ids or a
secondary index, we propose that the dedicated clustering path initially **reject** datasets that use
stable row ids or carry a remappable secondary index (drop the index, recluster, rebuild), as well as
`defer_index_remap=true`. Carrying row identity through the sort and rebuilding the mapping from the
sorted order is the follow-up that would lift this restriction.

## 8. Read path

No new pruning mechanism would be introduced. When a zonemap index has been declared on a clustering
column, the existing scan-time pruning could exploit the resulting value locality. Declaring
clustering alone would not declare, seed, or build a zonemap, so clustered data without a separately
created zonemap would receive no zonemap-based pruning benefit. Automatic deferred zonemap declaration
from the clustering spec, plus query-planning exposure for cost estimates and connector pushdown,
remain future work.

## 9. Proposed API surface (thin bindings)

We would centralize resolution and validation in Rust while keeping the binding-level names consistent
(`cluster_by`, `clustering`). All names below are proposals subject to review.

- **Rust:** `InsertBuilder::with_cluster_by_columns` and
  `FragmentCreateBuilder::with_cluster_by_columns` for explicit write columns;
  `Dataset::set_clustering` / `Dataset::clustering_spec` / `Dataset::clear_clustering` to
  declare/read/drop the config-backed spec. Reclustering would use `ClusteringCompactionPlanner`,
  `plan_clustering_compaction`, `ClusteringCompactionTask::execute`, `commit_clustering_compaction`,
  or the single-process `compact_files_with_clustering` helper. `CompactionMode` would gain no
  clustering variant; a separate `CompactionRequest::Clustering` discriminator would carry options
  normalized to `CompactionMode::Reencode`. `WriteParams` would contain no clustering field. Since a
  public `FragmentCreateBuilder::write_fragments` returning `Vec<Fragment>` cannot expose the
  declaration version needed by a separate manual commit, a hidden
  `write_fragments_with_clustering_version` transport would exist for binding transports.
- **Python:** `write_dataset(..., cluster_by=["a", "b"])`;
  `dataset.set_clustering(columns, *, curve, version, bits_per_dim)` / `dataset.clustering_spec()`
  (returns a dict or `None`) / `dataset.clear_clustering()`;
  `dataset.optimize.compact_files(compaction_mode="cluster")`. The string `"cluster"` would be a
  binding-level request discriminator, not a Rust `CompactionMode`: Python would normalize the core
  mode to `Reencode` and dispatch execute/plan/commit to the dedicated clustering APIs. The binding
  would marshal write columns into the Rust builder, where the declaration is resolved and
  authoritatively validated.
- **Java/JNI:** `WriteParams.Builder.withClusterBy(List<String>)` as the public Java binding surface,
  with JNI forwarding its column list to the Rust write builder; `Dataset.setClustering(ClusteringSpec)`
  / `Dataset.getClusteringSpec()` / `Dataset.clearClustering()`; and a `CompactionMode.CLUSTER` value.
  Java's `CLUSTER` would likewise be a binding-level request discriminator: the distributed `Compaction`
  API would pass it to JNI, which maps it to `CompactionRequest::Clustering`, normalizes the core mode
  to `Reencode`, and uses the dedicated clustering plan, tagged task/result, and commit path. It would
  not be a Rust `CompactionMode` variant; the Java `Dataset.compact` convenience method would dispatch
  the same binding-level value to `compact_files_with_clustering`. `ClusteringSpec` / `ClusteringCurve`
  would be thin value types mirroring the Rust/Python shape.
- **Spark connector:** map SQL `CLUSTER BY (a, b)` to the persisted spec; `OPTIMIZE` triggers the
  incremental recluster; keep `LanceScanBuilder` pruning as-is. (Connector lives outside this repo.)

## 10. Open questions

1. **Precision and curve evaluation.** Should Hilbert or Z-order be the default? Hilbert gives better
   locality; Z-order is cheaper. Benchmarking should guide defaults and whether per-column bit widths
   should change, and how to handle skewed or high-cardinality columns (whitening / rank normalization).
2. **Alternative quality metrics.** The proposed fragment-version marker would be the authoritative
   under-clustered predicate. Should a zonemap-overlap heuristic later supplement it when deciding
   whether already-current fragments would benefit from another rewrite?
3. **Merge scope during recluster.** How aggressively to pull in overlapping already-clustered
   fragments — trades write amplification against clustering quality (the classic incremental-clustering
   cost knob).
4. **Zonemap `rows_per_zone` alignment.** Clustering quality is only useful if zone granularity is fine
   enough; should the default zone size be coupled to the clustering config?
5. **Concurrency.** Recluster is a rewrite; confirm conflict resolution with concurrent appends/updates
   matches existing compaction guarantees (it should, since it reuses that path).
6. **Distributed payload hardening.** Validate task fragment descriptors against the immutable read
   snapshot, reject empty tasks/results at API boundaries, and verify row-count preservation before
   committing worker output. This should be scheduled explicitly rather than left as a trusted-worker
   assumption.

## 11. Proposed phased delivery

- **Phase 0 — spec plumbing.** A `ClusteringSpec` type, the complete declaration in `lance.clustering.*`
  config, and `Dataset::set_clustering` / `clustering_spec` / `clear_clustering`. No layout behavior
  change yet.
- **Phase 1 — encoder.** A general multi-column Z-order + Hilbert `SpaceFillingEncoder` in
  `lance-index::clustering`, with tests (order preservation, Hilbert adjacency, null handling, mixed
  supported types, bit-budget validation).
- **Phase 2 — write-side clustering.** Rust write builders accept explicit column lists separately from
  `WriteParams`, resolve the private spec from the dataset on append and overwrite, and sort with
  `cluster_sort_stream`. Commits stamp fragments governed by the active declaration in the private
  manifest sidecar. An explicit one-shot sort on a new or undeclared dataset is not stamped. Update and
  merge-insert do not inherit the sort; changed/new fragments from those paths are left under-clustered.
  Low-level Rust fragment writes must preserve the separately returned clustering-version transport when
  committing if their outputs are to be stamped. Automatic zonemap declaration from the clustering spec
  is deferred to Phase 4; inline seeding of a separately declared zonemap already uses the existing
  index-seed path.
- **Phase 3 — incremental recluster.** The dedicated `ClusteringCompactionPlanner` /
  `plan_clustering_compaction` path selects under-clustered fragments and emits
  `ClusteringCompactionTask` values whose execution reorders rows by the clustering key. It honors any
  configured per-run source budgets (unbounded by default); `ClusteringRewriteResult` values are
  committed through `commit_clustering_compaction`, and the per-fragment stamp is persisted through the
  proposed scalar `DataFragment` clustering-version protobuf field (`0` = unset) and consumed here.
  Initially deferred: reclustering on datasets with stable row ids or a remappable index (rejected to
  avoid corrupting the positional row mapping), and pulling in overlapping already-clustered fragments
  for better merge quality. Distributed task/result structural validation would be scoped here (see §10):
  callers initially provide valid non-empty task descriptors for the stated snapshot and preserve
  execution results unchanged, with enforcement hardened over the phase.
- **Phase 4 — bindings & connectors.** Python and Java wrappers over the Rust core: `cluster_by` on the
  write path, `set_clustering` / `clustering_spec` / `clear_clustering` for declaration, and the
  `"cluster"` / `CLUSTER` binding-level request discriminators, which route to the dedicated clustering
  core APIs while ordinary `compact_files` / `plan_compaction` remain unchanged. Deferred: the Spark
  `CLUSTER BY` + `OPTIMIZE` connector integration (out of this repo) and automatic zonemap declaration.
- **Phase 5 — docs & benchmarks.** Data-skipping recall vs unclustered baseline; write/optimize
  overhead; key-change convergence.

## 12. Alternatives considered

- **Fold reordering into `CompactionMode` / `compact_files`.** Rejected: `CompactionMode` is a public
  enum and ordinary compaction's positional old→new row mapping (index remap, stable-row-id rechunk)
  assumes preserved order. A row-reordering variant would either break that contract or risk
  cross-routing order-preserving and reordering work in distributed execution. Separate APIs and a
  separate request discriminator keep the existing surface untouched.
- **Reuse the stable `unenforced-clustering-key:position` marker.** Rejected: that marker asserts an
  already-achieved physical ordering for query-engine optimization, whereas liquid clustering is a
  policy that fragments converge to incrementally. Overloading it would conflate an assertion with a
  goal.
- **Add dedicated top-level manifest fields instead of `config` keys.** Rejected for the declaration
  itself: `config` is already an additive, atomically-updated map with an existing `UpdateConfig`
  operation, making layout changes cheap metadata-only commits. A per-fragment stamp, however, does
  warrant a real protobuf field because it must align with each fragment.
- **Extend the existing 2D geo `HilbertSorter`.** Rejected: it is bbox-specific and 2D; a general
  N-column encoder is a cleaner foundation and keeps geo code untouched.
- **Sort eagerly in update/merge-insert.** Rejected for the first iteration: it complicates those hot
  paths and their row-mapping guarantees. Leaving changed fragments under-clustered and converging via
  compaction keeps the write paths simple.

## 13. Alignment with upstream

- Would implement the space-filling curve `cluster_by` write work tracked by **#1434** and advance the
  **#1045** data-skipping EPIC (zonemap consumption).
- The incremental-recluster task echoes the "recluster as a compaction-like task" idea floated in
  **#6803**.
- Would supersede the not-planned **#952** by declaring keys once and maintaining incrementally instead
  of exposing partition/fragment mechanics.

Before implementation begins, we would confirm scope and API names with maintainers on the relevant
tracking issues.
