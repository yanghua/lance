# Liquid Clustering in Lance

Status: Implemented core with follow-up work
Tracking issues: lance-format/lance#1434 (space-filling curve `cluster_by` write param),
lance-format/lance#1045 (EPIC: statistics and data skipping), lance-format/lance#952 (implicit
partitioning, closed not-planned), lance-format/lance#6803 (recluster task for row-id healing).

> This document describes the implemented core of Databricks-style "liquid clustering" in Lance
> and the remaining follow-up work. A clustering layout is declared once, inherited by subsequent
> append and overwrite writes, and maintained by explicit clustering compaction. Separately
> declared zonemap indices can consume the resulting locality for data skipping.

## 1. Background and motivation

Before the liquid-clustering work described here, Lance had the read-side and write-side building
blocks for data skipping but not the layer that tied them into a clustering feature:

- **Read side (exists).** The zonemap scalar index records per-zone min/max/null over
  `rows_per_zone` rows and prunes zones at scan time (`ZoneMapIndex` in
  `rust/lance-index/src/scalar/zonemap.rs`). On append to a current-format dataset, an existing
  zonemap index with seeding enabled can be seeded inline through `IndexSeedWriter`.
- **Write side (historical gap).** `WriteParams` had no notion of clustering or sort order. Any
  clustering had to be arranged by the caller before data reached the writer. In
  the Spark connector, `PARTITIONED BY` approximates single-key clustering: Spark shuffles/sorts by
  the key, and `LanceDataWriter` rolls a fresh fragment whenever the key changes.
- **Compaction (historical gap).** Ordinary `compact_files` and `plan_compaction` explicitly do
  *not* reorder data — they continue to use the ordinary planner and try to preserve insertion
  order. Reordering is exposed through the separate `compact_files_with_clustering` and
  `plan_clustering_compaction` APIs, keeping `CompactionMode` unchanged.
- **Space-filling curves (historical gap).** Before this work, the only space-filling curve
  implementation was a 2D Hilbert sorter specialized for geometry bounding boxes
  (`HilbertSorter` in `rust/lance-index/src/scalar/rtree/sort/hilbert_sort.rs`). It was not a
  general multi-column encoder.

The core of liquid clustering — **re-clustering that re-sorts data by a multi-column key** — cannot
be implemented in a connector (Spark/LanceDB) alone, because compaction lives in the Lance core.
The implementation therefore lives primarily in the Rust core, with thin binding/connector
surfaces on top.

### Why not "just Z-ORDER + partitioning"

Legacy `ZORDER BY` + Hive-style partitioning requires the user to pick partition columns up front,
suffers small-file / skew problems, and needs full rewrites to change layout. Liquid clustering
replaces both: clustering keys are declared once and can change without an immediate rewrite. New
writes inherit the active declaration, while explicit clustering compaction brings older fragments
into the current layout.

## 2. Goals and non-goals

### Goals

1. Declare clustering keys on a dataset (`Dataset::set_clustering`), persisted in table metadata.
   SQL `CLUSTER BY (a, b, ...)` connector support is future work.
2. Cluster newly written data by those keys using a multi-column space-filling curve
   (Z-order/Hilbert), so each fragment/file is value-coherent across all key columns.
3. Re-cluster with an `OPTIMIZE`-driven task that picks up under-clustered data and merges it into
   the clustered layout. Optional source budgets can bound the amount of data touched per run.
4. Change clustering keys without rewriting already-clustered data; new keys apply to future
   clustering passes.
5. Reuse the existing zonemap index for read-side pruning — no new pruning mechanism.
6. Keep Python/Java bindings and the Spark/LanceDB connectors as thin wrappers over Rust.

### Non-goals (for the first iteration)

- Replacing vector indexes or ANN clustering (IVF/PQ) — unrelated subsystem.
- Automatic, learned clustering-key selection (Databricks "CLUSTER BY AUTO"). Keys are explicit.
- Cross-fragment global sort guarantees. Clustering is statistical locality, not a total order.
- Changing any *stable* on-disk format contract. Clustering metadata is additive.

## 3. Design overview

Three cooperating pieces, all in Rust core:

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
   │ explicit cols │        │  ClusteringCompactionPlanner +       │
   │ or declaration│──────► │  dedicated clustering task execution │
   └───────────────┘        └──────────────────────────────────────┘
        │                                   │
        │ produce clustered fragments,      │ re-sort selected fragments by
        │ optionally seed existing index    │ clustering key
        ▼                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Separately declared zonemap index → scan-time zone pruning   │
   └─────────────────────────────────────────────────────────────┘
```

- **Clustering spec** is stored in `Manifest::config` as a complete set of additive config keys.
  The `lance.clustering.` namespace is atomic: no key in that namespace means the declaration is
  absent, while any key in it requires all four known keys to be present and valid in the resulting
  manifest. Partial or malformed declarations are rejected. The clustering feature flag gates
  readers and writers that predate this contract.
- **`SpaceFillingEncoder`** turns N key columns into a single 1-D ordering value using a
  space-filling curve. Sorting by this value gives multi-dimensional locality.
- **The write path** optionally sorts each incoming batch stream by the clustering value before
  fragment rolling. On append, the normal index-seed path may also seed a separately declared
  zonemap index when that index has seeding enabled.
- **A clustering-aware compaction planner** selects under-clustered fragments and rewrites them
  through a reorder-enabled path. Optional source budgets make this incremental across runs; the
  defaults impose no source-volume limit.

## 4. Clustering key encoder

The implementation lives in `rust/lance-index/src/clustering/` and does not reuse the geo
`HilbertSorter`, which is 2D and bbox-specific. `SpaceFillingEncoder` maps N key columns into a
single fixed-size binary ordering value; `ClusteringCurve` selects Z-order or Hilbert encoding.

The current encoder accepts only top-level columns with these Arrow types: `Boolean`,
`Int8`/`Int16`/`Int32`/`Int64`, `UInt8`/`UInt16`/`UInt32`/`UInt64`, `Float32`, and `Float64`.
Strings, binary, dictionaries, temporal values, decimals, nested values, and other types are
rejected.

Encoding behavior:

1. **Normalize each key column to unsigned integer "bits."** Integers use an order-preserving
   signed or unsigned mapping, floats use the standard order-preserving IEEE-754 bit flip, and
   booleans map to two ordered values. Nulls map to the maximum coordinate.
2. **Quantize** each column to a fixed bit width `b` (e.g. 16 or 20 bits/column), tunable.
3. **Interleave** (Z-order) or apply the Hilbert transform across dimensions to produce the
   ordering value.

The current encoder uses fixed-domain most-significant-bit truncation so the same value receives
the same coordinate in every input batch. This can collapse small ranges of wide integer and
floating-point types at low bit widths. Null maps to the maximum coordinate in its own dimension;
a multi-dimensional curve does not promise row-level `NULLS LAST`.

Precision/whitening (per-column bit width, handling skew) is an open question — see §10.

## 5. Format / metadata changes (additive, no stable-format break)

The complete desired layout is stored in `manifest.config`. It deliberately does not reuse the
existing `lance-schema:unenforced-clustering-key:position` marker: that stable marker asserts an
already-achieved physical ordering for query-engine optimizations, while liquid clustering is a
policy to which existing fragments may only converge incrementally.

| Key | Type | Meaning |
|---|---|---|
| `lance.clustering.columns` | JSON array of strings | Ordered clustering-key columns |
| `lance.clustering.curve` | string (`hilbert` \| `zorder`) | Space-filling curve |
| `lance.clustering.version` | int | Layout version used to detect under-clustered fragments |
| `lance.clustering.bits_per_dim` | int | Per-column quantization bit width |

Rationale: `config` is already an additive `HashMap<String,String>` mutated through the existing
`UpdateConfig` operation (`rust/lance/src/dataset/metadata.rs`), so changing the desired layout is a
cheap metadata-only commit. The complete declaration is updated atomically. Any change to columns,
curve, or bit width must increase the layout version so existing fragments become eligible for
incremental reclustering. Manifests carrying the declaration set both the clustering reader and
writer feature flags, so older implementations do not open the dataset while ignoring the new
invariants.

**Per-fragment "clustered-at" marker.** To make re-clustering incremental we must know which
fragments are already clustered under the current spec. The implemented marker records the
active declared clustering `version` that governed a fragment's write. A fragment is
"under-clustered" unless its recorded version exactly equals the table's current
`lance.clustering.version`; an absent, older, or unexpected future stamp is selected. A one-shot
sort on an undeclared dataset therefore remains unstamped.

The protobuf `DataFragment.clustering_version` field (number 12, `0` = unset) persists this stamp.
In Rust, decoded stamps are kept in a private `Manifest` sidecar aligned positionally with
`Manifest::fragments`; they are not fields on the public `Fragment` type. Manifest serialization
writes the sidecar values back to `DataFragment.clustering_version`. A reader-and-writer feature
flag fences older readers and writers whenever an active clustering declaration or any fragment
stamp is present.

Field 12 is a stable scalar encoding whose zero value means "unset"; active clustering versions
must therefore be positive. That zero sentinel cannot later be reinterpreted as a valid version. A
future format needing explicit presence or version zero would require a new field or encoding.

A future quality heuristic could additionally infer overlap from zonemap ranges, but it would not
replace the version stamp as the authoritative current-layout predicate.

## 6. Write path

Rust keeps clustering out of `WriteParams`. Normal dataset writes accept explicit column names
through `InsertBuilder::with_cluster_by_columns`; multi-fragment writes can use
`FragmentCreateBuilder::with_cluster_by_columns`. The builders resolve those columns to a private
`ClusteringSpec` passed separately to the internal writer.

Effective write behavior:

1. **Resolve** the spec: `InsertBuilder` resolves explicit columns, or inherits the dataset's
   declared spec when columns are omitted, for both append and overwrite. Overwrite inherits the
   declaration because overwrite commits preserve dataset config. On a dataset with an active
   declaration, explicit columns must exactly match it. On a new or undeclared dataset, explicit
   columns request a one-shot sort and do not declare or stamp a persistent layout. Low-level
   multi-fragment writes must pass explicit columns through `FragmentCreateBuilder`; omitting them
   does not inherit the declaration.
2. **Sort** the write's batch stream by the encoded clustering value before fragments are written.
   *(Implemented: `lance_index::clustering::cluster_sort_stream`, a `SortExec` over a hidden
   `FixedSizeBinary` ordering column, spilling to disk for large inputs.)* Oversized input batches
   are split and deep-copied before they reach the sort because DataFusion cannot spill one batch
   that already exceeds its memory reservation; a single row larger than the cap is rejected. For
   the streaming writer
   this is a sort within the write unit; global ordering across concurrent writers is not guaranteed
   (that is what re-clustering converges).
3. **Seed an existing zonemap when configured.** Declaring clustering does not automatically
   declare or build a zonemap. On append to an existing V2 dataset, the normal `IndexSeedWriter`
   path can seed an already-declared zonemap index when that index has `use_seeds` enabled. A
   clustering column without such an index receives no automatic zonemap declaration or seed.
   Automatic deferred zonemap declaration from the clustering spec is future work.
4. **Stamp** each committed fragment governed by the active declaration with the current
   clustering version (§5). The write transaction carries the resolved version internally until
   final fragment IDs are assigned; manifest construction then reconciles the private sidecar.

Connectors keep their current role: Spark's `RequiresDistributionAndOrdering` can still pre-sort at
the engine for scale; the core sort is the correctness backstop when the engine does not.

## 7. Compaction / re-clustering

This is the part that must live in core because ordinary `compact_files` deliberately does not
reorder. The ordinary `compact_files` and `plan_compaction` entry points remain order-preserving;
neither dispatches to clustering. `CompactionMode` also remains the nonbreaking three-variant enum
(`Reencode`, `TryBinaryCopy`, and `ForceBinaryCopy`) used by ordinary compaction.

**Dedicated clustering operation. (implemented)** `ClusteringCompactionPlanner` and
`plan_clustering_compaction` produce a `ClusteringCompactionPlan`. Its tasks are executed as
`ClusteringCompactionTask` values, which resolve the declaration at the task's immutable read
version and re-sort rows through the private clustering rewrite strategy before
`write_fragments_internal_with_clustering` runs `cluster_sort_stream`. Each task yields a
`ClusteringRewriteResult`; `commit_clustering_compaction` validates and commits those results, then
records the task's clustering version after final fragment IDs are assigned. Clustering requires
the existing `CompactionMode::Reencode` behavior and does not use binary copy. Output still respects
`target_rows_per_fragment` / `max_bytes_per_file`.

`compact_files_with_clustering` is the single-process convenience entry point over that same flow:
it plans with `plan_clustering_compaction`, executes the dedicated tasks, and commits with
`commit_clustering_compaction`. Distributed callers use those three stages directly.

**Incremental planner. (implemented)** The dedicated `ClusteringCompactionPlanner` selects
*under-clustered* fragments — those whose
manifest-sidecar stamp does not equal the dataset's current `ClusteringSpec::version` — groups
adjacent ones up to `target_rows_per_fragment`, and honors the optional `max_source_fragments`,
`max_source_rows`, and `max_source_bytes` budgets. All three source budgets default to `None`, so
clustering compaction is unbounded by source volume unless the caller or dataset config sets at
least one. When several are set, all are hard upper bounds: planning stops before the next eligible
fragment would exceed any one. Rows mean live rows; bytes include source data and overlay files but
exclude separately stored Blob v2 payloads. If the first eligible fragment exceeds a configured
budget, the plan is empty and the budget must be raised. An already-clustered fragment breaks
adjacency so it is left untouched, and a second run at the same version is a no-op.

**Distributed operation separation.** `ClusteringCompactionTask` and
`ClusteringRewriteResult` use dedicated, version-tagged serialized envelopes rather than the
ordinary `CompactionTask` and `RewriteResult` payloads. The tags prevent accidental cross-routing
between order-preserving and row-reordering work. Task execution validates its read snapshot and
clustering declaration; `commit_clustering_compaction` requires one read version and clustering
version across all results and checks that version against the declaration at the task snapshot
before reserving fragment IDs. These checks provide operation separation and consistency
validation; they are not cryptographic proof against a caller that already has direct transaction
or file authority.

*Not yet done:* pulling in overlapping already-clustered fragments so new data merges into the
right place (the planner currently only reclusters under-clustered fragments among themselves), and
per-task combined-key-range grouping. These refine clustering quality and are follow-ups.

**Changing keys without full rewrite.** Bumping the clustering version while changing the columns
or other layout parameters marks all existing fragments as under-clustered *lazily*; they are
re-clustered by later `OPTIMIZE` runs. With a source budget configured, a large table can converge
over several bounded runs. With the default unbounded source budgets, one run may select all
eligible fragments. Old data stays readable throughout.

**Interaction with existing compaction machinery. (partial)** Ordinary compaction preserves row
order, so its positional old→new row mapping (for index remap and stable-row-id rechunk) holds.
Reclustering *reorders* rows, which breaks that positional assumption. Rather than silently
corrupt row ids or a secondary index, the dedicated clustering path currently **rejects** datasets
that use stable row ids or carry a remappable secondary index (drop the index, recluster, rebuild),
as well as `defer_index_remap=true`. Carrying row identity through the sort and rebuilding the
mapping from the sorted order is the follow-up that lifts this restriction.

## 8. Read path

No new pruning mechanism is introduced. When a zonemap index has been declared on a clustering
column, the existing scan-time pruning can exploit the resulting value locality. Declaring
clustering alone does not declare, seed, or build a zonemap, so clustered data without a separately
created zonemap receives no zonemap-based pruning benefit. Automatic deferred zonemap declaration
from the clustering spec, plus query-planning exposure for cost estimates and connector pushdown,
remain future work.

## 9. API surface (thin bindings)

Centralize resolution and validation in Rust while keeping the binding-level names consistent
(`cluster_by`, `clustering`).

- **Rust:** `InsertBuilder::with_cluster_by_columns` and
  `FragmentCreateBuilder::with_cluster_by_columns` for explicit write columns;
  `Dataset::set_clustering` / `Dataset::clustering_spec` / `Dataset::clear_clustering` to
  declare/read/drop the config-backed spec. Reclustering uses `ClusteringCompactionPlanner`,
  `plan_clustering_compaction`, `ClusteringCompactionTask::execute`,
  `commit_clustering_compaction`, or the single-process `compact_files_with_clustering` helper.
  `CompactionMode` has no clustering variant; `CompactionRequest::Clustering` is the separate
  request discriminator and carries options normalized to `CompactionMode::Reencode`.
  `WriteParams` contains no clustering field. *(Implemented.)*
- **Python: (implemented)** `write_dataset(..., cluster_by=["a", "b"])`;
  `dataset.set_clustering(columns, *, curve, version, bits_per_dim)` /
  `dataset.clustering_spec()` (returns a dict or `None`) / `dataset.clear_clustering()`;
  `dataset.optimize.compact_files(compaction_mode="cluster")`. The string `"cluster"` is a
  binding-level request discriminator, not a Rust `CompactionMode`: Python normalizes the core mode
  to `Reencode` and dispatches execute, plan, and commit to the dedicated clustering APIs. The
  binding marshals write columns into the Rust builder, where the declaration is resolved and
  authoritatively validated.
- **Java/JNI: (implemented)** `WriteParams.Builder.withClusterBy(List<String>)` remains the public
  Java binding surface and JNI forwards its column list to the Rust write builder;
  `Dataset.setClustering(ClusteringSpec)` / `Dataset.getClusteringSpec()` /
  `Dataset.clearClustering()`; `CompactionMode.CLUSTER`. Java's `CLUSTER` value is likewise a
  binding-level request discriminator: the distributed `Compaction` API passes it to JNI, which
  maps it to `CompactionRequest::Clustering`, normalizes the core mode to `Reencode`, and uses the
  dedicated clustering plan, tagged task/result, and commit path. It is not a Rust
  `CompactionMode` variant; the Java `Dataset.compact` convenience method dispatches the same
  binding-level value to `compact_files_with_clustering`.
  `ClusteringSpec` / `ClusteringCurve` are thin value types mirroring the Rust/Python shape.
- **Spark connector:** map SQL `CLUSTER BY (a, b)` to the persisted spec; `OPTIMIZE` triggers the
  incremental recluster; keep `LanceScanBuilder` pruning as-is. *(Not yet done — connector lives
  outside this repo.)*

## 10. Open questions

1. **Precision and curve evaluation.** Hilbert is the current default and Z-order is the cheaper
   alternative. Future benchmarking can guide whether defaults or per-column bit widths should
   change and how to handle skewed or high-cardinality columns (whitening / rank normalization).
2. **Alternative quality metrics.** The implemented fragment-version marker is the authoritative
   under-clustered predicate. A zonemap-overlap heuristic could later supplement it when deciding
   whether already-current fragments would benefit from another rewrite.
3. **Merge scope during recluster.** How aggressively to pull in overlapping already-clustered
   fragments — trades write amplification against clustering quality (the classic incremental-
   clustering cost knob).
4. **Zonemap `rows_per_zone` alignment.** Clustering quality is only useful if zone granularity is
   fine enough; do we couple the default zone size to the clustering config?
5. **Concurrency.** Recluster is a rewrite; confirm conflict resolution with concurrent
   appends/updates matches existing compaction guarantees (it should, since it reuses that path).

## 11. Phased delivery

- **Phase 0 — spec plumbing. (implemented)** `ClusteringSpec` type, the complete declaration in
  `lance.clustering.*` config, and `Dataset::set_clustering` /
  `clustering_spec` / `clear_clustering`. No layout behavior change yet.
- **Phase 1 — encoder. (implemented)** General multi-column Z-order + Hilbert `SpaceFillingEncoder`
  in `lance-index::clustering`, with tests (order preservation, Hilbert adjacency, null handling,
  mixed supported types, bit-budget validation).
- **Phase 2 — write-side clustering. (implemented)** Rust write builders accept explicit column
  lists separately from `WriteParams`, resolve the private spec from the dataset on append and
  overwrite, and sort with `cluster_sort_stream`. Commits stamp fragments governed by the active
  declaration in the private manifest sidecar. An explicit one-shot sort on a new or undeclared
  dataset is not stamped. Inline zonemap seeding from the clustering spec is deferred to Phase 4.
- **Phase 3 — incremental recluster. (implemented, partial)** The dedicated
  `ClusteringCompactionPlanner` / `plan_clustering_compaction` path selects under-clustered
  fragments and emits `ClusteringCompactionTask` values whose execution reorders rows by the
  clustering key. It honors any configured per-run source budgets (which are unbounded by default);
  `ClusteringRewriteResult` values are committed through `commit_clustering_compaction`, and the
  private per-fragment stamp is persisted through the optional
  `DataFragment.clustering_version` protobuf field and consumed here.
  Deferred: reclustering on datasets with stable row ids or a remappable index (currently rejected
  to avoid corrupting the positional row mapping), and pulling in overlapping already-clustered
  fragments for better merge quality.
- **Phase 4 — bindings & connectors. (implemented, partial)** Python and Java wrappers over the
  Rust core: `cluster_by` on the write path, `set_clustering` / `clustering_spec` /
  `clear_clustering` for declaration, and the `"cluster"` / `CLUSTER` binding-level request
  discriminators, which route to the dedicated clustering core APIs while ordinary
  `compact_files` / `plan_compaction` remain unchanged.
  Deferred: the Spark `CLUSTER BY` + `OPTIMIZE` connector integration (out of this repo) and
  automatic zonemap declaration.
- **Phase 5 — docs & benchmarks. (future work)** Data-skipping recall vs unclustered baseline;
  write/optimize overhead; key-change convergence.

## 12. Alignment with upstream

- Implements the space-filling curve `cluster_by` write work tracked by **#1434** and advances the
  **#1045** data-skipping EPIC (zonemap consumption).
- The incremental-recluster task echoes the "recluster as a compaction-like task" idea floated in
  **#6803**.
- Supersedes the not-planned **#952** by declaring keys once and maintaining incrementally instead
  of exposing partition/fragment mechanics.

Before expanding the remaining follow-up work, confirm its scope with maintainers on the relevant
tracking issues.
