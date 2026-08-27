# Design Draft: Liquid Clustering in Lance

Status: Draft / RFC
Tracking issues: lance-format/lance#1434 (space-filling curve `cluster_by` write param),
lance-format/lance#1045 (EPIC: statistics and data skipping), lance-format/lance#952 (implicit
partitioning, closed not-planned), lance-format/lance#6803 (recluster task for row-id healing).

> This is a proposal for adding Databricks-style "liquid clustering" to Lance. The goal is
> incremental, multi-dimensional data clustering that is declared once (`CLUSTER BY`), maintained
> automatically, and consumed by existing zonemap-based data skipping — without exposing
> fragment-level mechanics to users, and without forcing full-table rewrites.

## 1. Background and motivation

Lance today has the read-side and write-side building blocks for data skipping, but not the layer
that ties them into a self-maintaining clustering feature:

- **Read side (exists).** The zonemap scalar index records per-zone min/max/null over
  `rows_per_zone` rows and prunes zones at scan time
  (`rust/lance-index/src/scalar/zonemap.rs`, `ZoneMapIndex` at `zonemap.rs:109`). Zonemaps can be
  seeded inline during write via `IndexSeedWriter` (`ZoneMapSeedWriter` at `zonemap.rs:1457`, wired
  into `do_write_fragments_impl` at `rust/lance/src/dataset/write.rs:609` and `:656`).
- **Write side (partial).** `WriteParams` (`write.rs:270`) has no notion of clustering or sort
  order. Any clustering today must be arranged by the caller before data reaches the writer. In
  the Spark connector, `PARTITIONED BY` approximates single-key clustering: Spark shuffles/sorts by
  the key, and `LanceDataWriter` rolls a fresh fragment whenever the key changes.
- **Compaction (blocking gap).** `compact_files` explicitly does *not* reorder data — it
  "preserve[s] the insertion order of rows" (`optimize.rs:891`, `:898`), reads with
  `scan_in_order(true)` (`optimize.rs:1906`), and rewrites at `max_rows_per_file`
  (`optimize.rs:2272`). `CompactionMode` only distinguishes binary-copy variants
  (`optimize.rs:142`). There is no path that re-sorts data by a clustering key.
- **Space-filling curves (geo-only).** The only space-filling curve implementation is a 2D
  Hilbert sorter specialized for geometry bounding boxes (`HilbertSorter` at
  `rust/lance-index/src/scalar/rtree/sort/hilbert_sort.rs:26`). It is not a general multi-column
  encoder.

The core of liquid clustering — **incremental re-clustering that re-sorts data by a multi-column
key** — cannot be implemented in a connector (Spark/LanceDB) alone, because compaction lives in the
Lance core and deliberately does not reorder. This proposal therefore lands primarily in the Rust
core, with thin binding/connector surfaces on top.

### Why not "just Z-ORDER + partitioning"

Legacy `ZORDER BY` + Hive-style partitioning requires the user to pick partition columns up front,
suffers small-file / skew problems, and needs full rewrites to change layout. Liquid clustering
replaces both: clustering keys are declared once, changeable without rewriting existing data, and
maintained incrementally as data is written and during `OPTIMIZE`.

## 2. Goals and non-goals

### Goals

1. Declare clustering keys on a dataset (`CLUSTER BY (a, b, ...)`), persisted in table metadata.
2. Cluster newly written data by those keys using a multi-column space-filling curve
   (Z-order/Hilbert), so each fragment/file is value-coherent across all key columns.
3. Re-cluster incrementally: an `OPTIMIZE`-driven task that picks up under-clustered data and
   merges it into the clustered layout, touching a bounded amount of data per run (not a full-table
   rewrite).
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
                 └─────────────────────────────────────────────┘
                        ▲                    │
        declare/alter   │                    │ read by
       (UpdateConfig)   │                    ▼
   ┌───────────────┐    │   ┌──────────────────────────────────────┐
   │ Write path    │────┘   │ Compaction / recluster task          │
   │ WriteParams   │        │  ClusteringCompactionPlanner +       │
   │  .cluster_by  │──────► │  reorder-enabled rewrite_files       │
   └───────────────┘        └──────────────────────────────────────┘
        │                                   │
        │ produce clustered fragments,      │ re-sort selected fragments by
        │ seed zonemaps inline              │ clustering key, seed zonemaps
        ▼                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Zonemap scalar index (existing) → scan-time zone pruning     │
   └─────────────────────────────────────────────────────────────┘
```

- **Clustering spec** is stored in `manifest.config` (a `HashMap<String,String>` at
  `manifest.rs:96`), so it is additive and readable by every version without a format bump.
- **A `ClusteringKeyEncoder`** turns N key columns into a single 1-D ordering value using a
  space-filling curve. Sorting by this value gives multi-dimensional locality.
- **The write path** optionally sorts each incoming batch stream by the clustering value before
  fragment rolling, and seeds zonemaps on the key columns inline.
- **A clustering-aware compaction planner** selects under-clustered fragments and rewrites them
  through a reorder-enabled path, bounded per run for incrementality.

## 4. Clustering key encoder

New module, e.g. `rust/lance-index/src/clustering/` (or `rust/lance-core` if we want it dependency-
light). Do **not** reuse the geo `HilbertSorter`, which is 2D and bbox-specific.

```rust
/// Maps a row's clustering-key columns to a single ordering value that
/// preserves multi-dimensional locality.
pub trait ClusteringKeyEncoder: Send + Sync {
    /// Encode a batch of key columns into one ordering column (e.g. UInt64 or
    /// FixedSizeBinary for higher precision). Row i of the output orders row i.
    fn encode(&self, key_columns: &[ArrayRef]) -> Result<ArrayRef>;
}

pub enum ClusteringCurve {
    /// Z-order (Morton). Cheapest; discontinuities at power-of-two boundaries.
    ZOrder,
    /// Hilbert. Better locality, no large jumps; more compute. Default.
    Hilbert,
}
```

Encoding sketch:

1. **Normalize each key column to unsigned integer "bits."** For integers, bias to unsigned
   order-preserving form. For floats, use the standard order-preserving IEEE-754 bit flip. For
   strings/binary, take a fixed prefix; for dictionary/other, rank by value. Nulls sort to a
   reserved extreme.
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
| `lance.clustering.version` | int | Bumped on curve change or a forced recluster; used to detect under-clustered fragments |
| `lance.clustering.bits_per_dim` | int | Per-column quantization bit width |

Rationale: `config` is already an additive `HashMap<String,String>` mutated through the existing
`UpdateConfig` operation (`rust/lance/src/dataset/metadata.rs`), so changing the desired layout is a
cheap metadata-only commit and every reader tolerates unknown keys. The complete declaration is
updated atomically. Any change to columns, curve, or bit width must increase the layout version so
existing fragments become eligible for incremental reclustering.

**Per-fragment "clustered-at" marker.** To make re-clustering incremental we must know which
fragments are already clustered under the current spec. Options, in preference order:

- (A) Record the clustering `version` a fragment was written under in fragment metadata/properties.
  A fragment is "under-clustered" if its recorded version `<` the table's current
  `lance.clustering.version`, or absent. Preferred — precise and cheap.
- (B) Infer from zonemap zone overlap on the key columns (highly overlapping zones ⇒ poorly
  clustered). No format change, but heuristic and more expensive.

We chose (A) *(implemented)*: a new optional `DataFragment.clustering_version` proto field
(number 12, `0` = unset), surfaced as `Fragment::clustering_version: Option<u64>`. It is additive
and does not break reads: old readers ignore it, and fragments written before clustering was
declared simply read back as `None`. A writer-only feature flag prevents older writers from
dropping existing stamps or appending unstamped data while liquid clustering is active.

## 6. Write path

Extend `WriteParams` (`write.rs:270`) with an optional clustering directive:

```rust
pub struct WriteParams {
    // ...existing fields...
    /// When set, incoming data is sorted by the clustering key before fragment
    /// rolling, producing value-coherent fragments. Defaults to None.
    pub cluster_by: Option<ClusteringSpec>,
}
```

Behavior when `cluster_by` is `Some`:

1. **Resolve** the spec: explicit `WriteParams.cluster_by`, else (on create/append) the dataset's
   declared spec via `Dataset::clustering_spec()`. Overwrites do not inherit the existing spec.
   *(Implemented: `resolve_clustering_spec` in `write.rs`.)*
2. **Sort** the write's batch stream by the encoded clustering value before fragments are written.
   *(Implemented: `lance_index::clustering::cluster_sort_stream`, a `SortExec` over a hidden
   `FixedSizeBinary` ordering column, spilling to disk for large inputs.)* For the streaming writer
   this is a sort within the write unit; global ordering across concurrent writers is not guaranteed
   (that is what re-clustering converges).
3. **Seed zonemaps** on the key columns inline via the existing `IndexSeedWriter` path
   (`write.rs:656`), so a fresh clustered write also produces prunable stats without a separate
   index build. *(Not yet wired: seeds are still driven by existing zonemap indices; auto-seeding
   from the clustering spec is Phase 4.)*
4. **Stamp** each new fragment with the current clustering version (§5, option A). *(Implemented in
   Phase 3: `write_fragments_internal` sets `Fragment::clustering_version` from the resolved spec,
   backed by the new optional `DataFragment.clustering_version` proto field.)*

Connectors keep their current role: Spark's `RequiresDistributionAndOrdering` can still pre-sort at
the engine for scale; the core sort is the correctness backstop when the engine does not.

## 7. Compaction / re-clustering

This is the part that must live in core because `compact_files` deliberately does not reorder.

**Reorder-enabled rewrite. (implemented)** `CompactionMode::Cluster` makes the rewrite path re-sort
each task's rows by the clustering key instead of preserving insertion order: `rewrite_files` sets
`params.cluster_by` from the dataset's spec, so `write_fragments_internal` runs `cluster_sort_stream`
before writing and stamps the output fragments with the current clustering version. Binary copy is
disabled for `Cluster` mode. Output still respects `target_rows_per_fragment` / `max_bytes_per_file`.

**Incremental planner. (implemented)** `ClusteringCompactionPlanner` behind the pluggable
`CompactionPlanner` trait selects *under-clustered* fragments — those whose
`Fragment::clustering_version` does not equal the dataset's current
`ClusteringSpec::version` — groups adjacent ones up to `target_rows_per_fragment`, and honors the
existing `max_source_fragments` / `max_source_rows` / `max_source_bytes` budgets so a large table
converges over successive `optimize` runs rather than one full rewrite. An already-clustered
fragment breaks adjacency so it is left untouched, and a second run at the same version is a no-op.

*Not yet done:* pulling in overlapping already-clustered fragments so new data merges into the
right place (the planner currently only reclusters under-clustered fragments among themselves), and
per-task combined-key-range grouping. These refine clustering quality and are follow-ups.

**Changing keys without full rewrite.** Bumping the clustering version while changing the columns
or other layout parameters marks all existing fragments as under-clustered *lazily*; they are
re-clustered opportunistically over subsequent `OPTIMIZE` runs within budget, never in one forced
pass. Old data stays readable throughout.

**Interaction with existing compaction machinery. (partial)** Ordinary compaction preserves row
order, so its positional old→new row mapping (for index remap and stable-row-id rechunk) holds.
Reclustering *reorders* rows, which breaks that positional assumption. Rather than silently
corrupt row ids or a secondary index, `rewrite_files` currently **rejects** `Cluster` mode on
datasets that use stable row ids or carry a remappable secondary index (drop the index, recluster,
rebuild). Carrying row identity through the sort and rebuilding the mapping from the sorted order
is the follow-up that lifts this restriction.

## 8. Read path

No new pruning mechanism. Clustered layout + zonemaps on the key columns means the existing
zonemap scan-time pruning already skips non-matching zones/fragments. The read-side work is limited
to ensuring:

- A zonemap index exists on the clustering columns (auto-declared when `CLUSTER BY` is set — a
  cheap, deferred/metadata-only declaration; heavy build happens in `OPTIMIZE`).
- Query planning surfaces which columns are clustered so cost estimates and connector pushdown
  (e.g. Spark's `ZonemapFragmentPruner`) can rely on it.

## 9. API surface (thin bindings)

Centralize logic in Rust; keep parameter names identical across languages (`cluster_by`,
`clustering`).

- **Rust:** `WriteParams.cluster_by`; `Dataset::set_clustering` / `Dataset::clustering_spec` /
  `Dataset::clear_clustering` to declare/read/drop the config-backed spec;
  `CompactionMode::Cluster` on `CompactionOptions`. *(Implemented.)*
- **Python: (implemented)** `write_dataset(..., cluster_by=["a", "b"])`;
  `dataset.set_clustering(columns, *, curve, version, bits_per_dim)` /
  `dataset.clustering_spec()` (returns a dict or `None`) / `dataset.clear_clustering()`;
  `dataset.optimize.compact_files(compaction_mode="cluster")`. The bindings only marshal arguments
  — all validation lives in the Rust core.
- **Java/JNI: (implemented)** `WriteParams.Builder.withClusterBy(List<String>)`;
  `Dataset.setClustering(ClusteringSpec)` / `Dataset.getClusteringSpec()` /
  `Dataset.clearClustering()`; `CompactionMode.CLUSTER`. `ClusteringSpec` / `ClusteringCurve` are
  thin value types mirroring the Rust/Python shape.
- **Spark connector:** map SQL `CLUSTER BY (a, b)` to the persisted spec; `OPTIMIZE` triggers the
  incremental recluster; keep `LanceScanBuilder` pruning as-is. *(Not yet done — connector lives
  outside this repo.)*

## 10. Open questions

1. **Curve default & precision.** Hilbert (better locality) vs Z-order (cheaper). Per-column bit
   width and how to handle skewed / high-cardinality columns (whitening / rank normalization).
2. **Under-clustered detection.** Fragment version marker (§5-A) vs zonemap-overlap heuristic
   (§5-B). Marker is precise but adds a fragment property.
3. **Merge scope during recluster.** How aggressively to pull in overlapping already-clustered
   fragments — trades write amplification against clustering quality (the classic incremental-
   clustering cost knob).
4. **Zonemap `rows_per_zone` alignment.** Clustering quality is only useful if zone granularity is
   fine enough; do we couple the default zone size to the clustering config?
5. **Concurrency.** Recluster is a rewrite; confirm conflict resolution with concurrent
   appends/updates matches existing compaction guarantees (it should, since it reuses that path).
6. **Changing the clustering key without rewrite (goal 4).** The desired column set is stored in
   table config and can change together with a strictly increasing layout version; old fragments
   converge incrementally under the normal reclustering budget.

## 11. Phased delivery

- **Phase 0 — spec plumbing. (implemented)** `ClusteringSpec` type, the complete declaration in
  `lance.clustering.*` config, and `Dataset::set_clustering` /
  `clustering_spec` / `clear_clustering`. No layout behavior change yet.
- **Phase 1 — encoder. (implemented)** General multi-column Z-order + Hilbert `SpaceFillingEncoder`
  in `lance-index::clustering`, with tests (order preservation, Hilbert adjacency, null handling,
  mixed types, bit-budget validation).
- **Phase 2 — write-side clustering. (implemented)** `WriteParams.cluster_by` sorts the
  write stream by the space-filling curve (`cluster_sort_stream`), resolving the spec from the
  dataset on create/append and stamping written fragments with the clustering version. Inline
  zonemap seeding from the spec is deferred to Phase 4.
- **Phase 3 — incremental recluster. (implemented, partial)** `CompactionMode::Cluster` reorders a
  task's rows by the clustering key in `rewrite_files`; `ClusteringCompactionPlanner` selects
  under-clustered fragments and honors the per-run source budgets; the per-fragment
  `clustering_version` stamp (new optional `DataFragment` field) is written and consumed here.
  Deferred: reclustering on datasets with stable row ids or a remappable index (currently rejected
  to avoid corrupting the positional row mapping), and pulling in overlapping already-clustered
  fragments for better merge quality.
- **Phase 4 — bindings & connectors. (implemented, partial)** Python and Java wrappers over the
  Rust core: `cluster_by` on the write path, `set_clustering` / `clustering_spec` /
  `clear_clustering` for declaration, and the `"cluster"` / `CLUSTER` compaction mode (routed to
  `ClusteringCompactionPlanner` from the public `compact_files` / `plan_compaction` entrypoints).
  Deferred: the Spark `CLUSTER BY` + `OPTIMIZE` connector integration (out of this repo) and
  automatic zonemap declaration.
- **Phase 5 — docs & benchmarks.** Data-skipping recall vs unclustered baseline; write/optimize
  overhead; key-change convergence.

## 12. Alignment with upstream

- Directly implements the open, maintainer-assigned **#1434** ("space-filling curve `cluster_by`
  write param"), and advances the **#1045** data-skipping EPIC (zonemap consumption).
- The incremental-recluster task echoes the "recluster as a compaction-like task" idea floated in
  **#6803**.
- Supersedes the not-planned **#952** by declaring keys once and maintaining incrementally instead
  of exposing partition/fragment mechanics.

Before large implementation, confirm on #1434 that the maintainers still want this and agree on the
API shape (write-param + config vs. a different surface), since the umbrella #952 was closed as
not-planned.
