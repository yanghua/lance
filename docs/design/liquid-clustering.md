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

Precision/whitening (per-column bit width, handling skew) is an open question — see §10.

## 5. Format / metadata changes (additive, no stable-format break)

Persistence is split between the schema and the table config:

**Clustering columns → per-field schema markers.** The clustering-key columns (and their order)
reuse the existing *unenforced clustering key* mechanism: each key field carries the
`lance-schema:unenforced-clustering-key:position` metadata marker (1-based), read back through
`Schema::unenforced_clustering_key()`. This already exists in the schema layer but had no layout
consumer; liquid clustering becomes that consumer. Reusing it avoids a second, parallel column list
and inherits its immutability (`manifest_build.rs` rejects changing the key once set).

**Tuning parameters → `manifest.config`.** The remaining knobs that the schema marker cannot
express are reserved config keys:

| Key | Type | Meaning |
|---|---|---|
| `lance.clustering.curve` | string (`hilbert` \| `zorder`) | Space-filling curve |
| `lance.clustering.version` | int | Bumped on curve change or a forced recluster; used to detect under-clustered fragments |
| `lance.clustering.bits_per_dim` | int | Per-column quantization bit width |

Rationale: `config` is already an additive `HashMap<String,String>` mutated through the existing
`UpdateConfig` operation (`rust/lance/src/dataset/metadata.rs`), so changing tuning is a cheap
metadata-only commit and every reader tolerates unknown keys. `ClusteringSpec` is the in-memory
bundle of both surfaces (`ClusteringSpec::from_parts(columns, config)` /
`ClusteringSpec::to_config()`).

> Consequence of reusing the immutable marker: the **column set is fixed once declared**. Changing
> curve/bits/version is always allowed (that is what drives a recluster), but changing *which*
> columns cluster is rejected. Full "change the clustering key without rewrite" (§2 goal 4) is
> therefore deferred until the marker's immutability is relaxed — tracked as an open question.

**Per-fragment "clustered-at" marker.** To make re-clustering incremental we must know which
fragments are already clustered under the current spec. Options, in preference order:

- (A) Record the clustering `version` a fragment was written under in fragment metadata/properties.
  A fragment is "under-clustered" if its recorded version `<` the table's current
  `lance.clustering.version`, or absent. Preferred — precise and cheap.
- (B) Infer from zonemap zone overlap on the key columns (highly overlapping zones ⇒ poorly
  clustered). No format change, but heuristic and more expensive.

We propose (A), adding an optional fragment property; this is additive to fragment metadata and does
not alter any stable file format.

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

1. **Resolve** the spec: explicit `WriteParams.cluster_by`, else the dataset's persisted
   `lance.clustering.*` config (append inherits the table's declared keys).
2. **Sort** each write's batch stream by the encoded clustering value. For the streaming writer this
   is a sort within the write unit; global ordering across concurrent writers is not guaranteed
   (that is what re-clustering converges).
3. **Seed zonemaps** on the key columns inline via the existing `IndexSeedWriter` path
   (`write.rs:656`), so a fresh clustered write also produces prunable stats without a separate
   index build.
4. **Stamp** each new fragment with the current clustering version (§5, option A).

Connectors keep their current role: Spark's `RequiresDistributionAndOrdering` can still pre-sort at
the engine for scale; the core sort is the correctness backstop when the engine does not.

## 7. Compaction / re-clustering

This is the part that must live in core because `compact_files` deliberately does not reorder.

**Reorder-enabled rewrite.** Add a clustering mode so the rewrite path sorts by the clustering key
instead of reading in insertion order:

- Add `CompactionMode::Cluster` (extending the enum at `optimize.rs:142`), or a dedicated
  `recluster` entry point. When active, `rewrite_files` (`optimize.rs:2160`) sorts the merged input
  stream by the encoded clustering value rather than using `scan_in_order(true)`
  (`optimize.rs:1906`), and seeds zonemaps on output.
- Output still respects `target_rows_per_fragment` / `max_bytes_per_file`, so files stay
  right-sized (`optimize.rs:2272`).

**Incremental planner.** Implement a `ClusteringCompactionPlanner` behind the existing pluggable
`CompactionPlanner` trait (`optimize.rs:698`). It:

1. Selects under-clustered fragments (version marker from §5), plus optionally a bounded set of
   already-clustered fragments whose key ranges overlap them (so the new data merges into the right
   place rather than forming an isolated clustered island).
2. Respects the existing per-run budgets already on `CompactionOptions`
   (`max_source_fragments` / `max_source_rows` / `max_source_bytes`) for incrementality.
3. Groups selected fragments into tasks whose combined key range is compact, then rewrites each
   task through the reorder path.

**Changing keys without full rewrite.** Bumping `lance.clustering.version` marks all existing
fragments as under-clustered *lazily*; they are re-clustered opportunistically over subsequent
`OPTIMIZE` runs within budget, never in one forced pass. Old data stays readable throughout.

**Interaction with existing compaction machinery.** Re-clustering rewrites fragments, so it flows
through the same fragment-reuse-index / row-address remapping the current compaction already handles
(`optimize.rs:2191`+). No new remap semantics; stable row ids and index remapping behave as they do
for ordinary compaction.

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
  `Dataset::clear_clustering` to declare/read/drop the spec (columns via schema markers, tuning via
  config); `CompactionMode::Cluster` / a `recluster` option on `CompactionOptions`.
- **Python:** `write_dataset(..., cluster_by=[...])`, `dataset.alter_clustering(columns=[...])`,
  `dataset.optimize.compact_files(recluster=True)` (or auto when a spec is set).
- **Java/JNI:** mirror the Python shape (per cross-language contract in `python/AGENTS.md` /
  `java/AGENTS.md`).
- **Spark connector:** map SQL `CLUSTER BY (a, b)` to the persisted spec; `OPTIMIZE` triggers the
  incremental recluster; keep `LanceScanBuilder` pruning as-is.

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
6. **Changing the clustering key without rewrite (goal 4).** The column set is currently persisted
   via the *immutable* unenforced-clustering-key markers, so today only curve/bits/version can
   change on an existing dataset — not the columns. Delivering full key changes needs the marker's
   immutability relaxed (or a separate mutable column list). Deferred pending maintainer input.

## 11. Phased delivery

- **Phase 0 — spec plumbing. (implemented)** `ClusteringSpec` type, `lance.clustering.*` tuning
  config, columns via reused schema markers, and `Dataset::set_clustering` /
  `clustering_spec` / `clear_clustering`. No layout behavior change yet.
- **Phase 1 — encoder. (implemented)** General multi-column Z-order + Hilbert `SpaceFillingEncoder`
  in `lance-index::clustering`, with tests (order preservation, Hilbert adjacency, null handling,
  mixed types, bit-budget validation).
- **Phase 2 — write-side clustering.** `WriteParams.cluster_by` sorts + seeds zonemaps + stamps
  version.
- **Phase 3 — incremental recluster.** `ClusteringCompactionPlanner` + reorder-enabled
  `rewrite_files` (`CompactionMode::Cluster`), reusing per-run budgets.
- **Phase 4 — bindings & connectors.** Python/Java wrappers; Spark `CLUSTER BY` + `OPTIMIZE`
  integration; auto zonemap declaration.
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
