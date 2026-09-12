// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stream sorting by a clustering-key space-filling curve.
//!
//! [`cluster_sort_stream`] takes a record-batch stream and the clustering-key
//! column names, appends a hidden ordering column computed by
//! [`SpaceFillingEncoder`], sorts the stream by it (spilling to disk when
//! large), and drops the column again. The write path uses this so a clustered
//! write lays value-coherent rows next to each other on disk.

use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeBinaryArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, SortOptions};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{ColumnarValue, ScalarUDF, Signature, Volatility};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result as DataFusionResult};
use datafusion_expr::{ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_physical_expr::expressions::Column as DFColumn;
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use futures::future::BoxFuture;
use futures::{FutureExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_datafusion::exec::{
    HardCapBatchSizeExec, LanceExecutionOptions, OneShotExec, execute_plan, provider_to_stream,
};
use lance_datafusion::spill::spilling_table_provider;

use super::{ClusteringSpec, QuantileModelBuilder, SpaceFillingEncoder};

/// Name of the transient column holding the space-filling-curve ordering value.
const CLUSTERING_ORDER_FIELD: &str = "__lance_clustering_order";

/// Maximum size of a batch passed to DataFusion's sort operator.
///
/// DataFusion cannot spill a single batch that exceeds its memory pool, so
/// oversized batches must be split before sorting. This is also the cap used
/// by merge-insert's sorting path for the default memory pool. Smaller pools
/// scale the cap down further in [`sort_batch_byte_limit`].
const MAX_BATCH_BYTES: usize = 25 * 1024 * 1024;

/// Leave enough of the memory pool for DataFusion's sort bookkeeping.
///
/// Sorting reserves up to one third of the pool for its spill/merge phase and
/// estimates each input batch at roughly twice its Arrow memory size. The
/// one-eighth fraction follows the pool-relative merge-insert precedent and
/// leaves room for those reservations, sorted copies, and operator overhead.
fn sort_batch_byte_limit(memory_pool_size: u64) -> usize {
    let pool_fraction = usize::try_from(memory_pool_size / 8).unwrap_or(usize::MAX);
    MAX_BATCH_BYTES.min(pool_fraction).max(1)
}

/// Estimate the memory added while computing the clustering order column.
///
/// In addition to the final fixed-size-binary value, the encoder holds one
/// `u64` coordinate per clustering column and row. Including that scratch
/// space keeps projection itself within the same conservative budget used for
/// the batch eventually passed to the sort.
fn projection_bytes_per_row(num_columns: usize, order_width: usize) -> Result<usize> {
    num_columns
        .checked_mul(std::mem::size_of::<u64>())
        .and_then(|coordinate_bytes| coordinate_bytes.checked_add(order_width))
        .ok_or_else(|| Error::invalid_input("clustering projection memory estimate overflowed"))
}

fn estimated_projection_bytes(batch: &RecordBatch, added_bytes_per_row: usize) -> usize {
    batch
        .num_rows()
        .saturating_mul(added_bytes_per_row)
        .saturating_add(batch.get_array_memory_size())
        .saturating_add(std::mem::size_of::<FixedSizeBinaryArray>())
}

/// Lazily split one source batch before projection.
///
/// The original batch is retained only while rows remain, and at most one
/// copied candidate chunk is retained at a time. This avoids replacing one
/// large input allocation with a vector containing every copied chunk.
struct ProjectionBatchSplitter {
    batch: Option<RecordBatch>,
    next_offset: usize,
    rows_per_chunk: usize,
    max_bytes: usize,
    added_bytes_per_row: usize,
}

impl ProjectionBatchSplitter {
    fn new(batch: RecordBatch, max_bytes: usize, added_bytes_per_row: usize) -> Self {
        let estimated_bytes = estimated_projection_bytes(&batch, added_bytes_per_row);
        let rows_per_chunk = if estimated_bytes <= max_bytes || batch.num_rows() <= 1 {
            batch.num_rows().max(1)
        } else {
            ((max_bytes as u128 * batch.num_rows() as u128) / estimated_bytes as u128)
                .max(1)
                .min((batch.num_rows() - 1) as u128) as usize
        };
        Self {
            batch: Some(batch),
            next_offset: 0,
            rows_per_chunk,
            max_bytes,
            added_bytes_per_row,
        }
    }

    fn next_batch<F>(&mut self, on_materialized: &mut F) -> DataFusionResult<Option<RecordBatch>>
    where
        F: FnMut(),
    {
        let Some(batch) = self.batch.as_ref() else {
            return Ok(None);
        };
        let estimated_bytes = estimated_projection_bytes(batch, self.added_bytes_per_row);
        if self.next_offset == 0 && estimated_bytes <= self.max_bytes {
            return Ok(self.batch.take());
        }
        if batch.num_rows() == 0 {
            return Ok(self.batch.take());
        }

        let remaining_rows = batch.num_rows() - self.next_offset;
        let mut chunk_rows = self.rows_per_chunk.min(remaining_rows);
        loop {
            let end_offset = self.next_offset + chunk_rows;
            let start_index = u64::try_from(self.next_offset).map_err(|_| {
                Error::invalid_input("clustering projection row offset exceeds u64")
            })?;
            let end_index = u64::try_from(end_offset).map_err(|_| {
                Error::invalid_input("clustering projection row offset exceeds u64")
            })?;
            let indices = UInt64Array::from((start_index..end_index).collect::<Vec<_>>());
            let copied = arrow_select::take::take_record_batch(batch, &indices)?;
            let copied_estimated_bytes =
                estimated_projection_bytes(&copied, self.added_bytes_per_row);
            if copied_estimated_bytes <= self.max_bytes {
                on_materialized();
                self.next_offset = end_offset;
                if self.next_offset == batch.num_rows() {
                    self.batch.take();
                }
                return Ok(Some(copied));
            }
            if chunk_rows == 1 {
                self.batch.take();
                return Err(Error::invalid_input(format!(
                    "a single row is estimated to require {copied_estimated_bytes} bytes during \
                     clustering projection, which exceeds the maximum allowed batch size of {} \
                     bytes",
                    self.max_bytes
                ))
                .into());
            }

            // Variable-width rows can make the average-based first attempt too
            // large. Drop it, reduce the row count, and retry without retaining
            // multiple copied chunks.
            chunk_rows = ((self.max_bytes as u128 * chunk_rows as u128)
                / copied_estimated_bytes as u128)
                .max(1)
                .min((chunk_rows - 1) as u128) as usize;
        }
    }
}

fn cap_projection_input(
    data: SendableRecordBatchStream,
    max_bytes: usize,
    added_bytes_per_row: usize,
) -> SendableRecordBatchStream {
    cap_projection_input_with_materialization_callback(data, max_bytes, added_bytes_per_row, || {})
}

fn cap_projection_input_with_materialization_callback<F>(
    data: SendableRecordBatchStream,
    max_bytes: usize,
    added_bytes_per_row: usize,
    on_materialized: F,
) -> SendableRecordBatchStream
where
    F: FnMut() + Send + 'static,
{
    let schema = data.schema();
    let state = (data, None::<ProjectionBatchSplitter>, on_materialized);
    let capped = futures::stream::try_unfold(
        state,
        move |(mut data, mut splitter, mut on_materialized)| async move {
            loop {
                if let Some(active_splitter) = splitter.as_mut()
                    && let Some(batch) = active_splitter.next_batch(&mut on_materialized)?
                {
                    return Ok(Some((batch, (data, splitter, on_materialized))));
                }

                let Some(batch) = data.try_next().await? else {
                    return Ok(None);
                };
                splitter = Some(ProjectionBatchSplitter::new(
                    batch,
                    max_bytes,
                    added_bytes_per_row,
                ));
            }
        },
    );
    Box::pin(RecordBatchStreamAdapter::new(schema, capped))
}

/// Sort `data` so rows are ordered by the clustering-key space-filling curve.
///
/// All `spec.columns` must be present in the stream schema. The returned stream
/// yields the same schema as the input (the transient ordering column is
/// projected away). Sorting uses DataFusion's `SortExec`, which spills to disk
/// for inputs larger than memory.
pub fn cluster_sort_stream<'a>(
    data: SendableRecordBatchStream,
    spec: &'a ClusteringSpec,
) -> BoxFuture<'a, Result<SendableRecordBatchStream>> {
    cluster_sort_stream_with_options(
        data,
        spec,
        LanceExecutionOptions {
            use_spilling: true,
            ..Default::default()
        },
    )
    .boxed()
}

async fn cluster_sort_stream_with_options(
    data: SendableRecordBatchStream,
    spec: &ClusteringSpec,
    mut execution_options: LanceExecutionOptions,
) -> Result<SendableRecordBatchStream> {
    // Resolve an environment-configured pool exactly once so the cap and the
    // execution context cannot observe different values if the environment is
    // changed concurrently.
    let memory_pool_size = execution_options.mem_pool_size();
    execution_options.mem_pool_size = Some(memory_pool_size);
    let max_batch_bytes = sort_batch_byte_limit(memory_pool_size);

    let input_schema = data.schema();

    // Resolve each clustering column to its position in the input schema.
    let mut key_indices = Vec::with_capacity(spec.columns.len());
    for name in &spec.columns {
        let idx = input_schema.index_of(name).map_err(|_| {
            Error::invalid_input(format!(
                "clustering column {name:?} is not present in the data being written"
            ))
        })?;
        key_indices.push(idx);
    }
    let encoder = SpaceFillingEncoder::from_spec(spec)?;
    let order_width = encoder.output_width(spec.columns.len())?;
    let added_bytes_per_row = projection_bytes_per_row(spec.columns.len(), order_width)?;

    // A clustering stream is one-shot, but distribution-aware normalization
    // needs one pass to fit approximate ranks and a second pass to sort. Reuse
    // the existing memory-first replay spill so large writes remain bounded.
    // Split oversized source batches before sampling as well as sorting: fitting
    // materializes one u64 key per row and must obey the same memory cap as the
    // later projection.
    let replay_memory_limit = max_batch_bytes;
    let data = cap_projection_input(data, max_batch_bytes, added_bytes_per_row);
    let replay = spilling_table_provider(data, replay_memory_limit).await?;
    let mut sampling_stream = provider_to_stream(replay.clone()).await?;
    let mut quantile_builder = QuantileModelBuilder::new(key_indices.len())?;
    while let Some(batch) = sampling_stream.try_next().await? {
        let key_columns = key_indices
            .iter()
            .map(|index| batch.column(*index).clone())
            .collect::<Vec<_>>();
        quantile_builder.add(&key_columns)?;
    }
    let quantile_model = quantile_builder.finish();
    let data = provider_to_stream(replay).await?;

    // Projection 1: pass every input column through, then append the ordering
    // column computed from the key columns.
    let mut projection_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            (
                Arc::new(DFColumn::new(f.name(), idx)) as Arc<dyn PhysicalExpr>,
                f.name().clone(),
            )
        })
        .collect();

    let key_args: Vec<Arc<dyn PhysicalExpr>> = key_indices
        .iter()
        .map(|&idx| {
            Arc::new(DFColumn::new(input_schema.field(idx).name(), idx)) as Arc<dyn PhysicalExpr>
        })
        .collect();
    let encoder = encoder.with_quantile_model(quantile_model)?;
    let udf = ClusteringOrderUdf::new(encoder, spec.columns.len())?;
    debug_assert_eq!(order_width, udf.output_width);
    let source = Arc::new(OneShotExec::new(data));
    let order_expr = Arc::new(ScalarFunctionExpr::new(
        CLUSTERING_ORDER_FIELD,
        Arc::new(ScalarUDF::new_from_impl(udf)),
        key_args,
        Arc::new(ArrowField::new(
            CLUSTERING_ORDER_FIELD,
            DataType::FixedSizeBinary(order_width as i32),
            false,
        )),
        Arc::new(ConfigOptions::default()),
    )) as Arc<dyn PhysicalExpr>;
    let order_idx = projection_exprs.len();
    projection_exprs.push((order_expr, CLUSTERING_ORDER_FIELD.to_string()));

    let with_order = Arc::new(ProjectionExec::try_new(
        projection_exprs,
        source as Arc<dyn ExecutionPlan>,
    )?);

    // Sort by the ordering column.
    let sort_expr = PhysicalSortExpr {
        expr: Arc::new(DFColumn::new(CLUSTERING_ORDER_FIELD, order_idx)),
        options: SortOptions::default(),
    };
    // Validate the actual projected size as well. The source-side estimate
    // includes the known ordering and scratch allocations, while this catches
    // Arrow allocation overhead or estimator drift before the sort.
    let capped_projection = Arc::new(HardCapBatchSizeExec::new(
        with_order as Arc<dyn ExecutionPlan>,
        max_batch_bytes,
    ));
    let sorted = Arc::new(SortExec::new(
        [sort_expr].into(),
        capped_projection as Arc<dyn ExecutionPlan>,
    ));

    // Projection 2: drop the transient ordering column, restoring input schema.
    let drop_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            (
                Arc::new(DFColumn::new(f.name(), idx)) as Arc<dyn PhysicalExpr>,
                f.name().clone(),
            )
        })
        .collect();
    let restored = Arc::new(ProjectionExec::try_new(
        drop_exprs,
        sorted as Arc<dyn ExecutionPlan>,
    )?);

    let stream = execute_plan(restored, execution_options)?;
    Ok(stream)
}

/// A DataFusion scalar UDF that encodes clustering-key columns into a single
/// space-filling-curve ordering value.
#[derive(Debug, Clone)]
struct ClusteringOrderUdf {
    signature: Signature,
    encoder: SpaceFillingEncoder,
    num_columns: usize,
    output_width: usize,
}

impl ClusteringOrderUdf {
    fn new(encoder: SpaceFillingEncoder, num_columns: usize) -> Result<Self> {
        let output_width = encoder.output_width(num_columns)?;
        Ok(Self {
            signature: Signature::any(num_columns, Volatility::Immutable),
            encoder,
            num_columns,
            output_width,
        })
    }
}

impl PartialEq for ClusteringOrderUdf {
    fn eq(&self, other: &Self) -> bool {
        self.signature == other.signature
            && self.encoder == other.encoder
            && self.num_columns == other.num_columns
            && self.output_width == other.output_width
    }
}

impl Eq for ClusteringOrderUdf {}

impl std::hash::Hash for ClusteringOrderUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.signature.hash(state);
        self.encoder.hash(state);
        self.num_columns.hash(state);
        self.output_width.hash(state);
    }
}

impl ScalarUDFImpl for ClusteringOrderUdf {
    fn name(&self) -> &str {
        CLUSTERING_ORDER_FIELD
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::FixedSizeBinary(self.output_width as i32))
    }

    fn invoke_with_args(&self, func_args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        let num_rows = func_args.number_rows;
        let key_columns: Vec<ArrayRef> = func_args
            .args
            .into_iter()
            .map(|arg| arg.into_array(num_rows))
            .collect::<DataFusionResult<_>>()?;
        let encoded = self
            .encoder
            .encode(&key_columns)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(ColumnarValue::Array(encoded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::ClusteringCurve;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{BooleanArray, Int32Array, Int64Array, LargeBinaryArray, RecordBatch};
    use arrow_schema::{Field, Schema};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use futures::{StreamExt, TryStreamExt};

    fn stream_of(batch: RecordBatch) -> SendableRecordBatchStream {
        Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::iter(vec![Ok(batch)]),
        ))
    }

    async fn collect_column(
        stream: SendableRecordBatchStream,
        col: &str,
    ) -> (Vec<i32>, Arc<Schema>) {
        let schema = stream.schema();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let idx = schema.index_of(col).unwrap();
        let mut out = Vec::new();
        for b in &batches {
            let a = b.column(idx).as_any().downcast_ref::<Int32Array>().unwrap();
            out.extend((0..a.len()).map(|i| a.value(i)));
        }
        (out, schema)
    }

    fn assert_projection_inputs_fit(
        batches: &[RecordBatch],
        max_bytes: usize,
        added_bytes_per_row: usize,
    ) {
        assert!(!batches.is_empty());
        for batch in batches {
            let estimated_bytes = estimated_projection_bytes(batch, added_bytes_per_row);
            assert!(
                estimated_bytes <= max_bytes,
                "projection input estimate {estimated_bytes} exceeds cap {max_bytes}"
            );
        }
    }

    #[tokio::test]
    async fn sort_orders_by_single_key_and_preserves_schema() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("k", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
                Arc::new(Int32Array::from(vec![30, 10, 40, 20])),
            ],
        )
        .unwrap();

        let spec =
            ClusteringSpec::with_bits(vec!["k".into()], ClusteringCurve::ZOrder, 1, 32).unwrap();
        let sorted = cluster_sort_stream(stream_of(batch), &spec).await.unwrap();

        // Output schema is unchanged (transient ordering column dropped).
        let (ks, out_schema) = collect_column(sorted, "k").await;
        assert_eq!(out_schema.fields().len(), 2);
        assert_eq!(out_schema.field(0).name(), "id");
        assert_eq!(out_schema.field(1).name(), "k");
        // A single-column clustering key sorts by the raw value.
        assert_eq!(ks, vec![10, 20, 30, 40]);
    }

    #[tokio::test]
    async fn sort_uses_task_wide_ranks_for_small_int64_values() {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(vec![5_i64, 1, 3]))],
            )
            .unwrap(),
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(vec![4_i64, 0, 2]))],
            )
            .unwrap(),
        ];
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        ));
        let spec = ClusteringSpec::new(vec!["k".into()], ClusteringCurve::Hilbert).unwrap();

        let sorted = cluster_sort_stream(stream, &spec).await.unwrap();
        let batches = sorted.try_collect::<Vec<_>>().await.unwrap();
        let mut values = Vec::new();
        for batch in batches {
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            values.extend((0..array.len()).map(|row| array.value(row)));
        }
        assert_eq!(values, vec![0, 1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn sort_two_keys_groups_neighbours() {
        // Interleaving two keys must keep the input rows intact (a permutation),
        // and co-locate the two rows that are close in both dimensions.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2])),
                Arc::new(Int32Array::from(vec![0, 1000, 1])),
                Arc::new(Int32Array::from(vec![0, 1000, 1])),
            ],
        )
        .unwrap();

        let spec = ClusteringSpec::with_bits(
            vec!["x".into(), "y".into()],
            ClusteringCurve::Hilbert,
            1,
            32,
        )
        .unwrap();
        let sorted = cluster_sort_stream(stream_of(batch), &spec).await.unwrap();
        let (ids, _) = collect_column(sorted, "id").await;

        // All three input rows are still present.
        let mut seen = ids.clone();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
        // Rows 0 (0,0) and 2 (1,1) are neighbours; row 1 (1000,1000) is far, so
        // it must not sit between them.
        let p0 = ids.iter().position(|&v| v == 0).unwrap();
        let p2 = ids.iter().position(|&v| v == 2).unwrap();
        assert_eq!(p0.abs_diff(p2), 1, "near points should stay adjacent");
    }

    #[tokio::test]
    async fn sort_rejects_missing_column() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![0, 1]))])
                .unwrap();
        let spec = ClusteringSpec::new(vec!["missing".into()], ClusteringCurve::Hilbert).unwrap();
        assert!(cluster_sort_stream(stream_of(batch), &spec).await.is_err());
    }

    #[tokio::test]
    async fn projection_cap_rejects_single_row_that_cannot_fit() {
        const MAX_BYTES: usize = 1024;

        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("payload", DataType::LargeBinary, false),
        ]));
        let payload = vec![0_u8; MAX_BYTES];
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(LargeBinaryArray::from(vec![payload.as_slice()])),
            ],
        )
        .unwrap();

        let added_bytes_per_row = projection_bytes_per_row(1, 4).unwrap();
        let capped = cap_projection_input(stream_of(batch), MAX_BYTES, added_bytes_per_row);
        let error = capped.try_collect::<Vec<RecordBatch>>().await.unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("a single row is estimated to require"),
            "{message}"
        );
        assert!(
            message.contains("during clustering projection"),
            "{message}"
        );
        assert!(
            message.contains("exceeds the maximum allowed batch size"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn projection_cap_accounts_for_wide_ordering_keys() {
        const MAX_BYTES: usize = 4096;
        const NUM_ROWS: usize = 4096;

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Boolean, false),
            Field::new("y", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(BooleanArray::from(
                    (0..NUM_ROWS)
                        .map(|value| value % 2 == 0)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(BooleanArray::from(
                    (0..NUM_ROWS).map(|value| value % 4 < 2).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();

        let spec =
            ClusteringSpec::with_bits(vec!["x".into(), "y".into()], ClusteringCurve::ZOrder, 1, 64)
                .unwrap();
        let order_width = SpaceFillingEncoder::from_spec(&spec)
            .unwrap()
            .output_width(spec.columns.len())
            .unwrap();
        assert_eq!(order_width, 16);
        assert!(batch.get_array_memory_size() < MAX_BYTES);
        assert!(estimated_projection_bytes(&batch, order_width) > MAX_BYTES);

        let added_bytes_per_row =
            projection_bytes_per_row(spec.columns.len(), order_width).unwrap();
        let capped = cap_projection_input(stream_of(batch), MAX_BYTES, added_bytes_per_row);
        let batches: Vec<RecordBatch> = capped.try_collect().await.unwrap();
        assert_eq!(
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            NUM_ROWS
        );
        assert_projection_inputs_fit(&batches, MAX_BYTES, added_bytes_per_row);
    }

    #[tokio::test]
    async fn projection_cap_materializes_one_chunk_per_poll() {
        const MAX_BYTES: usize = 4096;
        const NUM_ROWS: usize = 4096;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "key",
            DataType::Boolean,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(BooleanArray::from(vec![true; NUM_ROWS]))],
        )
        .unwrap();
        let added_bytes_per_row = projection_bytes_per_row(1, 16).unwrap();
        assert!(estimated_projection_bytes(&batch, added_bytes_per_row) > MAX_BYTES);

        let materialized = Arc::new(AtomicUsize::new(0));
        let materialized_for_callback = materialized.clone();
        let mut capped = cap_projection_input_with_materialization_callback(
            stream_of(batch),
            MAX_BYTES,
            added_bytes_per_row,
            move || {
                materialized_for_callback.fetch_add(1, Ordering::SeqCst);
            },
        );

        assert_eq!(materialized.load(Ordering::SeqCst), 0);
        let first = capped.next().await.unwrap().unwrap();
        assert!(first.num_rows() < NUM_ROWS);
        assert_eq!(materialized.load(Ordering::SeqCst), 1);
        assert!(estimated_projection_bytes(&first, added_bytes_per_row) <= MAX_BYTES);

        let second = capped.next().await.unwrap().unwrap();
        assert!(second.num_rows() < NUM_ROWS);
        assert_eq!(materialized.load(Ordering::SeqCst), 2);
        assert!(estimated_projection_bytes(&second, added_bytes_per_row) <= MAX_BYTES);
    }

    #[tokio::test]
    async fn sort_rechunks_before_projection_for_constrained_pool() {
        const MEMORY_POOL_BYTES: u64 = 512 * 1024;
        const NUM_ROWS: usize = 10_000;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("x", DataType::Boolean, false),
            Field::new("y", DataType::Boolean, false),
        ]));
        let ids: Vec<i32> = (0..NUM_ROWS as i32).rev().collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids.clone())),
                Arc::new(BooleanArray::from(
                    ids.iter().map(|value| value % 2 == 0).collect::<Vec<_>>(),
                )),
                Arc::new(BooleanArray::from(
                    ids.iter().map(|value| value % 4 < 2).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let max_batch_bytes = sort_batch_byte_limit(MEMORY_POOL_BYTES);
        assert!(batch.get_array_memory_size() < max_batch_bytes);

        let spec =
            ClusteringSpec::with_bits(vec!["x".into(), "y".into()], ClusteringCurve::ZOrder, 1, 64)
                .unwrap();
        let order_width = SpaceFillingEncoder::from_spec(&spec)
            .unwrap()
            .output_width(spec.columns.len())
            .unwrap();
        let added_bytes_per_row =
            projection_bytes_per_row(spec.columns.len(), order_width).unwrap();
        assert!(estimated_projection_bytes(&batch, added_bytes_per_row) > max_batch_bytes);

        let sorted = cluster_sort_stream_with_options(
            stream_of(batch),
            &spec,
            LanceExecutionOptions {
                use_spilling: true,
                mem_pool_size: Some(MEMORY_POOL_BYTES),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let batches: Vec<RecordBatch> = sorted.try_collect().await.unwrap();
        assert_eq!(
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            NUM_ROWS
        );
        assert!(
            batches
                .iter()
                .all(|batch| batch.schema().fields().len() == 3)
        );
    }
}
