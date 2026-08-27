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

use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field as ArrowField, SortOptions};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::logical_expr::{ColumnarValue, ScalarUDF, Signature, Volatility};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result as DataFusionResult};
use datafusion_expr::{ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_physical_expr::expressions::Column as DFColumn;
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use lance_core::{Error, Result};
use lance_datafusion::exec::{LanceExecutionOptions, OneShotExec, execute_plan};

use super::{ClusteringSpec, SpaceFillingEncoder};

/// Name of the transient column holding the space-filling-curve ordering value.
const CLUSTERING_ORDER_FIELD: &str = "__lance_clustering_order";

/// Sort `data` so rows are ordered by the clustering-key space-filling curve.
///
/// All `spec.columns` must be present in the stream schema. The returned stream
/// yields the same schema as the input (the transient ordering column is
/// projected away). Sorting uses DataFusion's `SortExec`, which spills to disk
/// for inputs larger than memory.
pub async fn cluster_sort_stream(
    data: SendableRecordBatchStream,
    spec: &ClusteringSpec,
) -> Result<SendableRecordBatchStream> {
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

    let source = Arc::new(OneShotExec::new(data));

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
    let udf = ClusteringOrderUdf::new(SpaceFillingEncoder::from_spec(spec)?, spec.columns.len())?;
    let order_width = udf.output_width;
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
    let sorted = Arc::new(SortExec::new(
        [sort_expr].into(),
        with_order as Arc<dyn ExecutionPlan>,
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

    let stream = execute_plan(
        restored,
        LanceExecutionOptions {
            use_spilling: true,
            ..Default::default()
        },
    )?;
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
            && self.num_columns == other.num_columns
            && self.output_width == other.output_width
    }
}

impl Eq for ClusteringOrderUdf {}

impl std::hash::Hash for ClusteringOrderUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.signature.hash(state);
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
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{Field, Schema};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::TryStreamExt;
    use futures::stream;

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
}
