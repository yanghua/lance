// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{ArrayRef, UInt64Array};
use arrow_schema::{DataType, Field, SortOptions};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, Signature, Volatility,
};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result as DataFusionResult};
use datafusion_expr::ScalarUDFImpl;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use futures::TryStreamExt;
use lance_core::clustering::ClusteringSpec;
use lance_core::{Error, Result};
use lance_datafusion::exec::{
    LanceExecutionOptions, OneShotExec, execute_plan, provider_to_stream,
};
use lance_datafusion::spill::spilling_table_provider;

use super::{ClusteringModel, PartialClusteringModel};

const ORDER_COLUMN: &str = "__lance_clustering_order";
const REPLAY_MEMORY_LIMIT: usize = 64 * 1024 * 1024;

/// Sort a stream by the dataset's multi-column Hilbert key.
pub async fn cluster_sort_stream(
    data: datafusion::execution::SendableRecordBatchStream,
    spec: &ClusteringSpec,
) -> Result<datafusion::execution::SendableRecordBatchStream> {
    spec.validate()?;
    let input_schema = data.schema();
    let mut order_column = ORDER_COLUMN.to_string();
    while input_schema.index_of(&order_column).is_ok() {
        order_column.push('_');
    }
    let key_indices = spec
        .columns
        .iter()
        .map(|name| {
            input_schema.index_of(name).map_err(|_| {
                Error::invalid_input(format!(
                    "clustering column {name:?} is not present in the input"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let key_types = key_indices
        .iter()
        .map(|index| input_schema.field(*index).data_type().clone())
        .collect::<Vec<_>>();

    let replay = spilling_table_provider(data, REPLAY_MEMORY_LIMIT).await?;
    let mut sample_stream = provider_to_stream(replay.clone()).await?;
    let mut model = PartialClusteringModel::try_new(spec, &key_types)?;
    let mut next_row_address = 0_u64;
    while let Some(batch) = sample_stream.try_next().await? {
        let columns = key_indices
            .iter()
            .map(|index| batch.column(*index).clone())
            .collect::<Vec<_>>();
        let end_row_address = next_row_address
            .checked_add(batch.num_rows() as u64)
            .ok_or_else(|| Error::invalid_input("clustering row count overflowed u64"))?;
        let row_addresses = UInt64Array::from_iter_values(next_row_address..end_row_address);
        model.add(&columns, &row_addresses)?;
        next_row_address = end_row_address;
    }

    let model = model.finish();
    cluster_sort_stream_with_model(provider_to_stream(replay).await?, &model).await
}

/// Sort a stream with a precomputed model shared by all distributed workers.
#[doc(hidden)]
pub async fn cluster_sort_stream_with_model(
    data: datafusion::execution::SendableRecordBatchStream,
    model: &ClusteringModel,
) -> Result<datafusion::execution::SendableRecordBatchStream> {
    let input_schema = data.schema();
    let mut order_column = ORDER_COLUMN.to_string();
    while input_schema.index_of(&order_column).is_ok() {
        order_column.push('_');
    }
    let key_indices = model
        .columns()
        .iter()
        .map(|name| {
            input_schema.index_of(name).map_err(|_| {
                Error::invalid_input(format!(
                    "clustering column {name:?} is not present in the input"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let width = model.output_width()?;
    let source = Arc::new(OneShotExec::new(data));
    let mut projection = input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            (
                Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            )
        })
        .collect::<Vec<_>>();
    let key_args = key_indices
        .iter()
        .map(|index| {
            Arc::new(Column::new(input_schema.field(*index).name(), *index))
                as Arc<dyn PhysicalExpr>
        })
        .collect();
    let order_index = projection.len();
    projection.push((
        Arc::new(ScalarFunctionExpr::new(
            &order_column,
            Arc::new(ScalarUDF::new_from_impl(ClusteringOrderUdf::new(
                model.clone(),
                key_indices.len(),
                width,
            ))),
            key_args,
            Arc::new(Field::new(
                ORDER_COLUMN,
                DataType::FixedSizeBinary(width as i32),
                false,
            )),
            Arc::new(ConfigOptions::default()),
        )) as Arc<dyn PhysicalExpr>,
        order_column.clone(),
    ));

    let with_order = Arc::new(ProjectionExec::try_new(projection, source)?);
    let sorted = Arc::new(SortExec::new(
        [PhysicalSortExpr {
            expr: Arc::new(Column::new(&order_column, order_index)),
            options: SortOptions::default(),
        }]
        .into(),
        with_order,
    ));
    let restored = Arc::new(ProjectionExec::try_new(
        input_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| {
                (
                    Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>,
                    field.name().clone(),
                )
            })
            .collect::<Vec<_>>(),
        sorted,
    )?);
    execute_plan(
        restored,
        LanceExecutionOptions {
            use_spilling: true,
            ..Default::default()
        },
    )
}

#[derive(Debug, Clone)]
struct ClusteringOrderUdf {
    signature: Signature,
    encoder: ClusteringModel,
    width: usize,
}

impl ClusteringOrderUdf {
    fn new(encoder: ClusteringModel, columns: usize, width: usize) -> Self {
        Self {
            signature: Signature::any(columns, Volatility::Immutable),
            encoder,
            width,
        }
    }
}

impl PartialEq for ClusteringOrderUdf {
    fn eq(&self, other: &Self) -> bool {
        self.signature == other.signature
            && self.encoder == other.encoder
            && self.width == other.width
    }
}

impl Eq for ClusteringOrderUdf {}

impl std::hash::Hash for ClusteringOrderUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.signature.hash(state);
        self.encoder.hash(state);
        self.width.hash(state);
    }
}

impl ScalarUDFImpl for ClusteringOrderUdf {
    fn name(&self) -> &str {
        ORDER_COLUMN
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::FixedSizeBinary(self.width as i32))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        let columns = args
            .args
            .into_iter()
            .map(|value| value.into_array(args.number_rows))
            .collect::<DataFusionResult<Vec<ArrayRef>>>()?;
        self.encoder
            .encode(&columns)
            .map(ColumnarValue::Array)
            .map_err(|error| DataFusionError::External(Box::new(error)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::Schema;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::{TryStreamExt, stream};

    #[tokio::test]
    async fn sorts_single_key_and_preserves_schema() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(ORDER_COLUMN, DataType::Int32, false),
            Field::new("key", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2])),
                Arc::new(Int32Array::from(vec![30, 10, 20])),
            ],
        )
        .unwrap();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter([Ok(batch)]),
        ));
        let spec = ClusteringSpec::new(vec!["key".into()], 1).unwrap();
        let output = cluster_sort_stream(stream, &spec)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(output[0].schema(), schema);
        assert_eq!(
            output[0]["key"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[10, 20, 30]
        );
    }
}
