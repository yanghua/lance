// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Multi-column Hilbert clustering primitives.

use std::collections::BTreeMap;
use std::str::FromStr;

use arrow_array::builder::FixedSizeBinaryBuilder;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, SortOptions};
pub use lance_core::clustering::CLUSTERING_ALGORITHM_REVISION;
use lance_core::clustering::{ClusteringSpec, validate_data_type};
use lance_core::{Error, Result};
use prost::Message;

mod sort;
pub use sort::{cluster_sort_stream, cluster_sort_stream_with_model};

const BITS_PER_DIM: u32 = 16;
const MAX_SAMPLES_PER_COLUMN: usize = 65_536;
const MAX_SAMPLE_BYTES_PER_COLUMN: usize = 8 * 1024 * 1024;
// Stable across architectures so merged samples do not depend on worker pointer width.
const SAMPLE_ENTRY_OVERHEAD_BYTES: usize = 40;
const MODEL_FORMAT_VERSION: u32 = 1;
const ROW_DIGEST_DOMAIN: u64 = u64::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
struct SampleColumn {
    data_type: DataType,
    samples: BTreeMap<(u64, u64), Vec<u8>>,
    max_encoded_value_bytes: usize,
}

/// Mergeable deterministic sample state for distributed clustering.
///
/// Workers must identify rows with stable, unique identities such as physical row
/// addresses from the planned snapshot. Merging is independent of task arrival
/// order, retry order, and input partitioning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialClusteringModel {
    columns: Vec<String>,
    sample_columns: Vec<SampleColumn>,
    context: Vec<u8>,
    input_rows: u64,
    row_digest: RowDigest,
}

/// Order-independent row-identity fingerprint carried through distributed shuffle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct RowDigest {
    count: u64,
    xor: u64,
    sum: u64,
    sum_squares: u64,
}

impl RowDigest {
    #[doc(hidden)]
    pub fn from_row_addresses(row_addresses: &UInt64Array) -> Result<Self> {
        if row_addresses.null_count() != 0 {
            return Err(Error::invalid_input(
                "clustering row addresses must not contain nulls",
            ));
        }
        let mut digest = Self::default();
        for row_address in row_addresses.values() {
            digest.add_row_address(*row_address)?;
        }
        Ok(digest)
    }

    #[doc(hidden)]
    pub fn add_row_address(&mut self, row_address: u64) -> Result<()> {
        let value = sample_priority(row_address, ROW_DIGEST_DOMAIN);
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| Error::invalid_input("clustering row digest count overflowed u64"))?;
        self.xor ^= value;
        self.sum = self.sum.wrapping_add(value);
        self.sum_squares = self.sum_squares.wrapping_add(value.wrapping_mul(value));
        Ok(())
    }

    #[doc(hidden)]
    pub fn merge(&mut self, other: Self) -> Result<()> {
        self.count = self
            .count
            .checked_add(other.count)
            .ok_or_else(|| Error::invalid_input("clustering row digest count overflowed u64"))?;
        self.xor ^= other.xor;
        self.sum = self.sum.wrapping_add(other.sum);
        self.sum_squares = self.sum_squares.wrapping_add(other.sum_squares);
        Ok(())
    }

    #[doc(hidden)]
    pub fn to_bytes(self) -> [u8; 32] {
        let mut bytes = [0; 32];
        bytes[..8].copy_from_slice(&self.count.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.xor.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.sum.to_le_bytes());
        bytes[24..].copy_from_slice(&self.sum_squares.to_le_bytes());
        bytes
    }

    #[doc(hidden)]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let word = |offset: usize| {
            let mut value = [0_u8; 8];
            value.copy_from_slice(&bytes[offset..offset + 8]);
            u64::from_le_bytes(value)
        };
        Self {
            count: word(0),
            xor: word(8),
            sum: word(16),
            sum_squares: word(24),
        }
    }

    #[doc(hidden)]
    pub fn num_rows(self) -> u64 {
        self.count
    }

    fn from_words(words: &[u64]) -> Result<Self> {
        if words.len() != 4 {
            return Err(Error::invalid_input(format!(
                "clustering row digest must contain 4 words, got {}",
                words.len()
            )));
        }
        let digest = Self {
            count: words[0],
            xor: words[1],
            sum: words[2],
            sum_squares: words[3],
        };
        if digest.count == 0 && (digest.xor != 0 || digest.sum != 0 || digest.sum_squares != 0) {
            return Err(Error::invalid_input(
                "empty clustering row digest has non-zero accumulators",
            ));
        }
        Ok(digest)
    }

    fn words(self) -> [u64; 4] {
        [self.count, self.xor, self.sum, self.sum_squares]
    }
}

impl PartialClusteringModel {
    /// Build a deterministic partial model from an Arrow batch and physical row addresses.
    #[doc(hidden)]
    pub fn from_batch_bytes(
        context: Vec<u8>,
        columns: Vec<String>,
        batch: &RecordBatch,
        row_addresses: &UInt64Array,
    ) -> Result<Vec<u8>> {
        let spec = ClusteringSpec::new(columns, 1)?;
        let arrays = spec
            .columns
            .iter()
            .map(|column| {
                batch.column_by_name(column).cloned().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "clustering column {column:?} is not present in the input"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let data_types = arrays
            .iter()
            .map(|column| column.data_type().clone())
            .collect::<Vec<_>>();
        let mut model = Self::try_new_with_context(&spec, &data_types, context)?;
        model.add(&arrays, row_addresses)?;
        Ok(model.to_bytes())
    }

    pub(crate) fn try_new(spec: &ClusteringSpec, data_types: &[DataType]) -> Result<Self> {
        Self::try_new_with_context(spec, data_types, Vec::new())
    }

    /// Create an empty partial model bound to an opaque coordinator context.
    #[doc(hidden)]
    pub fn try_new_with_context(
        spec: &ClusteringSpec,
        data_types: &[DataType],
        context: Vec<u8>,
    ) -> Result<Self> {
        spec.validate()?;
        spec.validate_current_algorithm()?;
        if data_types.len() != spec.columns.len() {
            return Err(Error::invalid_input(format!(
                "clustering spec has {} columns but {} data types were supplied",
                spec.columns.len(),
                data_types.len()
            )));
        }
        let sample_columns = data_types
            .iter()
            .map(|data_type| {
                validate_data_type(data_type)?;
                Ok(SampleColumn {
                    data_type: data_type.clone(),
                    samples: BTreeMap::new(),
                    max_encoded_value_bytes: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            columns: spec.columns.clone(),
            sample_columns,
            context,
            input_rows: 0,
            row_digest: RowDigest::default(),
        })
    }

    /// Add clustering values identified by physical row addresses.
    #[doc(hidden)]
    pub fn add(&mut self, columns: &[ArrayRef], row_addresses: &UInt64Array) -> Result<()> {
        if columns.len() != self.sample_columns.len() {
            return Err(Error::invalid_input(format!(
                "clustering sample expected {} columns, got {}",
                self.sample_columns.len(),
                columns.len()
            )));
        }
        if columns
            .iter()
            .any(|column| column.len() != row_addresses.len())
        {
            return Err(Error::invalid_input(
                "clustering columns and row addresses must have equal lengths",
            ));
        }
        if row_addresses.null_count() != 0 {
            return Err(Error::invalid_input(
                "clustering sample row addresses must not contain nulls",
            ));
        }
        self.input_rows = self
            .input_rows
            .checked_add(row_addresses.len() as u64)
            .ok_or_else(|| Error::invalid_input("clustering input row count overflowed u64"))?;
        self.row_digest
            .merge(RowDigest::from_row_addresses(row_addresses)?)?;
        for (column_index, array) in columns.iter().enumerate() {
            let sample_column = &mut self.sample_columns[column_index];
            if array.data_type() != &sample_column.data_type {
                return Err(Error::invalid_input(format!(
                    "clustering column {column_index} changed type from {:?} to {:?}",
                    sample_column.data_type,
                    array.data_type()
                )));
            }
            let converter = row_converter(array.data_type().clone())?;
            let rows = converter
                .convert_columns(std::slice::from_ref(array))
                .map_err(|error| {
                    Error::invalid_input(format!(
                        "failed to encode clustering sample column {column_index}: {error}"
                    ))
                })?;
            for row_index in 0..array.len() {
                if !array.is_null(row_index) {
                    let row_address = row_addresses.value(row_index);
                    insert_sample(
                        sample_column,
                        column_index,
                        row_address,
                        rows.row(row_index).as_ref().to_vec(),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Merge partial states produced by independent workers.
    pub(crate) fn merge(partials: impl IntoIterator<Item = Self>) -> Result<Self> {
        let mut partials = partials.into_iter();
        let mut merged = partials.next().ok_or_else(|| {
            Error::invalid_input("at least one partial clustering model is required")
        })?;
        for partial in partials {
            if partial.columns != merged.columns {
                return Err(Error::invalid_input(
                    "partial clustering models have different columns",
                ));
            }
            if partial.context != merged.context {
                return Err(Error::invalid_input(
                    "partial clustering models belong to different plans",
                ));
            }
            merged.input_rows = merged
                .input_rows
                .checked_add(partial.input_rows)
                .ok_or_else(|| Error::invalid_input("clustering input row count overflowed u64"))?;
            merged.row_digest.merge(partial.row_digest)?;
            if partial.sample_columns.len() != merged.sample_columns.len() {
                return Err(Error::invalid_input(
                    "partial clustering models have different column counts",
                ));
            }
            for (column_index, (target, source)) in merged
                .sample_columns
                .iter_mut()
                .zip(partial.sample_columns)
                .enumerate()
            {
                if target.data_type != source.data_type {
                    return Err(Error::invalid_input(
                        "partial clustering models have different column types",
                    ));
                }
                target.max_encoded_value_bytes = target
                    .max_encoded_value_bytes
                    .max(source.max_encoded_value_bytes);
                for ((_, row_address), encoded_value) in source.samples {
                    insert_sample(target, column_index, row_address, encoded_value)?;
                }
                truncate_samples(target, column_index)?;
            }
        }
        Ok(merged)
    }

    #[doc(hidden)]
    pub fn finish(self) -> ClusteringModel {
        ClusteringModel {
            columns: self.columns,
            context: self.context,
            input_rows: self.input_rows,
            row_digest: self.row_digest,
            columns_data: self
                .sample_columns
                .into_iter()
                .map(|column| {
                    let mut samples = column.samples.into_values().collect::<Vec<_>>();
                    samples.sort_unstable();
                    ModelColumn {
                        data_type: column.data_type,
                        samples,
                    }
                })
                .collect(),
        }
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        crate::clustering_pb::PartialModel::from(self).encode_to_vec()
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::try_from(crate::clustering_pb::PartialModel::decode(bytes)?)
    }

    #[doc(hidden)]
    pub fn merge_bytes(partials: impl IntoIterator<Item = Vec<u8>>) -> Result<Vec<u8>> {
        let partials = partials
            .into_iter()
            .map(|bytes| Self::from_bytes(&bytes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::merge(partials)?.to_bytes())
    }

    /// Decode partial payloads, merge them, and return a final model.
    #[doc(hidden)]
    pub fn merge_bytes_to_model(
        partials: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<ClusteringModel> {
        let partials = partials
            .into_iter()
            .map(|bytes| Self::from_bytes(&bytes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::merge(partials)?.finish())
    }
}

fn insert_sample(
    column: &mut SampleColumn,
    column_index: usize,
    row_address: u64,
    encoded_value: Vec<u8>,
) -> Result<()> {
    column.max_encoded_value_bytes = column.max_encoded_value_bytes.max(encoded_value.len());
    let key = (
        sample_priority(row_address, column_index as u64),
        row_address,
    );
    if let Some(previous) = column.samples.get(&key)
        && previous != &encoded_value
    {
        return Err(Error::invalid_input(format!(
            "clustering sample row address {row_address} has conflicting values"
        )));
    }
    column.samples.insert(key, encoded_value);
    truncate_samples_to_capacity(column)
}

fn truncate_samples(column: &mut SampleColumn, column_index: usize) -> Result<()> {
    for ((priority, row_address), encoded_value) in &column.samples {
        if *priority != sample_priority(*row_address, column_index as u64) {
            return Err(Error::invalid_input(format!(
                "clustering sample for row address {} has an invalid priority",
                row_address
            )));
        }
        if encoded_value.len() > column.max_encoded_value_bytes {
            return Err(Error::invalid_input(format!(
                "clustering sample row address {row_address} exceeds the declared maximum encoded value size"
            )));
        }
    }
    truncate_samples_to_capacity(column)
}

fn truncate_samples_to_capacity(column: &mut SampleColumn) -> Result<()> {
    if column.samples.is_empty() {
        return Ok(());
    }
    let entry_bytes = column
        .max_encoded_value_bytes
        .checked_add(SAMPLE_ENTRY_OVERHEAD_BYTES)
        .ok_or_else(|| Error::invalid_input("clustering sample size overflowed usize"))?;
    let capacity = MAX_SAMPLES_PER_COLUMN.min(MAX_SAMPLE_BYTES_PER_COLUMN / entry_bytes);
    if capacity == 0 {
        return Err(Error::invalid_input(format!(
            "one clustering value requires more than the {MAX_SAMPLE_BYTES_PER_COLUMN}-byte sample limit"
        )));
    }
    while column.samples.len() > capacity {
        column.samples.pop_last();
    }
    Ok(())
}

fn sample_priority(row_address: u64, domain: u64) -> u64 {
    // SplitMix64 gives a stable, well-distributed priority without putting a
    // cryptographic hash in the per-row sampling hot path.
    let mut value =
        row_address ^ domain.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x4c41_4e43_455f_514e;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ModelColumn {
    data_type: DataType,
    samples: Vec<Vec<u8>>,
}

/// Final empirical-rank model shared by every worker in one clustering group.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClusteringModel {
    columns: Vec<String>,
    columns_data: Vec<ModelColumn>,
    context: Vec<u8>,
    input_rows: u64,
    row_digest: RowDigest,
}

impl ClusteringModel {
    #[doc(hidden)]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    #[doc(hidden)]
    pub fn matches_context(&self, context: &[u8]) -> bool {
        self.context == context
    }

    #[doc(hidden)]
    pub fn input_rows(&self) -> u64 {
        self.input_rows
    }

    #[doc(hidden)]
    pub fn row_digest(&self) -> RowDigest {
        self.row_digest
    }

    #[doc(hidden)]
    pub fn encode_batch(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let arrays = self
            .columns
            .iter()
            .map(|column| {
                batch.column_by_name(column).cloned().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "clustering column {column:?} is not present in the input"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.encode(&arrays)
    }

    pub(crate) fn encode(&self, columns: &[ArrayRef]) -> Result<ArrayRef> {
        let num_rows = columns
            .first()
            .ok_or_else(|| Error::invalid_input("clustering requires at least one column"))?
            .len();
        if columns.len() != self.columns_data.len() {
            return Err(Error::invalid_input(format!(
                "clustering model expected {} columns, got {}",
                self.columns_data.len(),
                columns.len()
            )));
        }
        if columns.iter().any(|column| column.len() != num_rows) {
            return Err(Error::invalid_input(
                "clustering columns must have equal lengths",
            ));
        }
        let coordinates = columns
            .iter()
            .enumerate()
            .map(|(column, array)| self.coordinates(column, array))
            .collect::<Result<Vec<_>>>()?;
        let width = output_width(columns.len())?;
        let mut builder = FixedSizeBinaryBuilder::with_capacity(num_rows, width as i32);
        let mut point = vec![0; columns.len()];
        for row in 0..num_rows {
            for (column, values) in coordinates.iter().enumerate() {
                point[column] = values[row];
            }
            axes_to_transpose(&mut point);
            let encoded = interleave(&point).to_be_bytes();
            builder
                .append_value(&encoded[encoded.len() - width..])
                .map_err(|error| {
                    Error::invalid_input(format!("failed to build clustering key: {error}"))
                })?;
        }
        Ok(std::sync::Arc::new(builder.finish()))
    }

    fn coordinates(&self, column: usize, array: &ArrayRef) -> Result<Vec<u64>> {
        let model = self.columns_data.get(column).ok_or_else(|| {
            Error::invalid_input(format!(
                "clustering model has {} columns but column {column} was requested",
                self.columns_data.len()
            ))
        })?;
        if &model.data_type != array.data_type() {
            return Err(Error::invalid_input(format!(
                "clustering model column {column} has type {:?}, got {:?}",
                model.data_type,
                array.data_type()
            )));
        }
        let converter = row_converter(array.data_type().clone())?;
        let rows = converter
            .convert_columns(std::slice::from_ref(array))
            .map_err(|error| {
                Error::invalid_input(format!("failed to encode clustering column: {error}"))
            })?;
        let null_coordinate = max_coordinate();
        let max_non_null_coordinate = null_coordinate - 1;
        Ok((0..array.len())
            .map(|index| {
                if array.is_null(index) {
                    return null_coordinate;
                }
                let encoded_row = rows.row(index);
                let row = encoded_row.as_ref();
                let upper = model
                    .samples
                    .partition_point(|sample| sample.as_slice() <= row);
                let rank = upper.saturating_sub(1);
                if model.samples.len() <= 1 {
                    0
                } else {
                    (rank as u128 * max_non_null_coordinate as u128
                        / (model.samples.len() - 1) as u128) as u64
                }
            })
            .collect())
    }

    pub(crate) fn output_width(&self) -> Result<usize> {
        output_width(self.columns_data.len())
    }

    #[doc(hidden)]
    pub fn to_bytes(&self) -> Vec<u8> {
        crate::clustering_pb::Model::from(self).encode_to_vec()
    }

    #[doc(hidden)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::try_from(crate::clustering_pb::Model::decode(bytes)?)
    }

    #[doc(hidden)]
    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&self.to_bytes()).as_bytes()
    }
}

fn output_width(num_columns: usize) -> Result<usize> {
    if !(1..=lance_core::clustering::MAX_CLUSTERING_COLUMNS).contains(&num_columns) {
        return Err(Error::invalid_input(format!(
            "clustering supports one to {} columns, got {num_columns}",
            lance_core::clustering::MAX_CLUSTERING_COLUMNS
        )));
    }
    Ok((num_columns * BITS_PER_DIM as usize).div_ceil(8))
}

fn row_converter(data_type: DataType) -> Result<RowConverter> {
    RowConverter::new(vec![SortField::new_with_options(
        data_type,
        SortOptions {
            descending: false,
            nulls_first: false,
        },
    )])
    .map_err(|error| Error::invalid_input(format!("unsupported clustering type: {error}")))
}

fn validate_wire_header(format_version: u32, algorithm_revision: &str) -> Result<()> {
    if format_version != MODEL_FORMAT_VERSION {
        return Err(Error::invalid_input(format!(
            "unsupported clustering model format version {format_version}"
        )));
    }
    if algorithm_revision != CLUSTERING_ALGORITHM_REVISION {
        return Err(Error::invalid_input(format!(
            "unsupported clustering algorithm revision {algorithm_revision:?}"
        )));
    }
    Ok(())
}

impl From<&PartialClusteringModel> for crate::clustering_pb::PartialModel {
    fn from(model: &PartialClusteringModel) -> Self {
        Self {
            format_version: MODEL_FORMAT_VERSION,
            algorithm_revision: CLUSTERING_ALGORITHM_REVISION.to_string(),
            columns: model.columns.clone(),
            context: model.context.clone(),
            input_rows: model.input_rows,
            row_digest: model.row_digest.words().to_vec(),
            sample_columns: model
                .sample_columns
                .iter()
                .map(|column| crate::clustering_pb::SampleColumn {
                    data_type: column.data_type.to_string(),
                    max_encoded_value_bytes: column.max_encoded_value_bytes as u64,
                    samples: column
                        .samples
                        .iter()
                        .map(|((priority, row_address), encoded_value)| {
                            crate::clustering_pb::Sample {
                                priority: *priority,
                                row_address: *row_address,
                                encoded_value: encoded_value.clone(),
                            }
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl TryFrom<crate::clustering_pb::PartialModel> for PartialClusteringModel {
    type Error = Error;

    fn try_from(model: crate::clustering_pb::PartialModel) -> Result<Self> {
        validate_wire_header(model.format_version, &model.algorithm_revision)?;
        ClusteringSpec::new(model.columns.clone(), 1)?;
        if model.columns.len() != model.sample_columns.len() {
            return Err(Error::invalid_input(
                "partial clustering model column metadata is inconsistent",
            ));
        }
        let row_digest = RowDigest::from_words(&model.row_digest)?;
        if row_digest.count != model.input_rows {
            return Err(Error::invalid_input(format!(
                "partial clustering model input_rows is {}, but its row digest counts {} rows",
                model.input_rows, row_digest.count
            )));
        }
        let mut result = Self {
            columns: model.columns,
            context: model.context,
            input_rows: model.input_rows,
            row_digest,
            sample_columns: model
                .sample_columns
                .into_iter()
                .map(|column| {
                    let data_type = DataType::from_str(&column.data_type).map_err(|error| {
                        Error::invalid_input(format!(
                            "invalid clustering model data type {:?}: {error}",
                            column.data_type
                        ))
                    })?;
                    validate_data_type(&data_type)?;
                    Ok(SampleColumn {
                        data_type,
                        max_encoded_value_bytes: usize::try_from(column.max_encoded_value_bytes)
                            .map_err(|_| {
                                Error::invalid_input(
                                    "clustering maximum encoded value size exceeds usize",
                                )
                            })?,
                        samples: column
                            .samples
                            .into_iter()
                            .map(|sample| {
                                ((sample.priority, sample.row_address), sample.encoded_value)
                            })
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };
        for (column_index, column) in result.sample_columns.iter_mut().enumerate() {
            let minimum_max = column
                .samples
                .values()
                .map(Vec::len)
                .max()
                .unwrap_or_default();
            if column.max_encoded_value_bytes < minimum_max {
                return Err(Error::invalid_input(format!(
                    "clustering sample column {column_index} declares max encoded value size \
                     {}, but a retained sample requires {minimum_max}",
                    column.max_encoded_value_bytes
                )));
            }
            truncate_samples(column, column_index)?;
        }
        Ok(result)
    }
}

impl From<&ClusteringModel> for crate::clustering_pb::Model {
    fn from(model: &ClusteringModel) -> Self {
        Self {
            format_version: MODEL_FORMAT_VERSION,
            algorithm_revision: CLUSTERING_ALGORITHM_REVISION.to_string(),
            columns: model.columns.clone(),
            context: model.context.clone(),
            input_rows: model.input_rows,
            row_digest: model.row_digest.words().to_vec(),
            model_columns: model
                .columns_data
                .iter()
                .map(|column| crate::clustering_pb::ModelColumn {
                    data_type: column.data_type.to_string(),
                    encoded_values: column.samples.clone(),
                })
                .collect(),
        }
    }
}

impl TryFrom<crate::clustering_pb::Model> for ClusteringModel {
    type Error = Error;

    fn try_from(model: crate::clustering_pb::Model) -> Result<Self> {
        validate_wire_header(model.format_version, &model.algorithm_revision)?;
        ClusteringSpec::new(model.columns.clone(), 1)?;
        if model.columns.is_empty() || model.columns.len() != model.model_columns.len() {
            return Err(Error::invalid_input(
                "clustering model column metadata is inconsistent",
            ));
        }
        let row_digest = RowDigest::from_words(&model.row_digest)?;
        if row_digest.count != model.input_rows {
            return Err(Error::invalid_input(format!(
                "clustering model input_rows is {}, but its row digest counts {} rows",
                model.input_rows, row_digest.count
            )));
        }
        let columns_data = model
            .model_columns
            .into_iter()
            .map(|column| {
                let data_type = DataType::from_str(&column.data_type).map_err(|error| {
                    Error::invalid_input(format!(
                        "invalid clustering model data type {:?}: {error}",
                        column.data_type
                    ))
                })?;
                validate_data_type(&data_type)?;
                if !column
                    .encoded_values
                    .windows(2)
                    .all(|pair| pair[0] <= pair[1])
                {
                    return Err(Error::invalid_input(
                        "clustering model samples must be sorted",
                    ));
                }
                let total_bytes =
                    column
                        .encoded_values
                        .iter()
                        .try_fold(0_usize, |sum, value| {
                            sum.checked_add(value.len()).ok_or_else(|| {
                                Error::invalid_input(
                                    "clustering model sample bytes overflowed usize",
                                )
                            })
                        })?;
                if column.encoded_values.len() > MAX_SAMPLES_PER_COLUMN
                    || total_bytes > MAX_SAMPLE_BYTES_PER_COLUMN
                {
                    return Err(Error::invalid_input(format!(
                        "clustering model exceeds the per-column limits of \
                         {MAX_SAMPLES_PER_COLUMN} samples and \
                         {MAX_SAMPLE_BYTES_PER_COLUMN} bytes"
                    )));
                }
                Ok(ModelColumn {
                    data_type,
                    samples: column.encoded_values,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            columns: model.columns,
            columns_data,
            context: model.context,
            input_rows: model.input_rows,
            row_digest,
        })
    }
}

fn max_coordinate() -> u64 {
    (1_u64 << BITS_PER_DIM) - 1
}

fn interleave(point: &[u64]) -> u128 {
    let mut result = 0_u128;
    for bit in (0..BITS_PER_DIM).rev() {
        for coordinate in point {
            result = (result << 1) | ((coordinate >> bit) & 1) as u128;
        }
    }
    result
}

fn axes_to_transpose(point: &mut [u64]) {
    let last = point.len() - 1;
    let mut q = 1_u64 << (BITS_PER_DIM - 1);
    while q > 1 {
        let p = q - 1;
        for index in 0..point.len() {
            if point[index] & q != 0 {
                point[0] ^= p;
            } else {
                let swap = (point[0] ^ point[index]) & p;
                point[0] ^= swap;
                point[index] ^= swap;
            }
        }
        q >>= 1;
    }
    for index in 1..point.len() {
        point[index] ^= point[index - 1];
    }
    let mut prefix = 0;
    q = 1_u64 << (BITS_PER_DIM - 1);
    while q > 1 {
        if point[last] & q != 0 {
            prefix ^= q - 1;
        }
        q >>= 1;
    }
    for coordinate in point {
        *coordinate ^= prefix;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{FixedSizeBinaryArray, Int32Array, StringArray};

    use super::*;

    fn spec(columns: &[&str]) -> ClusteringSpec {
        ClusteringSpec::new(
            columns.iter().map(|column| (*column).to_string()).collect(),
            1,
        )
        .unwrap()
    }

    #[test]
    fn empirical_ranks_preserve_scalar_order() {
        let numeric: ArrayRef = Arc::new(Int32Array::from(vec![Some(3), None, Some(1), Some(2)]));
        let strings: ArrayRef = Arc::new(StringArray::from(vec!["z", "a", "m"]));
        for (array, order) in [(numeric, vec![2, 3, 0, 1]), (strings, vec![1, 2, 0])] {
            let mut partial =
                PartialClusteringModel::try_new(&spec(&["key"]), &[array.data_type().clone()])
                    .unwrap();
            let row_ids = UInt64Array::from_iter_values(0..array.len() as u64);
            partial.add(std::slice::from_ref(&array), &row_ids).unwrap();
            let encoded = partial.finish().encode(&[array]).unwrap();
            let encoded = encoded
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            assert!(
                order
                    .windows(2)
                    .all(|pair| encoded.value(pair[0]) < encoded.value(pair[1]))
            );
        }
    }

    #[test]
    fn partial_models_merge_independently_of_partition_and_arrival_order() {
        let values: ArrayRef = Arc::new(Int32Array::from_iter_values(0..100));
        let ids = UInt64Array::from_iter_values(10_000..10_100);
        let mut whole =
            PartialClusteringModel::try_new(&spec(&["key"]), &[DataType::Int32]).unwrap();
        whole.add(std::slice::from_ref(&values), &ids).unwrap();

        let mut left =
            PartialClusteringModel::try_new(&spec(&["key"]), &[DataType::Int32]).unwrap();
        let left_ids = UInt64Array::from_iter_values(10_000..10_040);
        left.add(&[values.slice(0, 40)], &left_ids).unwrap();
        let mut right =
            PartialClusteringModel::try_new(&spec(&["key"]), &[DataType::Int32]).unwrap();
        let right_ids = UInt64Array::from_iter_values(10_040..10_100);
        right.add(&[values.slice(40, 60)], &right_ids).unwrap();

        let merged = PartialClusteringModel::merge([right.clone(), left.clone()]).unwrap();
        let reversed = PartialClusteringModel::merge([left, right]).unwrap();
        assert_eq!(whole.finish(), merged.clone().finish());
        assert_eq!(merged.finish(), reversed.finish());
    }

    #[test]
    fn bounded_partial_merge_matches_whole_input() {
        let value = "x".repeat(2047);
        let values = (0..5_000)
            .map(|index| format!("{value}{index:06}"))
            .collect::<Vec<_>>();
        let values: ArrayRef = Arc::new(StringArray::from(values));
        let ids = UInt64Array::from_iter_values(0..5_000);
        let mut whole =
            PartialClusteringModel::try_new(&spec(&["key"]), &[DataType::Utf8]).unwrap();
        whole.add(std::slice::from_ref(&values), &ids).unwrap();

        let mut partials = Vec::new();
        for start in (0..5_000).step_by(500) {
            let mut partial =
                PartialClusteringModel::try_new(&spec(&["key"]), &[DataType::Utf8]).unwrap();
            let partial_ids = UInt64Array::from_iter_values(start as u64..start as u64 + 500);
            partial
                .add(&[values.slice(start, 500)], &partial_ids)
                .unwrap();
            partials.push(partial);
        }
        partials.reverse();
        let merged = PartialClusteringModel::merge(partials).unwrap();

        assert_eq!(whole.finish(), merged.finish());
    }

    #[test]
    fn partial_wire_round_trip_allows_truncated_maximum_value() {
        let short_values = (0..128)
            .map(|index| format!("v{index}"))
            .collect::<Vec<_>>();
        let short_values: ArrayRef = Arc::new(StringArray::from(short_values));
        let short_ids = UInt64Array::from_iter_values(0..128);
        let mut partial =
            PartialClusteringModel::try_new(&spec(&["key"]), &[DataType::Utf8]).unwrap();
        partial
            .add(std::slice::from_ref(&short_values), &short_ids)
            .unwrap();

        let long_value: ArrayRef = Arc::new(StringArray::from(vec!["x".repeat(1024 * 1024)]));
        let retained_priorities = &partial.sample_columns[0].samples;
        let long_row_id = (128_u64..)
            .find(|row_id| {
                sample_priority(*row_id, 0)
                    > retained_priorities
                        .last_key_value()
                        .map(|((priority, _), _)| *priority)
                        .unwrap()
            })
            .unwrap();
        partial
            .add(&[long_value], &UInt64Array::from(vec![long_row_id]))
            .unwrap();

        let decoded = PartialClusteringModel::from_bytes(&partial.to_bytes()).unwrap();
        assert_eq!(partial, decoded);
    }

    #[test]
    fn model_wire_round_trip_preserves_keys_and_digest() {
        let values: ArrayRef = Arc::new(StringArray::from(vec![Some("z"), None, Some("a")]));
        let ids = UInt64Array::from_iter_values(0..3);
        let mut partial =
            PartialClusteringModel::try_new(&spec(&["key"]), &[DataType::Utf8]).unwrap();
        partial.add(std::slice::from_ref(&values), &ids).unwrap();
        let partial = PartialClusteringModel::from_bytes(&partial.to_bytes()).unwrap();
        let model = partial.finish();
        let decoded = ClusteringModel::from_bytes(&model.to_bytes()).unwrap();
        assert_eq!(model, decoded);
        assert_eq!(model.digest(), decoded.digest());
        let expected = model.encode(std::slice::from_ref(&values)).unwrap();
        let actual = decoded.encode(&[values]).unwrap();
        assert_eq!(
            expected
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap(),
            actual
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
        );
    }

    #[test]
    fn hilbert_curve_is_adjacent_on_small_grid() {
        let mut points = Vec::new();
        for x in 0..4 {
            for y in 0..4 {
                let mut point = [x, y];
                axes_to_transpose(&mut point);
                points.push((interleave(&point), x, y));
            }
        }
        points.sort_unstable();
        assert!(
            points
                .windows(2)
                .all(|pair| { pair[0].1.abs_diff(pair[1].1) + pair[0].2.abs_diff(pair[1].2) == 1 })
        );
    }
}
