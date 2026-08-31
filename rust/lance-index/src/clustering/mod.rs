// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Multi-column space-filling curve clustering.
//!
//! This module provides the building blocks for "liquid clustering": a
//! [`ClusteringSpec`] describing which columns to cluster by and how, and a
//! [`SpaceFillingEncoder`] that maps the clustering-key columns of a batch to a
//! single ordering value. Sorting rows by that value lays them out so that rows
//! close together in the multi-dimensional key space are close together on
//! disk, which makes zone-map data skipping effective across every key column
//! at once (instead of favoring a single leading column, as a plain
//! lexicographic sort would).
//!
//! Two curves are supported:
//!
//! * [`ClusteringCurve::ZOrder`] (Morton order) interleaves the bits of each
//!   normalized column. It is the cheapest to compute but has large "jumps" at
//!   power-of-two boundaries.
//! * [`ClusteringCurve::Hilbert`] applies the Skilling transform before
//!   interleaving. It preserves locality better (consecutive indices are always
//!   spatial neighbours) at a modest additional compute cost. This is the
//!   default.
//!
//! The encoder produces a big-endian [`FixedSizeBinaryArray`] so that the
//! lexicographic order of the bytes matches the numeric order of the underlying
//! curve index. This lets the write and compaction paths sort by the output
//! with the ordinary Arrow comparison kernels and generalizes to any number of
//! key columns (up to [`MAX_TOTAL_BITS`] combined bits).
//!
//! # Persistence
//!
//! The ordered clustering columns, curve, layout version, and per-dimension bit
//! width are stored together in dataset configuration under
//! `lance.clustering.*`. Keeping the liquid-clustering policy separate from
//! schema-level physical-ordering hints lets the policy be changed or cleared
//! without mutating those stable hints.

use std::collections::HashMap;

use arrow_array::builder::FixedSizeBinaryBuilder;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};

mod sort;
pub use sort::cluster_sort_stream;

/// Config key holding the space-filling curve name (`hilbert` or `zorder`).
pub const CLUSTERING_CURVE_KEY: &str = "lance.clustering.curve";
/// Config key holding the ordered clustering column names as a JSON array.
pub const CLUSTERING_COLUMNS_KEY: &str = "lance.clustering.columns";
/// Config key holding the clustering version, bumped whenever the curve
/// changes or a full recluster is forced so already-written fragments can be
/// detected as under-clustered.
pub const CLUSTERING_VERSION_KEY: &str = "lance.clustering.version";
/// Config key holding the per-column bit width used when quantizing values.
pub const CLUSTERING_BITS_PER_DIM_KEY: &str = "lance.clustering.bits_per_dim";

const CLUSTERING_CONFIG_PREFIX: &str = "lance.clustering.";

fn is_known_clustering_config_key(key: &str) -> bool {
    matches!(
        key,
        CLUSTERING_COLUMNS_KEY
            | CLUSTERING_CURVE_KEY
            | CLUSTERING_VERSION_KEY
            | CLUSTERING_BITS_PER_DIM_KEY
    )
}

/// Default per-column bit width. 16 bits per dimension keeps four columns
/// inside a 64-bit index while still distinguishing 65,536 buckets per column.
pub const DEFAULT_BITS_PER_DIM: u32 = 16;

/// Maximum combined bit width (`num_columns * bits_per_dim`). The encoder
/// accumulates the interleaved index in a `u128`, so the total cannot exceed
/// 128 bits.
pub const MAX_TOTAL_BITS: u32 = 128;

/// Validate that a data type can be encoded as a clustering coordinate.
///
/// Clustering currently supports booleans, signed and unsigned integers, and
/// floating-point values. Callers should validate declared clustering columns
/// before persisting the clustering config so an unsupported declaration is
/// not committed.
pub fn validate_clustering_data_type(data_type: &DataType) -> Result<()> {
    match data_type {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => Ok(()),
        other => Err(unsupported_clustering_type(other)),
    }
}

fn unsupported_clustering_type(data_type: &DataType) -> Error {
    Error::invalid_input(format!(
        "clustering does not support column type {data_type:?}; supported types are \
         integers, floats, and boolean"
    ))
}

/// The space-filling curve used to order clustering-key values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ClusteringCurve {
    /// Z-order (Morton) curve: interleave the bits of each column.
    #[serde(rename = "zorder")]
    ZOrder,
    /// Hilbert curve: better locality, no large jumps. Default.
    #[serde(rename = "hilbert")]
    #[default]
    Hilbert,
}

impl ClusteringCurve {
    /// The canonical lowercase name used in table configuration.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ZOrder => "zorder",
            Self::Hilbert => "hilbert",
        }
    }
}

impl std::str::FromStr for ClusteringCurve {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "zorder" | "z-order" | "morton" => Ok(Self::ZOrder),
            "hilbert" => Ok(Self::Hilbert),
            other => Err(Error::invalid_input(format!(
                "unknown clustering curve \"{other}\"; expected \"hilbert\" or \"zorder\""
            ))),
        }
    }
}

/// Description of how a dataset is clustered.
///
/// All fields are persisted in dataset config under `lance.clustering.*`, with
/// `columns` encoded as a JSON array to preserve ordering and arbitrary column
/// names. This policy is independent of schema-level physical-ordering hints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusteringSpec {
    /// Ordered list of clustering-key column names.
    pub columns: Vec<String>,
    /// Space-filling curve used to order the key values.
    pub curve: ClusteringCurve,
    /// Version of the clustering layout. Bumped whenever the layout is
    /// invalidated (curve change or a forced recluster); fragments written
    /// under an older version are treated as under-clustered and re-clustered
    /// opportunistically.
    pub version: u64,
    /// Per-column bit width used when quantizing values.
    pub bits_per_dim: u32,
}

impl ClusteringSpec {
    /// Create a new spec at version 1 with the given columns and curve, using
    /// the default per-column bit width.
    pub fn new(columns: Vec<String>, curve: ClusteringCurve) -> Result<Self> {
        Self::with_bits(columns, curve, 1, DEFAULT_BITS_PER_DIM)
    }

    /// Create a spec with an explicit version and bit width, validating the
    /// combined bit budget.
    pub fn with_bits(
        columns: Vec<String>,
        curve: ClusteringCurve,
        version: u64,
        bits_per_dim: u32,
    ) -> Result<Self> {
        let spec = Self {
            columns,
            curve,
            version,
            bits_per_dim,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Validate the spec invariants.
    ///
    /// This also validates specs built through struct literals or
    /// deserialization instead of [`ClusteringSpec::new`] or
    /// [`ClusteringSpec::with_bits`]. Callers that persist a caller-provided
    /// spec should invoke this before committing metadata.
    pub fn validate(&self) -> Result<()> {
        if self.columns.is_empty() {
            return Err(Error::invalid_input(
                "clustering spec must have at least one column",
            ));
        }
        if self.version == 0 {
            return Err(Error::invalid_input(
                "clustering version must be at least 1, got 0",
            ));
        }
        let mut first_positions = HashMap::with_capacity(self.columns.len());
        for (position, column) in self.columns.iter().enumerate() {
            if let Some(first_position) = first_positions.insert(column.as_str(), position) {
                return Err(Error::invalid_input(format!(
                    "clustering columns must be unique; duplicate column {column:?} at \
                     positions {first_position} and {position}"
                )));
            }
        }
        validate_bits(self.columns.len(), self.bits_per_dim)
    }

    /// Serialize the complete spec into `lance.clustering.*` configuration
    /// entries.
    pub fn to_config(&self) -> Vec<(String, String)> {
        let columns = serde_json::Value::Array(
            self.columns
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        )
        .to_string();
        vec![
            (CLUSTERING_COLUMNS_KEY.to_string(), columns),
            (
                CLUSTERING_CURVE_KEY.to_string(),
                self.curve.as_str().to_string(),
            ),
            (CLUSTERING_VERSION_KEY.to_string(), self.version.to_string()),
            (
                CLUSTERING_BITS_PER_DIM_KEY.to_string(),
                self.bits_per_dim.to_string(),
            ),
        ]
    }

    /// The config keys this spec writes, for use when clearing a declaration.
    pub fn config_keys() -> [&'static str; 4] {
        [
            CLUSTERING_COLUMNS_KEY,
            CLUSTERING_CURVE_KEY,
            CLUSTERING_VERSION_KEY,
            CLUSTERING_BITS_PER_DIM_KEY,
        ]
    }

    /// Read a clustering spec from a table configuration map.
    ///
    /// No `lance.clustering.*` entries means clustering is not declared. A
    /// declaration must contain all four keys written by [`Self::to_config`]; a
    /// partial declaration is rejected instead of being silently defaulted.
    pub fn from_config(config: &HashMap<String, String>) -> Result<Option<Self>> {
        let mut has_clustering_config = false;
        for key in config
            .keys()
            .filter(|key| key.starts_with(CLUSTERING_CONFIG_PREFIX))
        {
            has_clustering_config = true;
            if !is_known_clustering_config_key(key) {
                return Err(Error::invalid_input(format!(
                    "unknown clustering config key {key:?}; expected one of {}, {}, {}, or {}",
                    CLUSTERING_COLUMNS_KEY,
                    CLUSTERING_CURVE_KEY,
                    CLUSTERING_VERSION_KEY,
                    CLUSTERING_BITS_PER_DIM_KEY
                )));
            }
        }
        if !has_clustering_config {
            return Ok(None);
        }
        let required_value = |key| {
            config.get(key).ok_or_else(|| {
                Error::invalid_input(format!(
                    "incomplete clustering config: missing required {key}; all of \
                     {CLUSTERING_COLUMNS_KEY}, {CLUSTERING_CURVE_KEY}, \
                     {CLUSTERING_VERSION_KEY}, and {CLUSTERING_BITS_PER_DIM_KEY} must be set \
                     together"
                ))
            })
        };
        let columns = required_value(CLUSTERING_COLUMNS_KEY)?;
        let curve = required_value(CLUSTERING_CURVE_KEY)?;
        let version = required_value(CLUSTERING_VERSION_KEY)?;
        let bits_per_dim = required_value(CLUSTERING_BITS_PER_DIM_KEY)?;
        let columns = serde_json::from_str::<Vec<String>>(columns).map_err(|e| {
            Error::invalid_input(format!(
                "invalid {CLUSTERING_COLUMNS_KEY} value {columns:?}: expected a JSON array of \
                 column-name strings: {e}"
            ))
        })?;
        let curve = match curve.as_str() {
            "hilbert" => ClusteringCurve::Hilbert,
            "zorder" => ClusteringCurve::ZOrder,
            other => {
                return Err(Error::invalid_input(format!(
                    "invalid {CLUSTERING_CURVE_KEY} value {other:?}: expected \
                     \"hilbert\" or \"zorder\""
                )));
            }
        };
        let version = version.parse::<u64>().map_err(|e| {
            Error::invalid_input(format!(
                "invalid {CLUSTERING_VERSION_KEY} value {version:?}: {e}"
            ))
        })?;
        let bits_per_dim = bits_per_dim.parse::<u32>().map_err(|e| {
            Error::invalid_input(format!(
                "invalid {CLUSTERING_BITS_PER_DIM_KEY} value {bits_per_dim:?}: {e}"
            ))
        })?;
        Some(Self::with_bits(columns, curve, version, bits_per_dim)).transpose()
    }
}

fn validate_bits(num_columns: usize, bits_per_dim: u32) -> Result<()> {
    if !(1..=64).contains(&bits_per_dim) {
        return Err(Error::invalid_input(format!(
            "clustering bits_per_dim must be in 1..=64, got {bits_per_dim}"
        )));
    }
    let total = num_columns
        .checked_mul(bits_per_dim as usize)
        .ok_or_else(|| Error::invalid_input("clustering key bit width overflowed usize"))?;
    if total > MAX_TOTAL_BITS as usize {
        return Err(Error::invalid_input(format!(
            "clustering key is too wide: {num_columns} columns * {bits_per_dim} bits = \
                 {total} bits exceeds the {MAX_TOTAL_BITS}-bit limit; reduce the column count \
                 or bits_per_dim"
        )));
    }
    Ok(())
}

/// Encodes clustering-key columns into a single space-filling-curve ordering
/// value per row.
#[derive(Debug, Clone)]
pub struct SpaceFillingEncoder {
    curve: ClusteringCurve,
    bits_per_dim: u32,
}

impl SpaceFillingEncoder {
    /// Create an encoder for the given curve and per-column bit width.
    pub fn new(curve: ClusteringCurve, bits_per_dim: u32) -> Result<Self> {
        if !(1..=64).contains(&bits_per_dim) {
            return Err(Error::invalid_input(format!(
                "clustering bits_per_dim must be in 1..=64, got {bits_per_dim}"
            )));
        }
        Ok(Self {
            curve,
            bits_per_dim,
        })
    }

    /// Build an encoder matching a [`ClusteringSpec`].
    pub fn from_spec(spec: &ClusteringSpec) -> Result<Self> {
        spec.validate()?;
        Self::new(spec.curve, spec.bits_per_dim)
    }

    /// Number of bytes in each encoded ordering value for the given column count.
    pub fn output_width(&self, num_columns: usize) -> Result<usize> {
        validate_bits(num_columns, self.bits_per_dim)?;
        let total_bits = num_columns
            .checked_mul(self.bits_per_dim as usize)
            .ok_or_else(|| Error::invalid_input("clustering key bit width overflowed usize"))?;
        Ok(total_bits.div_ceil(8))
    }

    /// Encode the clustering-key columns into one ordering value per row.
    ///
    /// The output is a big-endian [`FixedSizeBinaryArray`]; comparing two
    /// outputs lexicographically yields the same order as comparing their
    /// curve indices. Null values are mapped to the maximum coordinate in their
    /// dimension, where they can tie the maximum non-null value at full
    /// precision. A multi-dimensional space-filling curve does not provide
    /// row-level `NULLS LAST` ordering.
    pub fn encode(&self, key_columns: &[ArrayRef]) -> Result<ArrayRef> {
        if key_columns.is_empty() {
            return Err(Error::invalid_input(
                "clustering encoder requires at least one key column",
            ));
        }
        validate_bits(key_columns.len(), self.bits_per_dim)?;

        let num_rows = key_columns[0].len();
        for (i, col) in key_columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(Error::invalid_input(format!(
                    "clustering key columns must all have the same length; column 0 has \
                         {num_rows} rows but column {i} has {}",
                    col.len()
                )));
            }
        }

        let bits = self.bits_per_dim;
        // Per-column quantized coordinates: coords[col][row], each in [0, 2^bits).
        let coords: Vec<Vec<u64>> = key_columns
            .iter()
            .map(|col| quantize_column(col, bits))
            .collect::<Result<_>>()?;

        let num_columns = key_columns.len();
        let width = self.output_width(num_columns)?;
        let mut builder = FixedSizeBinaryBuilder::with_capacity(num_rows, width as i32);
        let mut point = vec![0u64; num_columns];
        for row in 0..num_rows {
            for (col, coord) in coords.iter().enumerate() {
                point[col] = coord[row];
            }
            if self.curve == ClusteringCurve::Hilbert {
                // Every coordinate is overwritten at the start of the next
                // iteration, so the reusable row buffer can be transformed in
                // place without allocating a clone per row.
                axes_to_transpose(&mut point, bits);
            }
            let index = interleave(&point, bits);
            let be = index.to_be_bytes();
            builder.append_value(&be[be.len() - width..]).map_err(|e| {
                Error::invalid_input(format!("failed to build clustering index array: {e}"))
            })?;
        }
        Ok(std::sync::Arc::new(builder.finish()))
    }
}

/// Quantize one column to `bits` bits per value, mapping each value to an
/// order-preserving unsigned coordinate. Nulls become the maximum coordinate.
///
/// Quantization keeps the most-significant bits of a fixed mapping for the
/// value's complete type domain. It deliberately does not rescale using batch
/// min/max: the encoder is evaluated independently for each input batch, so a
/// data-dependent scale would make identical values encode differently across
/// batches and invalidate the global sort. As a consequence, a small range of
/// a wide integer type can share a coordinate at low `bits` values.
fn quantize_column(array: &ArrayRef, bits: u32) -> Result<Vec<u64>> {
    let keys = order_preserving_keys(array)?;
    let shift = 64 - bits;
    Ok(keys.into_iter().map(|k| k >> shift).collect())
}

/// Map each value of `array` to a 64-bit order-preserving key with the value's
/// most significant bits at the top of the word. Nulls map to `u64::MAX` so
/// they sort last within their dimension.
fn order_preserving_keys(array: &ArrayRef) -> Result<Vec<u64>> {
    macro_rules! unsigned {
        ($arr:expr, $width:expr) => {{
            let a = $arr;
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        u64::MAX
                    } else {
                        (a.value(i) as u64) << (64 - $width)
                    }
                })
                .collect()
        }};
    }
    macro_rules! signed {
        ($arr:expr, $unsigned:ty, $width:expr) => {{
            let a = $arr;
            let flip = 1 as $unsigned << ($width - 1);
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        u64::MAX
                    } else {
                        (((a.value(i) as $unsigned) ^ flip) as u64) << (64 - $width)
                    }
                })
                .collect()
        }};
    }

    let keys: Vec<u64> = match array.data_type() {
        DataType::UInt8 => unsigned!(as_array::<UInt8Array>(array)?, 8),
        DataType::UInt16 => unsigned!(as_array::<UInt16Array>(array)?, 16),
        DataType::UInt32 => unsigned!(as_array::<UInt32Array>(array)?, 32),
        DataType::UInt64 => unsigned!(as_array::<UInt64Array>(array)?, 64),
        DataType::Int8 => signed!(as_array::<Int8Array>(array)?, u8, 8),
        DataType::Int16 => signed!(as_array::<Int16Array>(array)?, u16, 16),
        DataType::Int32 => signed!(as_array::<Int32Array>(array)?, u32, 32),
        DataType::Int64 => signed!(as_array::<Int64Array>(array)?, u64, 64),
        DataType::Float32 => {
            let a = as_array::<Float32Array>(array)?;
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        u64::MAX
                    } else {
                        (order_preserving_f32(a.value(i)) as u64) << 32
                    }
                })
                .collect()
        }
        DataType::Float64 => {
            let a = as_array::<Float64Array>(array)?;
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        u64::MAX
                    } else {
                        order_preserving_f64(a.value(i))
                    }
                })
                .collect()
        }
        DataType::Boolean => {
            let a = as_array::<BooleanArray>(array)?;
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        u64::MAX
                    } else if a.value(i) {
                        1u64 << 63
                    } else {
                        0
                    }
                })
                .collect()
        }
        other => {
            return Err(unsupported_clustering_type(other));
        }
    };
    Ok(keys)
}

fn as_array<T: Array + 'static>(array: &ArrayRef) -> Result<&T> {
    array.as_any().downcast_ref::<T>().ok_or_else(|| {
        Error::invalid_input(format!(
            "clustering: failed to downcast array of type {:?}",
            array.data_type()
        ))
    })
}

/// Order-preserving map from `f32` to `u32`: increasing float value yields an
/// increasing integer, with negatives ordered before positives.
fn order_preserving_f32(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits & 0x8000_0000 != 0 {
        // Negative (or negative zero): flip all bits so more-negative sorts lower.
        !bits
    } else {
        // Positive: set the sign bit so all positives sort above all negatives.
        bits | 0x8000_0000
    }
}

/// Order-preserving map from `f64` to `u64`.
fn order_preserving_f64(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000_0000_0000
    }
}

/// Interleave the low `bits` bits of each coordinate, most-significant bit
/// first, into a single index. With `d` coordinates this packs `d * bits` bits:
/// the MSB of coordinate 0 becomes the highest bit of the result.
fn interleave(point: &[u64], bits: u32) -> u128 {
    let mut acc: u128 = 0;
    for bit in (0..bits).rev() {
        for &coord in point {
            let b = (coord >> bit) & 1;
            acc = (acc << 1) | b as u128;
        }
    }
    acc
}

/// Convert axis coordinates to their Hilbert "transpose" representation in
/// place, after which [`interleave`] yields the Hilbert curve distance.
///
/// This is the classic Skilling (2004) `AxesToTranspose` algorithm. Each entry
/// of `x` uses the low `bits` bits.
fn axes_to_transpose(x: &mut [u64], bits: u32) {
    let n = x.len();
    let m: u64 = 1 << (bits - 1);

    // Inverse undo excess work.
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..n {
            if x[i] & q != 0 {
                x[0] ^= p; // invert low bits of the leading axis
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }

    // Gray encode.
    for i in 1..n {
        x[i] ^= x[i - 1];
    }
    let mut t = 0u64;
    q = m;
    while q > 1 {
        if x[n - 1] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for xi in x.iter_mut() {
        *xi ^= t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::sync::Arc;

    fn encode_rows(encoder: &SpaceFillingEncoder, columns: &[ArrayRef]) -> Vec<Vec<u8>> {
        let out = encoder.encode(columns).unwrap();
        let fsb = out
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
            .unwrap();
        (0..fsb.len()).map(|i| fsb.value(i).to_vec()).collect()
    }

    #[test]
    fn single_column_preserves_order() {
        // A one-column clustering key must sort identically to the raw values.
        let values = Int32Array::from(vec![5, -3, 100, 0, -1000, 42]);
        let col: ArrayRef = Arc::new(values.clone());
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::ZOrder, 32).unwrap();
        let encoded = encode_rows(&encoder, std::slice::from_ref(&col));

        let mut by_value: Vec<usize> = (0..values.len()).collect();
        by_value.sort_by_key(|&i| values.value(i));
        let mut by_encoded: Vec<usize> = (0..encoded.len()).collect();
        by_encoded.sort_by(|&a, &b| encoded[a].cmp(&encoded[b]));
        assert_eq!(by_value, by_encoded);
    }

    #[test]
    fn output_width_matches_total_bits() {
        let a: ArrayRef = Arc::new(UInt8Array::from(vec![1u8, 2, 3]));
        let b: ArrayRef = Arc::new(UInt8Array::from(vec![4u8, 5, 6]));
        // 2 columns * 12 bits = 24 bits -> 3 bytes.
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::Hilbert, 12).unwrap();
        let out = encoder.encode(&[a, b]).unwrap();
        let fsb = out
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
            .unwrap();
        assert_eq!(fsb.value_length(), 3);
    }

    #[test]
    fn output_width_rejects_invalid_dimension_counts() {
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::Hilbert, 64).unwrap();
        let error = encoder
            .output_width(usize::MAX)
            .expect_err("an overflowing dimension count must be rejected");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("bit width"));
    }

    #[test]
    fn hilbert_two_columns_is_a_bijection_with_adjacency() {
        // Enumerate a 4x4 grid (2 bits per dimension). Hilbert indices must be a
        // permutation of 0..16, and consecutive indices must be spatial
        // neighbours (Manhattan distance exactly 1) — the defining Hilbert
        // property that Z-order lacks.
        //
        // `quantize_column` keeps the most-significant `bits_per_dim` bits, so
        // the grid coordinate must live in the top bits of the u8 value: encode
        // coordinate c (0..4) as `c << 6`.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for x in 0..4u8 {
            for y in 0..4u8 {
                xs.push(x << 6);
                ys.push(y << 6);
            }
        }
        let cols: Vec<ArrayRef> = vec![
            Arc::new(UInt8Array::from(xs.clone())),
            Arc::new(UInt8Array::from(ys.clone())),
        ];
        // 2 bits per dim so the top two bits of each value map onto a 4x4 grid.
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::Hilbert, 2).unwrap();
        let encoded = encode_rows(&encoder, &cols);

        let mut order: Vec<usize> = (0..encoded.len()).collect();
        order.sort_by(|&a, &b| encoded[a].cmp(&encoded[b]));

        // Distinct indices (bijection): all 16 encodings are unique.
        let mut uniq: Vec<&Vec<u8>> = encoded.iter().collect();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 16, "hilbert indices must be distinct");

        // Adjacency: consecutive points along the curve differ by 1 grid step.
        for w in order.windows(2) {
            let (i, j) = (w[0], w[1]);
            let dx = (xs[i] >> 6) as i32 - (xs[j] >> 6) as i32;
            let dy = (ys[i] >> 6) as i32 - (ys[j] >> 6) as i32;
            assert_eq!(dx.abs() + dy.abs(), 1, "hilbert steps must be neighbours");
        }
    }

    #[test]
    fn zorder_groups_nearby_points() {
        // Two points sharing high bits in both dims should be closer in z-order
        // than a point that differs in the high bits.
        let xs = UInt8Array::from(vec![0u8, 1, 200]);
        let ys = UInt8Array::from(vec![0u8, 1, 200]);
        let cols: Vec<ArrayRef> = vec![Arc::new(xs), Arc::new(ys)];
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::ZOrder, 8).unwrap();
        let e = encode_rows(&encoder, &cols);
        let d01 = e[0]
            .iter()
            .zip(&e[1])
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .sum::<i32>();
        let d02 = e[0]
            .iter()
            .zip(&e[2])
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .sum::<i32>();
        assert!(
            d01 < d02,
            "adjacent points should encode closer than distant ones"
        );
    }

    #[test]
    fn null_uses_max_coordinate_in_one_dimension() {
        let values = Int32Array::from(vec![Some(10), None, Some(-5), Some(1000)]);
        let col: ArrayRef = Arc::new(values);
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::Hilbert, 32).unwrap();
        let encoded = encode_rows(&encoder, std::slice::from_ref(&col));
        let mut order: Vec<usize> = (0..encoded.len()).collect();
        order.sort_by(|&a, &b| encoded[a].cmp(&encoded[b]));
        assert_eq!(*order.last().unwrap(), 1, "null row must sort last");
    }

    #[test]
    fn multidimensional_null_is_not_globally_last() {
        // A maximum coordinate is a per-dimension policy, not a row-level
        // NULLS LAST policy. Z-order interleaves (0, max) before (max, 0).
        let x: ArrayRef = Arc::new(UInt8Array::from(vec![0, 128]));
        let y: ArrayRef = Arc::new(UInt8Array::from(vec![None, Some(0)]));
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::ZOrder, 1).unwrap();
        let encoded = encode_rows(&encoder, &[x, y]);

        assert!(
            encoded[0] < encoded[1],
            "a row containing null can precede a fully non-null row on a multi-dimensional curve"
        );
    }

    #[test]
    fn float_order_including_negatives() {
        let values = Float64Array::from(vec![1.5, -2.0, 0.0, -0.0, 3.25, f64::NEG_INFINITY]);
        let col: ArrayRef = Arc::new(values.clone());
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::ZOrder, 64).unwrap();
        let encoded = encode_rows(&encoder, std::slice::from_ref(&col));
        let mut by_val: Vec<usize> = (0..values.len()).collect();
        by_val.sort_by(|&a, &b| values.value(a).partial_cmp(&values.value(b)).unwrap());
        let mut by_enc: Vec<usize> = (0..encoded.len()).collect();
        by_enc.sort_by(|&a, &b| encoded[a].cmp(&encoded[b]));
        // -0.0 and 0.0 compare equal in float ordering; normalize by value.
        let vals_by_val: Vec<f64> = by_val.iter().map(|&i| values.value(i)).collect();
        let vals_by_enc: Vec<f64> = by_enc.iter().map(|&i| values.value(i)).collect();
        assert_eq!(vals_by_val, vals_by_enc);
    }

    #[test]
    fn rejects_unsupported_type() {
        let col: ArrayRef = Arc::new(arrow_array::StringArray::from(vec!["a", "b"]));
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::Hilbert, 16).unwrap();
        let error = encoder
            .encode(std::slice::from_ref(&col))
            .expect_err("strings must not be accepted as clustering coordinates");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("Utf8"));
    }

    #[test]
    fn validates_clustering_data_types() {
        for data_type in [
            DataType::Boolean,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
        ] {
            validate_clustering_data_type(&data_type).unwrap();
        }

        let error = validate_clustering_data_type(&DataType::Utf8)
            .expect_err("strings must not be accepted as clustering columns");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("Utf8"));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let b: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let encoder = SpaceFillingEncoder::new(ClusteringCurve::Hilbert, 16).unwrap();
        assert!(encoder.encode(&[a, b]).is_err());
    }

    #[test]
    fn rejects_too_wide_key() {
        // 3 columns * 64 bits = 192 bits > 128-bit budget.
        assert!(
            ClusteringSpec::with_bits(
                vec!["a".into(), "b".into(), "c".into()],
                ClusteringCurve::Hilbert,
                1,
                64
            )
            .is_err()
        );
    }

    #[test]
    fn spec_rejects_zero_version() {
        let error = ClusteringSpec::with_bits(
            vec!["a".into()],
            ClusteringCurve::Hilbert,
            0,
            DEFAULT_BITS_PER_DIM,
        )
        .expect_err("version zero must not be accepted");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("version must be at least 1"));

        ClusteringSpec::with_bits(
            vec!["a".into()],
            ClusteringCurve::Hilbert,
            u64::MAX,
            DEFAULT_BITS_PER_DIM,
        )
        .unwrap();
    }

    #[test]
    fn spec_rejects_duplicate_columns() {
        let error = ClusteringSpec::new(
            vec!["a".into(), "b".into(), "a".into()],
            ClusteringCurve::Hilbert,
        )
        .expect_err("duplicate clustering columns must not be accepted");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        let message = error.to_string();
        assert!(message.contains("duplicate column \"a\""));
        assert!(message.contains("positions 0 and 2"));

        ClusteringSpec::new(vec!["a".into(), "A".into()], ClusteringCurve::Hilbert).unwrap();
    }

    #[test]
    fn from_spec_validates_manually_constructed_spec() {
        let spec = ClusteringSpec {
            columns: vec!["a".into()],
            curve: ClusteringCurve::Hilbert,
            version: 0,
            bits_per_dim: DEFAULT_BITS_PER_DIM,
        };
        let error = SpaceFillingEncoder::from_spec(&spec)
            .expect_err("the encoder must reject an invalid manually constructed spec");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("version must be at least 1"));
    }

    #[test]
    fn fixed_domain_quantization_is_batch_independent() {
        let encoder =
            SpaceFillingEncoder::new(ClusteringCurve::ZOrder, DEFAULT_BITS_PER_DIM).unwrap();
        let full: ArrayRef = Arc::new(Int32Array::from(vec![-100_000, 0, 65_536, 1_000_000]));
        let first: ArrayRef = Arc::new(Int32Array::from(vec![-100_000, 0]));
        let second: ArrayRef = Arc::new(Int32Array::from(vec![65_536, 1_000_000]));

        let expected = encode_rows(&encoder, &[full]);
        let actual: Vec<Vec<u8>> = encode_rows(&encoder, &[first])
            .into_iter()
            .chain(encode_rows(&encoder, &[second]))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn spec_config_round_trip() {
        let spec = ClusteringSpec::with_bits(
            vec!["a".into(), "b".into()],
            ClusteringCurve::Hilbert,
            7,
            20,
        )
        .unwrap();
        let mut config: HashMap<String, String> = spec.to_config().into_iter().collect();
        assert_eq!(config[CLUSTERING_COLUMNS_KEY], r#"["a","b"]"#);
        assert_eq!(
            ClusteringSpec::config_keys(),
            [
                CLUSTERING_COLUMNS_KEY,
                CLUSTERING_CURVE_KEY,
                CLUSTERING_VERSION_KEY,
                CLUSTERING_BITS_PER_DIM_KEY,
            ]
        );
        config.insert(
            "lance.clustering_other.key".to_string(),
            "value".to_string(),
        );
        let parsed = ClusteringSpec::from_config(&config).unwrap().unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn from_config_without_clustering_keys_is_none() {
        let config = HashMap::new();
        assert!(ClusteringSpec::from_config(&config).unwrap().is_none());

        let unrelated = HashMap::from([
            ("other.key".to_string(), "value".to_string()),
            (
                "lance.clustering_other.key".to_string(),
                "value".to_string(),
            ),
        ]);
        assert!(ClusteringSpec::from_config(&unrelated).unwrap().is_none());
    }

    #[test]
    fn from_config_rejects_partial_declaration() {
        let complete_config = HashMap::from([
            (CLUSTERING_COLUMNS_KEY.to_string(), r#"["x"]"#.to_string()),
            (CLUSTERING_CURVE_KEY.to_string(), "zorder".to_string()),
            (CLUSTERING_VERSION_KEY.to_string(), "1".to_string()),
            (CLUSTERING_BITS_PER_DIM_KEY.to_string(), "8".to_string()),
        ]);
        for missing_key in ClusteringSpec::config_keys() {
            let mut config = complete_config.clone();
            config.remove(missing_key);
            let error = ClusteringSpec::from_config(&config)
                .expect_err("a partial clustering declaration must be rejected");
            assert!(matches!(&error, Error::InvalidInput { .. }));
            let message = error.to_string();
            assert!(message.contains("incomplete clustering config"));
            assert!(message.contains(missing_key));
        }
    }

    #[test]
    fn from_config_rejects_unknown_clustering_keys() {
        let unknown_key = "lance.clustering.future_option";
        let mut configs = vec![HashMap::new()];
        configs.push(
            ClusteringSpec::new(vec!["x".into()], ClusteringCurve::Hilbert)
                .unwrap()
                .to_config()
                .into_iter()
                .collect(),
        );

        for mut config in configs {
            config.insert(unknown_key.to_string(), "value".to_string());
            let error = ClusteringSpec::from_config(&config)
                .expect_err("unknown keys in the clustering namespace must be rejected");
            assert!(matches!(&error, Error::InvalidInput { .. }));
            let message = error.to_string();
            assert!(message.contains("unknown clustering config key"));
            assert!(message.contains(unknown_key));
        }
    }

    #[rstest]
    #[case::empty_columns(r#"[]"#, "16", "at least one column")]
    #[case::duplicate_columns(r#"["x","x"]"#, "16", "duplicate column \"x\"")]
    #[case::zero_bits(r#"["x"]"#, "0", "bits_per_dim must be in 1..=64")]
    #[case::too_many_bits(r#"["x"]"#, "65", "bits_per_dim must be in 1..=64")]
    #[case::too_wide_key(r#"["a","b","c"]"#, "64", "exceeds the 128-bit limit")]
    fn from_config_rejects_semantically_invalid_declaration(
        #[case] columns: &str,
        #[case] bits_per_dim: &str,
        #[case] expected_message: &str,
    ) {
        let config = HashMap::from([
            (CLUSTERING_COLUMNS_KEY.to_string(), columns.to_string()),
            (CLUSTERING_CURVE_KEY.to_string(), "hilbert".to_string()),
            (CLUSTERING_VERSION_KEY.to_string(), "1".to_string()),
            (
                CLUSTERING_BITS_PER_DIM_KEY.to_string(),
                bits_per_dim.to_string(),
            ),
        ]);

        let error = ClusteringSpec::from_config(&config)
            .expect_err("a semantically invalid clustering declaration must be rejected");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains(expected_message),
            "expected {expected_message:?} in {error}"
        );
    }

    #[test]
    fn from_config_rejects_invalid_active_config() {
        for version in ["0", "not-a-version"] {
            let config = HashMap::from([
                (CLUSTERING_COLUMNS_KEY.to_string(), r#"["x"]"#.to_string()),
                (CLUSTERING_CURVE_KEY.to_string(), "hilbert".to_string()),
                (CLUSTERING_VERSION_KEY.to_string(), version.to_string()),
                (CLUSTERING_BITS_PER_DIM_KEY.to_string(), "16".to_string()),
            ]);
            let error = ClusteringSpec::from_config(&config)
                .expect_err("an invalid active clustering version must be rejected");
            assert!(matches!(&error, Error::InvalidInput { .. }));
            if version == "0" {
                assert!(error.to_string().contains("version must be at least 1"));
            } else {
                assert!(error.to_string().contains(CLUSTERING_VERSION_KEY));
            }
        }

        let config = HashMap::from([
            (CLUSTERING_COLUMNS_KEY.to_string(), r#"["x"]"#.to_string()),
            (CLUSTERING_CURVE_KEY.to_string(), "hilbert".to_string()),
            (CLUSTERING_VERSION_KEY.to_string(), "1".to_string()),
            (
                CLUSTERING_BITS_PER_DIM_KEY.to_string(),
                "not-a-width".to_string(),
            ),
        ]);
        let error = ClusteringSpec::from_config(&config)
            .expect_err("an invalid active bit width must be rejected");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(CLUSTERING_BITS_PER_DIM_KEY));
    }

    #[test]
    fn from_config_rejects_bad_columns_and_curve() {
        for columns in [r#"{"x":1}"#, r#"["x",1]"#, "not-json"] {
            let config = HashMap::from([
                (CLUSTERING_COLUMNS_KEY.to_string(), columns.to_string()),
                (CLUSTERING_CURVE_KEY.to_string(), "hilbert".to_string()),
                (CLUSTERING_VERSION_KEY.to_string(), "1".to_string()),
                (CLUSTERING_BITS_PER_DIM_KEY.to_string(), "16".to_string()),
            ]);
            let error = ClusteringSpec::from_config(&config)
                .expect_err("invalid clustering columns JSON must be rejected");
            assert!(matches!(&error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains(CLUSTERING_COLUMNS_KEY));
        }

        let config = HashMap::from([
            (CLUSTERING_COLUMNS_KEY.to_string(), r#"["x"]"#.to_string()),
            (CLUSTERING_VERSION_KEY.to_string(), "1".to_string()),
            (CLUSTERING_CURVE_KEY.to_string(), "spiral".to_string()),
            (CLUSTERING_BITS_PER_DIM_KEY.to_string(), "16".to_string()),
        ]);
        let error = ClusteringSpec::from_config(&config)
            .expect_err("an invalid active curve must be rejected");
        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("spiral"));
    }
}
