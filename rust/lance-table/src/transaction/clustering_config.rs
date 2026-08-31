// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Validation for the liquid-clustering declaration stored in manifest config.
//!
//! The encoder and public API live in `lance-index`, which already depends on
//! `lance-table`. The persistence boundary therefore keeps a small local model
//! of the config contract instead of introducing a dependency cycle.

use std::collections::{HashMap, HashSet};

use arrow_schema::DataType;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};

use crate::format::Manifest;

const CLUSTERING_CONFIG_PREFIX: &str = "lance.clustering.";
const CLUSTERING_COLUMNS_KEY: &str = "lance.clustering.columns";
const CLUSTERING_CURVE_KEY: &str = "lance.clustering.curve";
const CLUSTERING_VERSION_KEY: &str = "lance.clustering.version";
const CLUSTERING_BITS_PER_DIM_KEY: &str = "lance.clustering.bits_per_dim";
const MAX_TOTAL_BITS: usize = 128;

fn is_known_clustering_config_key(key: &str) -> bool {
    matches!(
        key,
        CLUSTERING_COLUMNS_KEY
            | CLUSTERING_CURVE_KEY
            | CLUSTERING_VERSION_KEY
            | CLUSTERING_BITS_PER_DIM_KEY
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClusteringCurve {
    Hilbert,
    ZOrder,
}

#[derive(Debug, PartialEq, Eq)]
struct ClusteringConfig {
    columns: Vec<String>,
    curve: ClusteringCurve,
    version: u64,
    bits_per_dim: u32,
}

impl ClusteringConfig {
    fn layout_eq(&self, other: &Self) -> bool {
        self.columns == other.columns
            && self.curve == other.curve
            && self.bits_per_dim == other.bits_per_dim
    }
}

/// Return the version of the complete clustering declaration on `manifest`.
///
/// This deliberately uses the same parser as transition validation so a
/// partial or malformed declaration cannot authorize a fragment stamp.
pub(super) fn clustering_version(manifest: &Manifest) -> Result<Option<u64>> {
    Ok(parse_clustering_config(&manifest.config)?.map(|config| config.version))
}

pub(super) fn contains_clustering_key(config: &HashMap<String, String>) -> bool {
    config
        .keys()
        .any(|key| key.starts_with(CLUSTERING_CONFIG_PREFIX))
}

/// Validate the liquid-clustering declaration on the manifest being published.
///
/// This is called from manifest construction, which is repeated against the
/// latest manifest on every commit retry. Besides validating the config update
/// itself, checking the final schema here catches a concurrent or independent
/// schema operation that would leave an existing declaration dangling.
pub(super) fn validate_clustering_config_transition(
    current_manifest: Option<&Manifest>,
    next_manifest: &Manifest,
) -> Result<()> {
    let Some(next) = parse_clustering_config(&next_manifest.config)? else {
        // Removing every reserved key is the supported way to disable liquid
        // clustering, including when repairing an incomplete declaration.
        return Ok(());
    };

    validate_clustering_columns(&next, &next_manifest.schema)?;

    let Some(current_manifest) = current_manifest else {
        return Ok(());
    };

    if let Some(current) = parse_clustering_config(&current_manifest.config)? {
        if next.version < current.version {
            return Err(Error::invalid_input(format!(
                "clustering version cannot decrease from {} to {}",
                current.version, next.version
            )));
        }
        if !next.layout_eq(&current) && next.version == current.version {
            return Err(Error::invalid_input(format!(
                "changing clustering columns, curve, or bits_per_dim requires a version \
                 greater than {}; got {}",
                current.version, next.version
            )));
        }
    } else if let Some(max_fragment_version) = current_manifest
        .fragment_clustering_versions()
        .iter()
        .copied()
        .flatten()
        .max()
        && next.version <= max_fragment_version
    {
        return Err(Error::invalid_input(format!(
            "clustering version must be greater than the maximum existing fragment \
             clustering version {max_fragment_version} when re-enabling clustering; got {}",
            next.version
        )));
    }

    Ok(())
}

fn parse_clustering_config(config: &HashMap<String, String>) -> Result<Option<ClusteringConfig>> {
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

    let columns_value = required_config_value(config, CLUSTERING_COLUMNS_KEY)?;
    let columns = serde_json::from_str::<Vec<String>>(columns_value).map_err(|error| {
        Error::invalid_input(format!(
            "invalid {CLUSTERING_COLUMNS_KEY} value {columns_value:?}: expected a JSON array \
             of column-name strings: {error}"
        ))
    })?;
    if columns.is_empty() {
        return Err(Error::invalid_input(
            "clustering config must declare at least one column",
        ));
    }
    let mut seen = HashSet::with_capacity(columns.len());
    for column in &columns {
        if !seen.insert(column) {
            return Err(Error::invalid_input(format!(
                "clustering columns must be unique; duplicate column {column:?}"
            )));
        }
    }

    let curve_value = required_config_value(config, CLUSTERING_CURVE_KEY)?;
    let curve = match curve_value.as_str() {
        "hilbert" => ClusteringCurve::Hilbert,
        "zorder" => ClusteringCurve::ZOrder,
        _ => {
            return Err(Error::invalid_input(format!(
                "invalid {CLUSTERING_CURVE_KEY} value {curve_value:?}: expected \
                 \"hilbert\" or \"zorder\""
            )));
        }
    };

    let version_value = required_config_value(config, CLUSTERING_VERSION_KEY)?;
    let version = version_value.parse::<u64>().map_err(|error| {
        Error::invalid_input(format!(
            "invalid {CLUSTERING_VERSION_KEY} value {version_value:?}: expected a positive \
             u64: {error}"
        ))
    })?;
    if version == 0 {
        return Err(Error::invalid_input(
            "clustering version must be at least 1, got 0",
        ));
    }

    let bits_value = required_config_value(config, CLUSTERING_BITS_PER_DIM_KEY)?;
    let bits_per_dim = bits_value.parse::<u32>().map_err(|error| {
        Error::invalid_input(format!(
            "invalid {CLUSTERING_BITS_PER_DIM_KEY} value {bits_value:?}: expected an integer \
             in 1..=64: {error}"
        ))
    })?;
    if !(1..=64).contains(&bits_per_dim) {
        return Err(Error::invalid_input(format!(
            "clustering bits_per_dim must be in 1..=64, got {bits_per_dim}"
        )));
    }
    let total_bits = columns
        .len()
        .checked_mul(bits_per_dim as usize)
        .ok_or_else(|| Error::invalid_input("clustering key bit width overflowed usize"))?;
    if total_bits > MAX_TOTAL_BITS {
        return Err(Error::invalid_input(format!(
            "clustering key is too wide: {} columns * {bits_per_dim} bits = {total_bits} bits \
             exceeds the {MAX_TOTAL_BITS}-bit limit",
            columns.len()
        )));
    }

    Ok(Some(ClusteringConfig {
        columns,
        curve,
        version,
        bits_per_dim,
    }))
}

fn required_config_value<'a>(config: &'a HashMap<String, String>, key: &str) -> Result<&'a String> {
    config.get(key).ok_or_else(|| {
        Error::invalid_input(format!(
            "incomplete clustering config: missing required {key}; all of \
             {CLUSTERING_COLUMNS_KEY}, {CLUSTERING_CURVE_KEY}, {CLUSTERING_VERSION_KEY}, and \
             {CLUSTERING_BITS_PER_DIM_KEY} must be set together"
        ))
    })
}

fn validate_clustering_columns(config: &ClusteringConfig, schema: &Schema) -> Result<()> {
    for column in &config.columns {
        let Some(field) = schema.fields.iter().find(|field| field.name == *column) else {
            if schema.field(column).is_some() {
                return Err(Error::invalid_input(format!(
                    "clustering column {column:?} is a nested path; only top-level columns are \
                     supported"
                )));
            }
            return Err(Error::invalid_input(format!(
                "clustering column {column:?} does not exist in the dataset schema"
            )));
        };
        let data_type = field.data_type();
        if !matches!(
            &data_type,
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
                | DataType::Float64
        ) {
            return Err(Error::invalid_input(format!(
                "clustering column {column:?} has unsupported type {data_type:?}; supported \
                 types are boolean, signed and unsigned integers, and float32/float64"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_config() -> HashMap<String, String> {
        HashMap::from([
            (CLUSTERING_COLUMNS_KEY.to_string(), r#"["x"]"#.to_string()),
            (CLUSTERING_CURVE_KEY.to_string(), "hilbert".to_string()),
            (CLUSTERING_VERSION_KEY.to_string(), "1".to_string()),
            (CLUSTERING_BITS_PER_DIM_KEY.to_string(), "16".to_string()),
        ])
    }

    #[test]
    fn parse_rejects_unknown_clustering_keys() {
        let unknown_key = "lance.clustering.future_option";
        for mut config in [HashMap::new(), complete_config()] {
            config.insert(unknown_key.to_string(), "value".to_string());
            let error = parse_clustering_config(&config)
                .expect_err("unknown keys in the clustering namespace must be rejected");
            assert!(matches!(&error, Error::InvalidInput { .. }));
            let message = error.to_string();
            assert!(message.contains("unknown clustering config key"));
            assert!(message.contains(unknown_key));
        }
    }

    #[test]
    fn parse_ignores_near_prefix_keys() {
        let config = HashMap::from([(
            "lance.clustering_other.key".to_string(),
            "value".to_string(),
        )]);
        assert_eq!(parse_clustering_config(&config).unwrap(), None);

        let mut config = complete_config();
        config.insert(
            "lance.clustering_other.key".to_string(),
            "value".to_string(),
        );
        assert!(parse_clustering_config(&config).unwrap().is_some());
    }
}
