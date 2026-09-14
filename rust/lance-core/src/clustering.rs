// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Persisted clustering declaration shared by table and execution crates.

use std::collections::{HashMap, HashSet};

use arrow_schema::DataType;

use crate::datatypes::Schema;
use crate::{Error, Result, is_system_column};

pub const CLUSTERING_CONFIG_PREFIX: &str = "lance.clustering.";
pub const CLUSTERING_COLUMNS_KEY: &str = "lance.clustering.columns";
pub const CLUSTERING_ALGORITHM_REVISION_KEY: &str = "lance.clustering.algorithm_revision";
pub const CLUSTERING_VERSION_KEY: &str = "lance.clustering.version";
pub const CLUSTERING_ALGORITHM_REVISION: &str = "typed-quantile-rank-v1";
pub const MAX_CLUSTERING_COLUMNS: usize = 4;

/// Desired clustering columns and the internally managed layout generation.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusteringSpec {
    pub columns: Vec<String>,
    pub algorithm_revision: String,
    pub version: u64,
}

impl ClusteringSpec {
    pub fn new(columns: Vec<String>, version: u64) -> Result<Self> {
        let spec = Self {
            columns,
            algorithm_revision: CLUSTERING_ALGORITHM_REVISION.to_string(),
            version,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        if self.columns.is_empty() {
            return Err(Error::invalid_input(
                "clustering requires at least one column",
            ));
        }
        if self.columns.len() > MAX_CLUSTERING_COLUMNS {
            return Err(Error::invalid_input(format!(
                "clustering supports at most {MAX_CLUSTERING_COLUMNS} columns, got {}",
                self.columns.len()
            )));
        }
        if self.version == 0 {
            return Err(Error::invalid_input("clustering version must be positive"));
        }
        if self.algorithm_revision.is_empty() {
            return Err(Error::invalid_input(
                "clustering algorithm revision must not be empty",
            ));
        }
        if let Some(column) = self.columns.iter().find(|column| column.is_empty()) {
            return Err(Error::invalid_input(format!(
                "clustering column names must not be empty: {column:?}"
            )));
        }
        if let Some(column) = self.columns.iter().find(|column| is_system_column(column)) {
            return Err(Error::invalid_input(format!(
                "clustering column {column:?} is a reserved system column"
            )));
        }
        let mut seen = HashSet::with_capacity(self.columns.len());
        if let Some(column) = self
            .columns
            .iter()
            .find(|column| !seen.insert(column.as_str()))
        {
            return Err(Error::invalid_input(format!(
                "clustering columns must be unique; duplicate column {column:?}"
            )));
        }
        Ok(())
    }

    pub fn validate_schema(&self, schema: &Schema) -> Result<()> {
        self.validate()?;
        for column in &self.columns {
            let field = schema
                .fields
                .iter()
                .find(|field| field.name == *column)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "clustering column {column:?} must name an existing top-level column"
                    ))
                })?;
            validate_data_type(&field.data_type())?;
        }
        Ok(())
    }

    pub fn from_config(config: &HashMap<String, String>) -> Result<Option<Self>> {
        let has_clustering = config
            .keys()
            .any(|key| key.starts_with(CLUSTERING_CONFIG_PREFIX));
        if !has_clustering {
            return Ok(None);
        }
        if let Some(key) = config.keys().find(|key| {
            key.starts_with(CLUSTERING_CONFIG_PREFIX)
                && key.as_str() != CLUSTERING_COLUMNS_KEY
                && key.as_str() != CLUSTERING_ALGORITHM_REVISION_KEY
                && key.as_str() != CLUSTERING_VERSION_KEY
        }) {
            return Err(Error::invalid_input(format!(
                "unknown clustering config key {key:?}"
            )));
        }
        let columns = config.get(CLUSTERING_COLUMNS_KEY).ok_or_else(|| {
            Error::invalid_input(format!(
                "incomplete clustering config: missing {CLUSTERING_COLUMNS_KEY}"
            ))
        })?;
        let version = config.get(CLUSTERING_VERSION_KEY).ok_or_else(|| {
            Error::invalid_input(format!(
                "incomplete clustering config: missing {CLUSTERING_VERSION_KEY}"
            ))
        })?;
        let algorithm_revision = config
            .get(CLUSTERING_ALGORITHM_REVISION_KEY)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "incomplete clustering config: missing {CLUSTERING_ALGORITHM_REVISION_KEY}"
                ))
            })?
            .clone();
        let columns = serde_json::from_str(columns).map_err(|error| {
            Error::invalid_input(format!(
                "invalid {CLUSTERING_COLUMNS_KEY}: expected a JSON string array: {error}"
            ))
        })?;
        let version = version.parse().map_err(|error| {
            Error::invalid_input(format!(
                "invalid {CLUSTERING_VERSION_KEY}: expected a positive u64: {error}"
            ))
        })?;
        let spec = Self {
            columns,
            algorithm_revision,
            version,
        };
        spec.validate()?;
        Ok(Some(spec))
    }

    pub fn to_config(&self) -> Result<[(String, String); 3]> {
        self.validate()?;
        Ok([
            (
                CLUSTERING_COLUMNS_KEY.to_string(),
                serde_json::to_string(&self.columns)?,
            ),
            (
                CLUSTERING_ALGORITHM_REVISION_KEY.to_string(),
                self.algorithm_revision.clone(),
            ),
            (CLUSTERING_VERSION_KEY.to_string(), self.version.to_string()),
        ])
    }

    pub fn config_keys() -> [&'static str; 3] {
        [
            CLUSTERING_COLUMNS_KEY,
            CLUSTERING_ALGORITHM_REVISION_KEY,
            CLUSTERING_VERSION_KEY,
        ]
    }

    pub fn validate_current_algorithm(&self) -> Result<()> {
        if self.algorithm_revision != CLUSTERING_ALGORITHM_REVISION {
            return Err(Error::not_supported(format!(
                "clustering algorithm revision {:?} is not supported by this build; \
                 redeclare the clustering columns to upgrade to {CLUSTERING_ALGORITHM_REVISION:?}",
                self.algorithm_revision
            )));
        }
        Ok(())
    }
}

pub fn validate_data_type(data_type: &DataType) -> Result<()> {
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
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Date32
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Decimal128(_, _) => Ok(()),
        other => Err(Error::invalid_input(format!(
            "clustering does not support column type {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_public_columns_contract() {
        for columns in [vec![], vec!["x".into(), "x".into()], vec!["_rowid".into()]] {
            assert!(ClusteringSpec::new(columns, 1).is_err());
        }
        assert!(
            ClusteringSpec::new(
                (0..=MAX_CLUSTERING_COLUMNS)
                    .map(|i| format!("k{i}"))
                    .collect(),
                1
            )
            .is_err()
        );
        assert!(ClusteringSpec::new(vec!["x".into()], 0).is_err());
    }

    #[test]
    fn config_round_trip_includes_algorithm_revision() {
        let spec = ClusteringSpec::new(vec!["x".into(), "y".into()], 7).unwrap();
        let config = HashMap::from(spec.to_config().unwrap());
        assert_eq!(ClusteringSpec::from_config(&config).unwrap(), Some(spec));
        assert_eq!(config.len(), 3);
        assert_eq!(
            config[CLUSTERING_ALGORITHM_REVISION_KEY],
            CLUSTERING_ALGORITHM_REVISION
        );
    }
}
