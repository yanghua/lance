// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Liquid-clustering state shared by table and execution crates.

use std::collections::HashSet;

use arrow_schema::DataType;

use crate::datatypes::Schema;
use crate::deepsize::DeepSizeOf;
use crate::{Error, Result, is_system_column};

pub const CLUSTERING_ALGORITHM_REVISION: &str = "typed-quantile-rank-v1";
pub const MAX_CLUSTERING_COLUMNS: usize = 4;

/// Physical layout algorithm used by liquid clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub enum ClusteringAlgorithm {
    TypedQuantileRankV1,
}

impl ClusteringAlgorithm {
    pub const fn revision(self) -> &'static str {
        match self {
            Self::TypedQuantileRankV1 => CLUSTERING_ALGORITHM_REVISION,
        }
    }
}

/// Persisted table-level liquid-clustering state.
///
/// The ordered clustering key is intentionally not repeated here. It is stored
/// on schema fields as the unenforced clustering key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub struct LiquidClusteringState {
    pub enabled: bool,
    pub generation: u64,
    pub algorithm: ClusteringAlgorithm,
}

impl LiquidClusteringState {
    pub fn new(enabled: bool, generation: u64, algorithm: ClusteringAlgorithm) -> Result<Self> {
        let state = Self {
            enabled,
            generation,
            algorithm,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<()> {
        if self.generation == 0 {
            return Err(Error::invalid_input(
                "clustering generation must be positive",
            ));
        }
        Ok(())
    }
}

/// Desired clustering columns and the internally managed layout generation.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusteringSpec {
    pub columns: Vec<String>,
    pub algorithm: ClusteringAlgorithm,
    pub generation: u64,
}

impl ClusteringSpec {
    pub fn new(columns: Vec<String>, generation: u64) -> Result<Self> {
        let spec = Self {
            columns,
            algorithm: ClusteringAlgorithm::TypedQuantileRankV1,
            generation,
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
        if self.generation == 0 {
            return Err(Error::invalid_input(
                "clustering generation must be positive",
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

    pub fn validate_current_algorithm(&self) -> Result<()> {
        match self.algorithm {
            ClusteringAlgorithm::TypedQuantileRankV1 => Ok(()),
        }
    }

    /// Build an executable spec from typed manifest state and the schema key.
    pub fn from_state(
        state: Option<&LiquidClusteringState>,
        schema: &Schema,
    ) -> Result<Option<Self>> {
        let Some(state) = state.filter(|state| state.enabled) else {
            return Ok(None);
        };
        state.validate()?;
        let key = schema.unenforced_clustering_key();
        if key.is_empty() {
            return Err(Error::invalid_input(
                "enabled liquid clustering requires an unenforced clustering key",
            ));
        }
        if key.len() > MAX_CLUSTERING_COLUMNS {
            return Err(Error::invalid_input(format!(
                "clustering supports at most {MAX_CLUSTERING_COLUMNS} columns, got {}",
                key.len()
            )));
        }
        for (expected, field) in (1_u32..).zip(key.iter()) {
            if !schema
                .fields
                .iter()
                .any(|top_level| top_level.id == field.id)
            {
                return Err(Error::invalid_input(format!(
                    "clustering column {:?} must be a top-level column",
                    field.name
                )));
            }
            if field.unenforced_clustering_key_position != Some(expected) {
                return Err(Error::invalid_input(format!(
                    "unenforced clustering key positions must be unique and contiguous from 1; \
                     expected position {expected} for column {:?}, got {:?}",
                    field.name, field.unenforced_clustering_key_position
                )));
            }
            validate_data_type(&field.data_type())?;
        }
        Ok(Some(Self {
            columns: key.iter().map(|field| field.name.clone()).collect(),
            algorithm: state.algorithm,
            generation: state.generation,
        }))
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
    fn disabled_state_preserves_a_valid_generation() {
        assert!(
            LiquidClusteringState::new(false, 7, ClusteringAlgorithm::TypedQuantileRankV1).is_ok()
        );
        assert!(
            LiquidClusteringState::new(false, 0, ClusteringAlgorithm::TypedQuantileRankV1).is_err()
        );
    }
}
