// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Persistent liquid-clustering state shared by table and execution crates.

use crate::deepsize::DeepSizeOf;
use crate::{Error, Result};

/// The physical-layout algorithm used by liquid clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub enum ClusteringAlgorithm {
    /// Order rows by per-column empirical quantile ranks encoded into one key.
    TypedQuantileRankV1,
}

/// Persistent table-level liquid-clustering state.
///
/// The ordered clustering key is stored separately on schema fields through
/// `Field::unenforced_clustering_key_position`. A disabled state remains
/// present so its generation cannot be reused if clustering is re-enabled.
///
/// # Example
///
/// ```
/// use lance_core::clustering::{ClusteringAlgorithm, LiquidClusteringState};
///
/// let state = LiquidClusteringState::new(
///     true,
///     1,
///     ClusteringAlgorithm::TypedQuantileRankV1,
/// )?;
/// assert!(state.enabled);
/// # Ok::<(), lance_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub struct LiquidClusteringState {
    /// Whether new writes should use the declared clustering layout.
    pub enabled: bool,
    /// Monotonically increasing physical-layout generation.
    pub generation: u64,
    /// Algorithm defining the physical layout for this generation.
    pub algorithm: ClusteringAlgorithm,
}

impl LiquidClusteringState {
    /// Create a validated liquid-clustering state.
    pub fn new(enabled: bool, generation: u64, algorithm: ClusteringAlgorithm) -> Result<Self> {
        if generation == 0 {
            return Err(Error::invalid_input(
                "clustering generation must be positive",
            ));
        }
        Ok(Self {
            enabled,
            generation,
            algorithm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clustering_generation_must_be_positive() {
        let error = LiquidClusteringState::new(true, 0, ClusteringAlgorithm::TypedQuantileRankV1)
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("must be positive"));
    }
}
