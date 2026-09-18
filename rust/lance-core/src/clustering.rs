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
    /// An algorithm written by a newer Lance implementation.
    ///
    /// Readers preserve the raw protobuf value because clustering metadata does
    /// not affect logical row values. Operations that need to interpret the
    /// physical layout must reject algorithms they do not understand.
    Unknown(i32),
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
/// assert!(state.enabled());
/// # Ok::<(), lance_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub struct LiquidClusteringState {
    /// Whether new writes should use the declared clustering layout.
    enabled: bool,
    /// Monotonically increasing physical-layout generation.
    generation: u64,
    /// Algorithm defining the physical layout for this generation.
    algorithm: ClusteringAlgorithm,
}

impl LiquidClusteringState {
    /// Create a validated liquid-clustering state.
    pub fn new(enabled: bool, generation: u64, algorithm: ClusteringAlgorithm) -> Result<Self> {
        if let ClusteringAlgorithm::Unknown(value) = algorithm {
            return Err(Error::not_supported(format!(
                "cannot configure unknown liquid clustering algorithm {value}"
            )));
        }
        Self::from_persisted(enabled, generation, algorithm)
    }

    /// Restore persisted state, including an algorithm introduced by a newer writer.
    ///
    /// This is for format decoding only. New clustering declarations must use
    /// [`Self::new`], which accepts only algorithms implemented by this build.
    #[doc(hidden)]
    pub fn from_persisted(
        enabled: bool,
        generation: u64,
        algorithm: ClusteringAlgorithm,
    ) -> Result<Self> {
        if generation == 0 {
            return Err(Error::invalid_input(
                "clustering generation must be positive",
            ));
        }
        if algorithm == ClusteringAlgorithm::Unknown(0) {
            return Err(Error::invalid_input(
                "liquid clustering algorithm must be specified",
            ));
        }
        Ok(Self {
            enabled,
            generation,
            algorithm,
        })
    }

    /// Whether new writes should use the declared clustering layout.
    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Return the monotonically increasing physical-layout generation.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Return the algorithm defining the physical layout for this generation.
    pub const fn algorithm(&self) -> ClusteringAlgorithm {
        self.algorithm
    }

    /// Return this state with clustering disabled while retaining its generation.
    ///
    /// Retaining the generation prevents a later declaration from reusing an
    /// identifier that appeared in an earlier dataset version.
    pub const fn disabled(self) -> Self {
        Self {
            enabled: false,
            ..self
        }
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

    #[test]
    fn clustering_algorithm_must_be_known_when_configured() {
        let error =
            LiquidClusteringState::new(true, 1, ClusteringAlgorithm::Unknown(7)).unwrap_err();

        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("unknown"));
    }
}
