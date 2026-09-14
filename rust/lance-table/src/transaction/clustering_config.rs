// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::clustering::ClusteringSpec;
use lance_core::{Error, Result};

use crate::format::Manifest;

pub(super) fn clustering_version(manifest: &Manifest) -> Result<Option<u64>> {
    Ok(ClusteringSpec::from_config(&manifest.config)?.map(|spec| spec.version))
}

/// Validate the declaration against the final schema and enforce monotonic layout generations.
pub(super) fn validate_clustering_config_transition(
    current_manifest: Option<&Manifest>,
    next_manifest: &Manifest,
) -> Result<()> {
    let Some(next) = ClusteringSpec::from_config(&next_manifest.config)? else {
        return Ok(());
    };
    next.validate_schema(&next_manifest.schema)?;

    if let Some(current) = current_manifest {
        if let Some(previous) = ClusteringSpec::from_config(&current.config)? {
            if next.version < previous.version {
                return Err(Error::invalid_input(format!(
                    "clustering version cannot decrease from {} to {}",
                    previous.version, next.version
                )));
            }
            if (next.columns != previous.columns
                || next.algorithm_revision != previous.algorithm_revision)
                && next.version == previous.version
            {
                return Err(Error::invalid_input(format!(
                    "changing clustering columns or algorithm revision requires a version greater \
                     than {}; got {}",
                    previous.version, next.version
                )));
            }
        } else if let Some(max_fragment_version) = current
            .fragment_clustering_versions()
            .into_iter()
            .flatten()
            .max()
            && next.version <= max_fragment_version
        {
            return Err(Error::invalid_input(format!(
                "clustering version must be greater than existing fragment version \
                 {max_fragment_version}; got {}",
                next.version
            )));
        }
    }
    next.validate_current_algorithm()
}
