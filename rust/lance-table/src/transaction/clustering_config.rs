// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::clustering::{
    ClusteringSpec, LiquidClusteringState, MAX_CLUSTERING_COLUMNS, validate_data_type,
};
use lance_core::datatypes::LANCE_UNENFORCED_CLUSTERING_KEY_POSITION;
use lance_core::{Error, Result};

use crate::format::Manifest;

/// Validate the declaration against the final schema and enforce monotonic layout generations.
pub(super) fn validate_clustering_config_transition(
    current_manifest: Option<&Manifest>,
    next_manifest: &Manifest,
) -> Result<()> {
    if let Some(current) = current_manifest {
        let previous_key = current
            .schema
            .unenforced_clustering_key()
            .into_iter()
            .map(|field| field.id)
            .collect::<Vec<_>>();
        let next_key = next_manifest
            .schema
            .unenforced_clustering_key()
            .into_iter()
            .map(|field| field.id)
            .collect::<Vec<_>>();
        if !previous_key.is_empty() && next_key != previous_key {
            return Err(Error::invalid_input(
                "the unenforced clustering key cannot be changed once set",
            ));
        }
    }
    let Some(next_state) = next_manifest.liquid_clustering.as_ref() else {
        return Ok(());
    };
    next_state.validate()?;
    ClusteringSpec::from_state(Some(next_state), &next_manifest.schema)?;

    if let Some(current) = current_manifest {
        if let Some(previous) = current.liquid_clustering.as_ref() {
            if next_state.generation < previous.generation {
                return Err(Error::invalid_input(format!(
                    "clustering generation cannot decrease from {} to {}",
                    previous.generation, next_state.generation
                )));
            }
            if (next_state.algorithm != previous.algorithm
                || (next_state.enabled && !previous.enabled))
                && next_state.generation == previous.generation
            {
                return Err(Error::invalid_input(format!(
                    "enabling clustering or changing its algorithm requires a generation greater \
                     than {}; got {}",
                    previous.generation, next_state.generation
                )));
            }
        } else if let Some(max_fragment_version) = current
            .fragment_clustering_generations()
            .into_iter()
            .flatten()
            .max()
            && next_state.generation <= max_fragment_version
        {
            return Err(Error::invalid_input(format!(
                "clustering generation must be greater than existing fragment generation \
                 {max_fragment_version}; got {}",
                next_state.generation
            )));
        }
    }
    Ok(())
}

pub(super) fn apply_clustering_update(
    manifest: &mut Manifest,
    state: LiquidClusteringState,
    clustering_fields: &[i32],
) -> Result<()> {
    state.validate()?;
    if !state.enabled {
        if !clustering_fields.is_empty() {
            return Err(Error::invalid_input(
                "disabling clustering must not specify clustering fields",
            ));
        }
        let Some(current) = manifest.liquid_clustering else {
            return Err(Error::invalid_input(
                "cannot disable clustering because it has not been configured",
            ));
        };
        if state.generation != current.generation || state.algorithm != current.algorithm {
            return Err(Error::invalid_input(format!(
                "disabling clustering must preserve generation {} and algorithm {:?}; got \
                 generation {} and algorithm {:?}",
                current.generation, current.algorithm, state.generation, state.algorithm
            )));
        }
        manifest.liquid_clustering = Some(state);
        return Ok(());
    }

    if clustering_fields.is_empty() {
        return Err(Error::invalid_input(
            "enabling clustering requires at least one clustering field",
        ));
    }
    if clustering_fields.len() > MAX_CLUSTERING_COLUMNS {
        return Err(Error::invalid_input(format!(
            "clustering supports at most {MAX_CLUSTERING_COLUMNS} columns, got {}",
            clustering_fields.len()
        )));
    }
    let mut seen = std::collections::HashSet::with_capacity(clustering_fields.len());
    for field_id in clustering_fields {
        if !seen.insert(*field_id) {
            return Err(Error::invalid_input(format!(
                "clustering fields must be unique; duplicate field id {field_id}"
            )));
        }
        let field = manifest
            .schema
            .fields
            .iter()
            .find(|field| field.id == *field_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "clustering field id {field_id} must identify a top-level field"
                ))
            })?;
        validate_data_type(&field.data_type())?;
    }

    let existing_key = manifest
        .schema
        .unenforced_clustering_key()
        .into_iter()
        .map(|field| field.id)
        .collect::<Vec<_>>();
    if !existing_key.is_empty() && existing_key != clustering_fields {
        return Err(Error::invalid_input(
            "the unenforced clustering key cannot be changed once set",
        ));
    }
    if existing_key.is_empty() {
        for (position, field_id) in (1_u32..).zip(clustering_fields) {
            let field = manifest.schema.field_by_id_mut(*field_id).ok_or_else(|| {
                Error::internal(format!(
                    "validated clustering field id {field_id} disappeared from the schema"
                ))
            })?;
            field.unenforced_clustering_key_position = Some(position);
            field.metadata.insert(
                LANCE_UNENFORCED_CLUSTERING_KEY_POSITION.to_string(),
                position.to_string(),
            );
        }
    }
    manifest.liquid_clustering = Some(state);
    Ok(())
}
