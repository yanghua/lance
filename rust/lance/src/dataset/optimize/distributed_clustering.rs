// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Versioned coordination protocol for distributed liquid clustering.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use lance_core::ROW_ADDR;
use lance_core::clustering::ClusteringSpec;
use lance_index::clustering::{ClusteringModel, PartialClusteringModel, RowDigest};
use lance_table::format::Fragment;
use prost::Message;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::{
    CompactionMetrics, CompactionMode, CompactionOptions, IgnoreRemap, RewriteResult, TaskData,
    collect_metrics, commit_compaction_internal, limit_tasks_to_source_budget,
    maintain_clustering_zonemaps, task_source_bytes,
};
use crate::dataset::Dataset;
use crate::index::DatasetIndexExt;
use crate::recluster_pb;
use crate::{Error, Result};

const PROTOCOL_VERSION: u32 = 1;

struct PlanningUnit {
    fragments: Vec<Fragment>,
    rows: usize,
    bytes: u64,
    group_id: Option<String>,
    is_under_clustered: bool,
    is_excluded: bool,
}

/// One atomic source-fragment replacement in a distributed reclustering plan.
#[derive(Debug, Clone, PartialEq)]
pub struct ReclusterGroup {
    id: Uuid,
    source_fragments: Vec<Fragment>,
    expected_live_rows: u64,
}

impl ReclusterGroup {
    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn source_fragments(&self) -> &[Fragment] {
        &self.source_fragments
    }

    pub fn source_fragment_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.source_fragments.iter().map(|fragment| fragment.id)
    }

    pub fn expected_live_rows(&self) -> u64 {
        self.expected_live_rows
    }

    pub(super) fn task_data(&self) -> TaskData {
        TaskData {
            fragments: self.source_fragments.clone(),
        }
    }
}

/// Opaque, versioned description of distributed reclustering work.
#[derive(Debug, Clone, PartialEq)]
pub struct ReclusterPlan {
    id: Uuid,
    dataset_uri: String,
    read_version: u64,
    spec: ClusteringSpec,
    groups: Vec<ReclusterGroup>,
    schema_digest: [u8; 32],
    target_rows_per_fragment: usize,
}

impl ReclusterPlan {
    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn read_version(&self) -> u64 {
        self.read_version
    }

    pub fn clustering_version(&self) -> u64 {
        self.spec.version
    }

    pub fn algorithm_revision(&self) -> &str {
        &self.spec.algorithm_revision
    }

    pub fn columns(&self) -> &[String] {
        &self.spec.columns
    }

    pub fn groups(&self) -> &[ReclusterGroup] {
        &self.groups
    }

    pub fn model_context(&self) -> &[u8] {
        self.id.as_bytes()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        recluster_pb::Plan::from(self).encode_to_vec()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::try_from(recluster_pb::Plan::decode(bytes)?)
    }

    fn group(&self, group_id: Uuid) -> Result<&ReclusterGroup> {
        self.groups
            .iter()
            .find(|group| group.id == group_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "recluster group {group_id} does not belong to plan {}",
                    self.id
                ))
            })
    }

    fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        if self.id.is_nil() {
            return Err(Error::invalid_input("recluster plan id must not be nil"));
        }
        if self.read_version == 0 {
            return Err(Error::invalid_input(
                "recluster plan read version must be positive",
            ));
        }
        if self.schema_digest == [0; 32] {
            return Err(Error::invalid_input(
                "recluster plan schema digest must not be empty",
            ));
        }
        if self.target_rows_per_fragment == 0 {
            return Err(Error::invalid_input(
                "recluster target_rows_per_fragment must be positive",
            ));
        }
        if self.dataset_uri.is_empty() {
            return Err(Error::invalid_input(
                "recluster plan dataset URI must not be empty",
            ));
        }
        let mut group_ids = HashSet::with_capacity(self.groups.len());
        let mut fragment_ids = RoaringBitmap::new();
        for group in &self.groups {
            if group.id.is_nil() {
                return Err(Error::invalid_input("recluster group id must not be nil"));
            }
            if !group_ids.insert(group.id) {
                return Err(Error::invalid_input(format!(
                    "recluster plan contains duplicate group id {}",
                    group.id
                )));
            }
            if group.source_fragments.is_empty() {
                return Err(Error::invalid_input(format!(
                    "recluster group {} has no source fragments",
                    group.id
                )));
            }
            for fragment in &group.source_fragments {
                let fragment_id = u32::try_from(fragment.id).map_err(|_| {
                    Error::not_supported(format!(
                        "recluster source fragment id {} exceeds the physical row-address limit",
                        fragment.id
                    ))
                })?;
                if !fragment_ids.insert(fragment_id) {
                    return Err(Error::invalid_input(format!(
                        "recluster plan contains source fragment {} more than once",
                        fragment.id
                    )));
                }
            }
        }
        Ok(())
    }
}

/// A worker-produced set of staged fragments for one reclustering group.
#[derive(Debug, Clone, PartialEq)]
pub struct ReclusterResult {
    plan_id: Uuid,
    group_id: Uuid,
    dataset_uri: String,
    read_version: u64,
    clustering_version: u64,
    algorithm_revision: String,
    model_digest: [u8; 32],
    source_fragment_ids: Vec<u64>,
    new_fragments: Vec<Fragment>,
    input_rows: u64,
    output_rows: u64,
    output_row_digest: [u8; 32],
}

impl ReclusterResult {
    pub fn try_new(
        plan: &ReclusterPlan,
        group_id: Uuid,
        model: &ClusteringModel,
        new_fragments: Vec<Fragment>,
        output_row_digest: [u8; 32],
    ) -> Result<Self> {
        let group = plan.group(group_id)?;
        if model.columns() != plan.columns() {
            return Err(Error::invalid_input(
                "clustering model columns do not match the recluster plan",
            ));
        }
        if !model.matches_context(plan.model_context()) {
            return Err(Error::invalid_input(
                "clustering model does not belong to this recluster plan",
            ));
        }
        let output_rows = validate_staged_fragments(group, &new_fragments)?;
        let output_digest = RowDigest::from_bytes(output_row_digest);
        if output_digest.num_rows() != output_rows {
            return Err(Error::invalid_input(format!(
                "recluster group {group_id} output row digest counts {} rows, but its fragments \
                 contain {output_rows}",
                output_digest.num_rows()
            )));
        }
        Ok(Self {
            plan_id: plan.id,
            group_id,
            dataset_uri: plan.dataset_uri.clone(),
            read_version: plan.read_version,
            clustering_version: plan.spec.version,
            algorithm_revision: plan.algorithm_revision().to_string(),
            model_digest: model.digest(),
            source_fragment_ids: group.source_fragment_ids().collect(),
            new_fragments,
            input_rows: group.expected_live_rows,
            output_rows,
            output_row_digest,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        recluster_pb::Result::from(self).encode_to_vec()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::try_from(recluster_pb::Result::decode(bytes)?)
    }
}

/// Plan bounded groups of new, outdated, or undersized clustering layout groups.
pub async fn plan_recluster(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<ReclusterPlan> {
    let mut options = options.clone();
    validate_recluster_dataset(dataset, &mut options).await?;
    let spec = super::super::metadata::clustering_spec(dataset)?.ok_or_else(|| {
        Error::invalid_input("recluster requires clustering columns to be declared first")
    })?;
    spec.validate_current_algorithm()?;
    let excluded_fragment_ids: RoaringBitmap =
        options.excluded_fragment_ids.iter().copied().collect();
    let schema_field_ids = options.max_source_bytes.map(|_| {
        dataset
            .schema()
            .field_ids()
            .into_iter()
            .collect::<HashSet<_>>()
    });
    let fragments_with_rows = futures::stream::iter(dataset.get_fragments())
        .map(|fragment| async move {
            let rows = if let Some(rows) = fragment.metadata.num_rows() {
                rows
            } else {
                collect_metrics(&fragment).await?.num_rows()
            };
            Ok::<_, Error>((fragment, rows))
        })
        .buffered(dataset.object_store.as_ref().io_parallelism())
        .try_collect::<Vec<_>>()
        .await?;
    let live_rows_by_fragment = fragments_with_rows
        .iter()
        .map(|(fragment, rows)| (fragment.id() as u64, *rows))
        .collect::<HashMap<_, _>>();
    let mut units = Vec::<PlanningUnit>::new();
    let mut group_units = HashMap::<String, usize>::new();
    for (fragment, rows) in fragments_with_rows {
        let fragment_id = fragment.id() as u64;
        let group_id = dataset
            .manifest
            .fragment_clustering_group(fragment_id)
            .map(str::to_owned);
        let is_under_clustered = dataset.manifest.fragment_clustering_version(fragment_id)
            != Some(spec.version)
            || group_id.is_none();
        let is_excluded = u32::try_from(fragment.id())
            .is_ok_and(|fragment_id| excluded_fragment_ids.contains(fragment_id));
        let bytes = if let Some(schema_field_ids) = &schema_field_ids {
            task_source_bytes(
                &TaskData {
                    fragments: vec![fragment.metadata.clone()],
                },
                schema_field_ids,
            )?
        } else {
            0
        };
        if let Some(group_id) = group_id {
            if let Some(position) = group_units.get(&group_id).copied() {
                let unit = &mut units[position];
                unit.rows = unit.rows.checked_add(rows).ok_or_else(|| {
                    Error::invalid_input("clustering layout group row count overflowed usize")
                })?;
                unit.bytes = unit.bytes.checked_add(bytes).ok_or_else(|| {
                    Error::invalid_input("clustering layout group byte count overflowed u64")
                })?;
                unit.fragments.push(fragment.metadata);
                unit.is_under_clustered |= is_under_clustered;
                unit.is_excluded |= is_excluded;
            } else {
                group_units.insert(group_id.clone(), units.len());
                units.push(PlanningUnit {
                    fragments: vec![fragment.metadata],
                    rows,
                    bytes,
                    group_id: Some(group_id),
                    is_under_clustered,
                    is_excluded,
                });
            }
        } else {
            units.push(PlanningUnit {
                fragments: vec![fragment.metadata],
                rows,
                bytes,
                group_id: None,
                is_under_clustered,
                is_excluded,
            });
        }
    }

    let mut tasks_with_rows = Vec::new();
    let mut current = Vec::new();
    let mut current_rows = 0_usize;
    let mut current_bytes = 0_u64;
    let mut current_fragments = 0_usize;
    let mut current_has_under_clustered = false;
    let mut current_groups = HashSet::new();
    for unit in units {
        let is_partial_group = !unit.is_under_clustered
            && unit.group_id.is_some()
            && unit.rows < options.target_rows_per_fragment;
        if unit.is_excluded || (!unit.is_under_clustered && !is_partial_group) {
            push_group(
                &mut tasks_with_rows,
                &mut current,
                &mut current_rows,
                &mut current_bytes,
                &mut current_fragments,
                &mut current_has_under_clustered,
                &mut current_groups,
            );
            continue;
        }
        let would_exceed_group_budget = !current.is_empty()
            && (options
                .max_source_fragments
                .is_some_and(|max| current_fragments.saturating_add(unit.fragments.len()) > max)
                || options
                    .max_source_rows
                    .is_some_and(|max| current_rows.saturating_add(unit.rows) > max)
                || options
                    .max_source_bytes
                    .is_some_and(|max| current_bytes.saturating_add(unit.bytes) > max));
        if would_exceed_group_budget {
            push_group(
                &mut tasks_with_rows,
                &mut current,
                &mut current_rows,
                &mut current_bytes,
                &mut current_fragments,
                &mut current_has_under_clustered,
                &mut current_groups,
            );
        }
        current_rows = current_rows
            .checked_add(unit.rows)
            .ok_or_else(|| Error::invalid_input("recluster source row count overflowed usize"))?;
        current_bytes = current_bytes
            .checked_add(unit.bytes)
            .ok_or_else(|| Error::invalid_input("recluster source byte count overflowed u64"))?;
        current_has_under_clustered |= unit.is_under_clustered;
        if let Some(group_id) = unit.group_id {
            current_groups.insert(group_id);
        }
        current_fragments = current_fragments
            .checked_add(unit.fragments.len())
            .ok_or_else(|| Error::invalid_input("recluster source fragment count overflowed"))?;
        current.extend(unit.fragments);
        if current_rows >= options.target_rows_per_fragment {
            push_group(
                &mut tasks_with_rows,
                &mut current,
                &mut current_rows,
                &mut current_bytes,
                &mut current_fragments,
                &mut current_has_under_clustered,
                &mut current_groups,
            );
        }
    }
    push_group(
        &mut tasks_with_rows,
        &mut current,
        &mut current_rows,
        &mut current_bytes,
        &mut current_fragments,
        &mut current_has_under_clustered,
        &mut current_groups,
    );
    let tasks = limit_tasks_to_source_budget(&options, dataset.schema(), tasks_with_rows)?;
    let groups = tasks
        .into_iter()
        .map(|task| {
            let expected_live_rows = task.fragments.iter().try_fold(0_u64, |total, fragment| {
                let rows = live_rows_by_fragment.get(&fragment.id).ok_or_else(|| {
                    Error::internal(format!(
                        "live row count is missing for recluster source fragment {}",
                        fragment.id
                    ))
                })?;
                total.checked_add(*rows as u64).ok_or_else(|| {
                    Error::invalid_input("recluster source row count overflowed u64")
                })
            })?;
            Ok(ReclusterGroup {
                id: Uuid::new_v4(),
                source_fragments: task.fragments,
                expected_live_rows,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let plan = ReclusterPlan {
        id: Uuid::new_v4(),
        dataset_uri: dataset.uri().to_string(),
        read_version: dataset.manifest.version,
        spec,
        groups,
        schema_digest: schema_digest(dataset.schema()),
        target_rows_per_fragment: options.target_rows_per_fragment,
    };
    plan.validate()?;
    Ok(plan)
}

/// Build the single model shared by every rewrite group in a local reference run.
pub(super) async fn build_recluster_model(
    dataset: &Dataset,
    plan: &ReclusterPlan,
) -> Result<(ClusteringModel, HashMap<Uuid, RowDigest>)> {
    plan.validate()?;
    if plan.groups.is_empty() {
        return Err(Error::invalid_input(
            "cannot build a clustering model for an empty recluster plan",
        ));
    }
    let data_types = plan
        .spec
        .columns
        .iter()
        .map(|column| {
            dataset
                .schema()
                .field(column)
                .map(|field| field.data_type())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "clustering column {column:?} does not exist in the dataset schema"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut partial = PartialClusteringModel::try_new_with_context(
        &plan.spec,
        &data_types,
        plan.model_context().to_vec(),
    )?;
    let fragments = plan
        .groups
        .iter()
        .flat_map(|group| group.source_fragments.iter().cloned())
        .collect::<Vec<_>>();
    let fragment_groups = plan
        .groups
        .iter()
        .flat_map(|group| {
            group
                .source_fragments
                .iter()
                .map(move |fragment| (fragment.id, group.id))
        })
        .collect::<HashMap<_, _>>();
    let mut group_row_digests = plan
        .groups
        .iter()
        .map(|group| (group.id, RowDigest::default()))
        .collect::<HashMap<_, _>>();
    let mut scanner = dataset.scan();
    scanner
        .with_fragments(fragments)
        .scan_in_order(true)
        .project(&plan.spec.columns)?
        .with_row_address();
    let mut stream = scanner.try_into_stream().await?;
    while let Some(batch) = stream.try_next().await? {
        let row_addresses = batch
            .column_by_name(ROW_ADDR)
            .and_then(|array| array.as_any().downcast_ref::<arrow_array::UInt64Array>())
            .ok_or_else(|| {
                Error::internal(
                    "recluster model scan did not return a UInt64 _rowaddr column".to_string(),
                )
            })?;
        let columns = plan
            .spec
            .columns
            .iter()
            .map(|column| {
                batch.column_by_name(column).cloned().ok_or_else(|| {
                    Error::internal(format!(
                        "recluster model scan did not return clustering column {column:?}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        partial.add(&columns, row_addresses)?;
        for row_address in row_addresses.values() {
            let fragment_id =
                u64::from(lance_core::utils::address::RowAddress::from(*row_address).fragment_id());
            let group_id = fragment_groups.get(&fragment_id).ok_or_else(|| {
                Error::internal(format!(
                    "recluster model scan returned row address {row_address} outside the plan"
                ))
            })?;
            group_row_digests
                .get_mut(group_id)
                .ok_or_else(|| Error::internal("recluster group digest is missing".to_string()))?
                .add_row_address(*row_address)?;
        }
    }
    Ok((partial.finish(), group_row_digests))
}

fn push_group(
    groups: &mut Vec<(TaskData, usize)>,
    fragments: &mut Vec<Fragment>,
    rows: &mut usize,
    bytes: &mut u64,
    fragment_count: &mut usize,
    has_under_clustered: &mut bool,
    group_ids: &mut HashSet<String>,
) {
    if !fragments.is_empty() {
        let fragments = std::mem::take(fragments);
        if *has_under_clustered || group_ids.len() > 1 {
            groups.push((TaskData { fragments }, *rows));
        }
        *rows = 0;
        *bytes = 0;
        *fragment_count = 0;
        *has_under_clustered = false;
        group_ids.clear();
    }
}

pub(super) async fn validate_recluster_dataset(
    dataset: &Dataset,
    options: &mut CompactionOptions,
) -> Result<()> {
    if options.defer_index_remap {
        return Err(Error::invalid_input(
            "recluster does not support deferred index remapping",
        ));
    }
    if options.compaction_mode() != CompactionMode::Reencode {
        return Err(Error::invalid_input(
            "recluster requires compaction_mode=reencode",
        ));
    }
    if dataset.manifest.uses_stable_row_ids() {
        return Err(Error::not_supported(
            "recluster does not yet support stable row ids",
        ));
    }
    if options.num_threads == Some(0) {
        return Err(Error::invalid_input(
            "recluster requires num_threads to be greater than zero",
        ));
    }
    if dataset.manifest.index_section.is_some()
        && dataset.load_indices().await?.iter().any(|index| {
            !index
                .index_details
                .as_ref()
                .is_some_and(|details| details.type_url.ends_with("ZoneMapIndexDetails"))
        })
    {
        return Err(Error::not_supported(
            "recluster does not yet support indices other than zonemaps; drop the index, \
             recluster, and rebuild it",
        ));
    }
    options.compaction_mode = Some(CompactionMode::Reencode);
    options.validate()
}

/// Validate and atomically commit worker-produced clustering fragments.
pub async fn commit_recluster(
    dataset: &mut Dataset,
    plan: &ReclusterPlan,
    model: &ClusteringModel,
    results: Vec<ReclusterResult>,
) -> Result<CompactionMetrics> {
    plan.validate()?;
    if plan.dataset_uri != dataset.uri() {
        return Err(Error::invalid_input(format!(
            "recluster plan targets dataset {:?}, got {:?}",
            plan.dataset_uri,
            dataset.uri()
        )));
    }
    let snapshot = if dataset.manifest.version == plan.read_version {
        Cow::Borrowed(dataset as &Dataset)
    } else {
        Cow::Owned(dataset.checkout_version(plan.read_version).await?)
    };
    let snapshot_spec = super::super::metadata::clustering_spec(snapshot.as_ref())?;
    if snapshot_spec.as_ref() != Some(&plan.spec) {
        return Err(Error::invalid_input(
            "recluster plan does not match the declaration at its read version",
        ));
    }
    let current_spec = super::super::metadata::clustering_spec(dataset)?;
    if current_spec.as_ref() != Some(&plan.spec) {
        return Err(Error::invalid_input(
            "clustering declaration changed after this recluster plan was created",
        ));
    }
    if schema_digest(dataset.schema()) != plan.schema_digest {
        return Err(Error::invalid_input(
            "dataset schema changed after this recluster plan was created",
        ));
    }
    if model.columns() != plan.columns() {
        return Err(Error::invalid_input(
            "clustering model columns do not match the recluster plan",
        ));
    }
    if !model.matches_context(plan.model_context()) {
        return Err(Error::invalid_input(
            "clustering model does not belong to this recluster plan",
        ));
    }
    validate_plan_snapshot(plan, snapshot.as_ref()).await?;
    let planned_rows = plan.groups.iter().try_fold(0_u64, |total, group| {
        total
            .checked_add(group.expected_live_rows)
            .ok_or_else(|| Error::invalid_input("recluster plan row count overflowed u64"))
    })?;
    if model.input_rows() != planned_rows {
        return Err(Error::invalid_input(format!(
            "clustering model contains {} input rows, but the recluster plan contains \
             {planned_rows}",
            model.input_rows()
        )));
    }
    let mut completed_group_ids = HashSet::with_capacity(results.len());
    let mut rewrites = Vec::with_capacity(results.len());
    let mut output_digest = lance_index::clustering::RowDigest::default();
    let source_paths = plan
        .groups
        .iter()
        .flat_map(|group| &group.source_fragments)
        .flat_map(|fragment| {
            fragment
                .files
                .iter()
                .map(|file| (file.base_id, file.path.clone()))
        })
        .collect::<HashSet<_>>();
    let mut output_paths = HashSet::new();
    for result in results {
        if !completed_group_ids.insert(result.group_id) {
            return Err(Error::invalid_input(format!(
                "recluster group {} was completed more than once",
                result.group_id
            )));
        }
        validate_result(plan, model, &result)?;
        for file in result
            .new_fragments
            .iter()
            .flat_map(|fragment| &fragment.files)
        {
            let path = (file.base_id, file.path.clone());
            if source_paths.contains(&path) || !output_paths.insert(path) {
                return Err(Error::invalid_input(format!(
                    "recluster output file {:?} is not a unique staged file",
                    file.path
                )));
            }
        }
        output_digest.merge(lance_index::clustering::RowDigest::from_bytes(
            result.output_row_digest,
        ))?;
        let group = plan.group(result.group_id)?;
        rewrites.push(RewriteResult {
            metrics: CompactionMetrics {
                fragments_removed: group.source_fragments.len(),
                fragments_added: result.new_fragments.len(),
                files_removed: group
                    .source_fragments
                    .iter()
                    .map(|fragment| {
                        fragment.files.len() + usize::from(fragment.deletion_file.is_some())
                    })
                    .sum(),
                files_added: result
                    .new_fragments
                    .iter()
                    .map(|fragment| fragment.files.len())
                    .sum(),
            },
            new_fragments: result.new_fragments,
            read_version: plan.read_version,
            original_fragments: group.source_fragments.clone(),
            row_addrs: None,
        });
    }
    if completed_group_ids.len() != plan.groups.len() {
        return Err(Error::invalid_input(format!(
            "recluster commit requires all {} plan groups, got {} results",
            plan.groups.len(),
            completed_group_ids.len()
        )));
    }
    if output_digest != model.row_digest() {
        return Err(Error::invalid_input(
            "recluster output row identities do not match the planned input rows",
        ));
    }

    let mut options = CompactionOptions {
        compaction_mode: Some(CompactionMode::Reencode),
        ..Default::default()
    };
    validate_recluster_dataset(dataset, &mut options).await?;
    let metrics = commit_compaction_internal(
        dataset,
        rewrites,
        Arc::new(IgnoreRemap {}),
        &options,
        Some(plan.spec.version),
    )
    .await?;
    if let Err(error) = maintain_clustering_zonemaps(dataset).await {
        tracing::warn!("recluster committed, but refreshing zonemap coverage failed: {error}");
    }
    Ok(metrics)
}

fn validate_result(
    plan: &ReclusterPlan,
    model: &ClusteringModel,
    result: &ReclusterResult,
) -> Result<()> {
    let group = plan.group(result.group_id)?;
    if result.plan_id != plan.id
        || result.dataset_uri != plan.dataset_uri
        || result.read_version != plan.read_version
        || result.clustering_version != plan.spec.version
        || result.algorithm_revision != plan.algorithm_revision()
        || result.model_digest != model.digest()
    {
        return Err(Error::invalid_input(format!(
            "recluster result for group {} does not match plan {}",
            result.group_id, plan.id
        )));
    }
    if result.source_fragment_ids != group.source_fragment_ids().collect::<Vec<_>>() {
        return Err(Error::invalid_input(format!(
            "recluster result for group {} has different source fragments",
            result.group_id
        )));
    }
    let output_rows = validate_staged_fragments(group, &result.new_fragments)?;
    if RowDigest::from_bytes(result.output_row_digest).num_rows() != output_rows {
        return Err(Error::invalid_input(format!(
            "recluster result for group {} has an output row digest with the wrong row count",
            result.group_id
        )));
    }
    if result.input_rows != group.expected_live_rows
        || result.output_rows != output_rows
        || result.input_rows != result.output_rows
    {
        return Err(Error::invalid_input(format!(
            "recluster result for group {} has inconsistent row counts",
            result.group_id
        )));
    }
    Ok(())
}

fn validate_staged_fragments(group: &ReclusterGroup, fragments: &[Fragment]) -> Result<u64> {
    let source_paths = group
        .source_fragments
        .iter()
        .flat_map(|fragment| {
            fragment
                .files
                .iter()
                .map(|file| (file.base_id, file.path.as_str()))
        })
        .collect::<HashSet<_>>();
    let mut output_paths = HashSet::new();
    if fragments.iter().any(|fragment| {
        fragment.id != 0
            || fragment.files.is_empty()
            || fragment.physical_rows.is_none()
            || fragment.deletion_file.is_some()
            || fragment.row_id_meta.is_some()
            || !fragment.overlays.is_empty()
            || fragment.created_at_version_meta.is_some()
            || fragment.last_updated_at_version_meta.is_some()
            || fragment.files.iter().any(|file| {
                let path = (file.base_id, file.path.as_str());
                source_paths.contains(&path) || !output_paths.insert(path)
            })
    }) {
        return Err(Error::invalid_input(format!(
            "recluster result for group {} contains invalid staged fragment metadata",
            group.id
        )));
    }
    let output_rows = sum_fragment_rows(fragments, "recluster output")?;
    if output_rows != group.expected_live_rows {
        return Err(Error::invalid_input(format!(
            "recluster group {} changed the live row count from {} to {output_rows}",
            group.id, group.expected_live_rows
        )));
    }
    Ok(output_rows)
}

async fn validate_plan_snapshot(plan: &ReclusterPlan, snapshot: &Dataset) -> Result<()> {
    let snapshot_fragments = snapshot.get_fragments();
    let planned_fragment_ids = plan
        .groups
        .iter()
        .flat_map(ReclusterGroup::source_fragment_ids)
        .map(|fragment_id| fragment_id as u32)
        .collect::<RoaringBitmap>();
    let planned_layout_groups = plan
        .groups
        .iter()
        .flat_map(|group| &group.source_fragments)
        .filter_map(|fragment| snapshot.manifest.fragment_clustering_group(fragment.id))
        .collect::<HashSet<_>>();
    let mut current_group_rows = HashMap::<&str, u64>::new();
    for fragment in &snapshot_fragments {
        let Some(group_id) = snapshot
            .manifest
            .fragment_clustering_group(fragment.id() as u64)
            .filter(|group_id| planned_layout_groups.contains(group_id))
        else {
            continue;
        };
        if !u32::try_from(fragment.id())
            .is_ok_and(|fragment_id| planned_fragment_ids.contains(fragment_id))
        {
            return Err(Error::invalid_input(format!(
                "recluster plan contains only part of clustering layout group {group_id:?}"
            )));
        }
        if snapshot
            .manifest
            .fragment_clustering_version(fragment.id() as u64)
            == Some(plan.spec.version)
        {
            let rows = collect_metrics(fragment).await?.num_rows() as u64;
            let total = current_group_rows.entry(group_id).or_default();
            *total = total.checked_add(rows).ok_or_else(|| {
                Error::invalid_input("clustering layout group row count overflowed u64")
            })?;
        }
    }
    if let Some((group_id, rows)) = current_group_rows
        .iter()
        .find(|(_, rows)| **rows >= plan.target_rows_per_fragment as u64)
    {
        return Err(Error::invalid_input(format!(
            "recluster plan contains stable clustering layout group {group_id:?} with {rows} \
             rows"
        )));
    }

    let fragments = snapshot_fragments
        .into_iter()
        .map(|fragment| (fragment.id() as u64, fragment))
        .collect::<std::collections::HashMap<_, _>>();
    for group in &plan.groups {
        let mut actual_rows = 0_u64;
        for planned in &group.source_fragments {
            let actual = fragments.get(&planned.id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "recluster plan source fragment {} does not exist at read version {}",
                    planned.id, plan.read_version
                ))
            })?;
            if actual.metadata != *planned {
                return Err(Error::invalid_input(format!(
                    "recluster plan source fragment {} does not match read version {}",
                    planned.id, plan.read_version
                )));
            }
            actual_rows = actual_rows
                .checked_add(collect_metrics(actual).await?.num_rows() as u64)
                .ok_or_else(|| Error::invalid_input("recluster source row count overflowed u64"))?;
        }
        if actual_rows != group.expected_live_rows {
            return Err(Error::invalid_input(format!(
                "recluster group {} expected_live_rows is {}, but its source fragments \
                 contain {actual_rows} live rows",
                group.id, group.expected_live_rows
            )));
        }
    }
    Ok(())
}

fn sum_fragment_rows(fragments: &[Fragment], context: &str) -> Result<u64> {
    fragments.iter().try_fold(0_u64, |total, fragment| {
        let rows = fragment.num_rows().ok_or_else(|| {
            Error::invalid_input(format!(
                "{context} fragment {} is missing row counts",
                fragment.id
            ))
        })?;
        total
            .checked_add(rows as u64)
            .ok_or_else(|| Error::invalid_input(format!("{context} row count overflowed u64")))
    })
}

impl From<&ReclusterPlan> for recluster_pb::Plan {
    fn from(plan: &ReclusterPlan) -> Self {
        Self {
            format_version: PROTOCOL_VERSION,
            plan_id: plan.id.to_string(),
            dataset_uri: plan.dataset_uri.clone(),
            read_version: plan.read_version,
            clustering_version: plan.spec.version,
            algorithm_revision: plan.spec.algorithm_revision.clone(),
            columns: plan.spec.columns.clone(),
            groups: plan
                .groups
                .iter()
                .map(|group| recluster_pb::Group {
                    group_id: group.id.to_string(),
                    source_fragments: group
                        .source_fragments
                        .iter()
                        .map(lance_table::format::pb::DataFragment::from)
                        .collect(),
                    expected_live_rows: group.expected_live_rows,
                })
                .collect(),
            schema_digest: plan.schema_digest.to_vec(),
            target_rows_per_fragment: plan.target_rows_per_fragment as u64,
        }
    }
}

impl TryFrom<recluster_pb::Plan> for ReclusterPlan {
    type Error = Error;

    fn try_from(plan: recluster_pb::Plan) -> Result<Self> {
        validate_protocol_version(plan.format_version, "recluster plan")?;
        let result = Self {
            id: parse_uuid(&plan.plan_id, "recluster plan id")?,
            dataset_uri: plan.dataset_uri,
            read_version: plan.read_version,
            spec: ClusteringSpec {
                columns: plan.columns,
                algorithm_revision: plan.algorithm_revision,
                version: plan.clustering_version,
            },
            groups: plan
                .groups
                .into_iter()
                .map(|group| {
                    Ok(ReclusterGroup {
                        id: parse_uuid(&group.group_id, "recluster group id")?,
                        source_fragments: group
                            .source_fragments
                            .into_iter()
                            .map(Fragment::try_from)
                            .collect::<Result<Vec<_>>>()?,
                        expected_live_rows: group.expected_live_rows,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            schema_digest: plan.schema_digest.try_into().map_err(|value: Vec<u8>| {
                Error::invalid_input(format!(
                    "recluster schema digest must contain 32 bytes, got {}",
                    value.len()
                ))
            })?,
            target_rows_per_fragment: usize::try_from(plan.target_rows_per_fragment).map_err(
                |_| Error::invalid_input("recluster target_rows_per_fragment exceeds usize"),
            )?,
        };
        result.validate()?;
        result.spec.validate_current_algorithm()?;
        Ok(result)
    }
}

impl From<&ReclusterResult> for recluster_pb::Result {
    fn from(result: &ReclusterResult) -> Self {
        Self {
            format_version: PROTOCOL_VERSION,
            plan_id: result.plan_id.to_string(),
            group_id: result.group_id.to_string(),
            dataset_uri: result.dataset_uri.clone(),
            read_version: result.read_version,
            clustering_version: result.clustering_version,
            algorithm_revision: result.algorithm_revision.clone(),
            model_digest: result.model_digest.to_vec(),
            source_fragment_ids: result.source_fragment_ids.clone(),
            new_fragments: result
                .new_fragments
                .iter()
                .map(lance_table::format::pb::DataFragment::from)
                .collect(),
            input_rows: result.input_rows,
            output_rows: result.output_rows,
            output_row_digest: result.output_row_digest.to_vec(),
        }
    }
}

impl TryFrom<recluster_pb::Result> for ReclusterResult {
    type Error = Error;

    fn try_from(result: recluster_pb::Result) -> Result<Self> {
        validate_protocol_version(result.format_version, "recluster result")?;
        if result.algorithm_revision.is_empty() {
            return Err(Error::invalid_input(
                "recluster result algorithm revision must not be empty",
            ));
        }
        let model_digest: [u8; 32] = result.model_digest.try_into().map_err(|value: Vec<u8>| {
            Error::invalid_input(format!(
                "recluster model digest must contain 32 bytes, got {}",
                value.len()
            ))
        })?;
        let output_row_digest: [u8; 32] =
            result
                .output_row_digest
                .try_into()
                .map_err(|value: Vec<u8>| {
                    Error::invalid_input(format!(
                        "recluster output row digest must contain 32 bytes, got {}",
                        value.len()
                    ))
                })?;
        Ok(Self {
            plan_id: parse_uuid(&result.plan_id, "recluster result plan id")?,
            group_id: parse_uuid(&result.group_id, "recluster result group id")?,
            dataset_uri: result.dataset_uri,
            read_version: result.read_version,
            clustering_version: result.clustering_version,
            algorithm_revision: result.algorithm_revision,
            model_digest,
            source_fragment_ids: result.source_fragment_ids,
            new_fragments: result
                .new_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
            input_rows: result.input_rows,
            output_rows: result.output_rows,
            output_row_digest,
        })
    }
}

fn validate_protocol_version(version: u32, context: &str) -> Result<()> {
    if version != PROTOCOL_VERSION {
        return Err(Error::invalid_input(format!(
            "unsupported {context} format version {version}"
        )));
    }
    Ok(())
}

fn parse_uuid(value: &str, name: &str) -> Result<Uuid> {
    Uuid::parse_str(value)
        .map_err(|error| Error::invalid_input(format!("invalid {name} {value:?}: {error}")))
}

fn schema_digest(schema: &lance_core::datatypes::Schema) -> [u8; 32] {
    let arrow_schema: arrow_schema::Schema = schema.into();
    let mut dictionary_tracker = arrow_ipc::writer::DictionaryTracker::new(false);
    let mut encoder = arrow_ipc::convert::IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut dictionary_tracker);
    let encoded = encoder.schema_to_fb(&arrow_schema);
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"lance-recluster-schema-v1");
    hasher.update(&(encoded.finished_data().len() as u64).to_le_bytes());
    hasher.update(encoded.finished_data());
    for field in schema.fields_pre_order() {
        hasher.update(&field.id.to_le_bytes());
        hasher.update(&field.parent_id.to_le_bytes());
        hasher.update(&[match field.encoding.as_ref() {
            None => 0,
            Some(lance_core::datatypes::Encoding::Plain) => 1,
            Some(lance_core::datatypes::Encoding::VarBinary) => 2,
            Some(lance_core::datatypes::Encoding::Dictionary) => 3,
            Some(lance_core::datatypes::Encoding::RLE) => 4,
        }]);
        hasher.update(
            &field
                .unenforced_primary_key_position
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        hasher.update(
            &field
                .unenforced_clustering_key_position
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int32Array, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::utils::address::RowAddress;
    use lance_index::clustering::PartialClusteringModel;

    use super::*;
    use crate::dataset::fragment::write::FragmentCreateBuilder;
    use crate::dataset::{WriteMode, WriteParams};
    use lance_core::utils::tempfile::TempStrDir;

    #[tokio::test]
    async fn plan_round_trip_and_result_validation() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap();
        let uri = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)].into_iter(), schema),
            &uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.set_clustering(vec!["key".into()]).await.unwrap();
        dataset.delete("key = 0").await.unwrap();

        let plan = plan_recluster(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        let decoded = ReclusterPlan::from_bytes(&plan.to_bytes()).unwrap();
        assert_eq!(plan, decoded);
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].source_fragments.len(), 2);
        assert_eq!(plan.groups[0].expected_live_rows, 19);

        let mut first = dataset.schema().clone();
        first.metadata.insert("a".into(), "1".into());
        first.metadata.insert("b".into(), "2".into());
        let mut second = dataset.schema().clone();
        second.metadata.insert("b".into(), "2".into());
        second.metadata.insert("a".into(), "1".into());
        assert_eq!(schema_digest(&first), schema_digest(&second));
        second.metadata.insert("a".into(), "changed".into());
        assert_ne!(schema_digest(&first), schema_digest(&second));
        second.metadata.insert("a".into(), "1".into());
        second.fields[0].id += 1;
        assert_ne!(schema_digest(&first), schema_digest(&second));
    }

    #[tokio::test]
    async fn planner_keeps_persisted_layout_groups_atomic() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..30))],
        )
        .unwrap();
        let uri = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)].into_iter(), schema),
            &uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.set_clustering(vec!["key".into()]).await.unwrap();
        let version = super::super::super::metadata::clustering_spec(&dataset)
            .unwrap()
            .unwrap()
            .version;
        Arc::make_mut(&mut dataset.manifest)
            .set_fragment_clustering_metadata(vec![
                (Some(version + 1), Some("outdated-group".to_string())),
                (Some(version + 1), Some("outdated-group".to_string())),
                (None, None),
            ])
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 100,
            ..Default::default()
        };
        let plan = plan_recluster(&dataset, &options).await.unwrap();
        assert_eq!(plan.groups().len(), 1);
        assert_eq!(
            plan.groups()[0].source_fragment_ids().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let mut partial_plan = plan.clone();
        partial_plan.groups[0].source_fragments.remove(0);
        let error = validate_plan_snapshot(&partial_plan, &dataset)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("contains only part of clustering layout group")
        );

        let excluded = CompactionOptions {
            excluded_fragment_ids: vec![0],
            ..options
        };
        let plan = plan_recluster(&dataset, &excluded).await.unwrap();
        assert_eq!(plan.groups().len(), 1);
        assert_eq!(
            plan.groups()[0].source_fragment_ids().collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[tokio::test]
    async fn distributed_result_commits_and_rejects_foreign_model() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap();
        let uri = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())].into_iter(), schema),
            &uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.set_clustering(vec!["key".into()]).await.unwrap();

        let plan = plan_recluster(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        let other_plan = plan_recluster(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        let row_addresses = arrow_array::UInt64Array::from_iter_values(
            (0..10)
                .map(|offset| u64::from(RowAddress::new_from_parts(0, offset)))
                .chain((0..10).map(|offset| u64::from(RowAddress::new_from_parts(1, offset)))),
        );
        let partial = PartialClusteringModel::from_batch_bytes(
            plan.model_context().to_vec(),
            plan.columns().to_vec(),
            &batch,
            &row_addresses,
        )
        .unwrap();
        let model = PartialClusteringModel::merge_bytes_to_model([partial]).unwrap();
        let group = &plan.groups()[0];
        assert!(
            ReclusterResult::try_new(
                &plan,
                group.id(),
                &model,
                group.source_fragments().to_vec(),
                model.row_digest().to_bytes(),
            )
            .is_err()
        );
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let output = FragmentCreateBuilder::new(dataset.uri())
            .schema(dataset.schema())
            .write_params(&write_params)
            .write(
                RecordBatchIterator::new([Ok(batch.clone())].into_iter(), batch.schema()),
                None,
            )
            .await
            .unwrap();
        let short_digest = RowDigest::from_row_addresses(&arrow_array::UInt64Array::from(vec![0]));
        assert!(
            ReclusterResult::try_new(
                &plan,
                group.id(),
                &model,
                vec![output.clone()],
                short_digest.unwrap().to_bytes(),
            )
            .is_err()
        );
        let result = ReclusterResult::try_new(
            &plan,
            group.id(),
            &model,
            vec![output],
            model.row_digest().to_bytes(),
        )
        .unwrap();
        let mut corrupt = result.clone();
        corrupt.output_row_digest[8] ^= 1;
        let mut unchanged = dataset.clone();
        let error = commit_recluster(&mut unchanged, &plan, &model, vec![corrupt])
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("output row identities do not match")
        );

        assert!(
            ReclusterResult::try_new(
                &other_plan,
                other_plan.groups()[0].id(),
                &model,
                other_plan.groups()[0].source_fragments().to_vec(),
                model.row_digest().to_bytes(),
            )
            .is_err()
        );

        commit_recluster(&mut dataset, &plan, &model, vec![result])
            .await
            .unwrap();
        assert!(
            dataset
                .manifest
                .fragment_clustering_versions()
                .iter()
                .all(|version| *version == Some(plan.clustering_version()))
        );
        let committed_groups = dataset
            .manifest
            .fragment_clustering_groups()
            .into_iter()
            .collect::<Vec<_>>();
        assert!(committed_groups.iter().all(Option::is_some));
        assert!(committed_groups.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(
            plan_recluster(&dataset, &CompactionOptions::default())
                .await
                .unwrap()
                .groups()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn commit_rejects_plan_after_declaration_changes() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..4))],
        )
        .unwrap();
        let uri = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())].into_iter(), schema),
            &uri,
            None,
        )
        .await
        .unwrap();
        dataset.set_clustering(vec!["key".into()]).await.unwrap();
        let plan = plan_recluster(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        let row_addresses = arrow_array::UInt64Array::from_iter_values(
            (0..4).map(|offset| u64::from(RowAddress::new_from_parts(0, offset))),
        );
        let partial = PartialClusteringModel::from_batch_bytes(
            plan.model_context().to_vec(),
            plan.columns().to_vec(),
            &batch,
            &row_addresses,
        )
        .unwrap();
        let model = PartialClusteringModel::merge_bytes_to_model([partial]).unwrap();
        let output = FragmentCreateBuilder::new(dataset.uri())
            .schema(dataset.schema())
            .write(
                RecordBatchIterator::new([Ok(batch.clone())].into_iter(), batch.schema()),
                None,
            )
            .await
            .unwrap();
        let result = ReclusterResult::try_new(
            &plan,
            plan.groups()[0].id(),
            &model,
            vec![output],
            model.row_digest().to_bytes(),
        )
        .unwrap();
        dataset.clear_clustering().await.unwrap();

        let error = commit_recluster(&mut dataset, &plan, &model, vec![result])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("declaration changed"));
    }
}
