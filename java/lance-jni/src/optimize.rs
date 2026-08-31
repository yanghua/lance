// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use jni::{
    JNIEnv,
    objects::{JByteArray, JObject, JValueGen},
    sys::jlong,
};
use lance::dataset::{
    index::DatasetIndexRemapperOptions,
    optimize::{
        ClusteringCompactionTask, ClusteringRewriteResult, CompactionMetrics, CompactionMode,
        CompactionOptions, CompactionRequest, CompactionTask, IndexRemapperOptions, RewriteResult,
        TaskData, commit_clustering_compaction, commit_compaction, plan_clustering_compaction,
        plan_compaction,
    },
};

use crate::{
    block_on,
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    fragment::{ImportedFragment, export_fragments_with_clustering_versions},
    traits::{FromJObjectWithEnv, IntoJava, import_vec_from_method, import_vec_to_rust},
    utils::{
        build_compaction_request, to_java_boolean_obj, to_java_float_obj, to_java_list,
        to_java_long_obj, to_java_optional,
    },
};

use crate::error::Result;
use crate::transaction::split_imported_fragments;

#[derive(Debug)]
struct ImportedTaskData {
    task: TaskData,
}

#[derive(Debug)]
struct ImportedRewriteResult {
    result: RewriteResult,
    new_fragment_clustering_version: Option<u64>,
    clustering_result_payload: Option<Vec<u8>>,
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_compaction_Compaction_nativePlanCompaction<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    java_dataset: JObject,                    // Dataset
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_plan_compaction(
            &mut env,
            java_dataset,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap,
            compaction_mode,
            binary_copy_read_batch_bytes,
            max_source_fragments,
            max_source_rows,
            max_source_bytes,
            excluded_fragment_ids
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_plan_compaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> Result<JObject<'local>> {
    let config = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset.inner.manifest.config.clone()
    };
    let compaction_request = build_compaction_request(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
        &compaction_mode,
        &binary_copy_read_batch_bytes,
        &max_source_fragments,
        &max_source_rows,
        &max_source_bytes,
        &excluded_fragment_ids,
        &config,
    )?;

    let (plan_data, fragment_clustering_versions) = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        let versions = dataset
            .inner
            .manifest()
            .fragments
            .iter()
            .map(|fragment| {
                (
                    fragment.id,
                    dataset
                        .inner
                        .manifest()
                        .fragment_clustering_version(fragment.id),
                )
            })
            .collect();
        let plan_data = match compaction_request {
            CompactionRequest::Compaction(options) => {
                let plan = block_on(plan_compaction(&dataset.inner, &options))?;
                (plan.tasks, plan.read_version, plan.options, None)
            }
            CompactionRequest::Clustering(options) => {
                let plan = block_on(plan_clustering_compaction(&dataset.inner, &options))?;
                let payloads = plan
                    .compaction_tasks()
                    .map(|task| serde_json::to_vec(&task).map_err(Into::into))
                    .collect::<Result<Vec<_>>>()?;
                (
                    plan.tasks().to_vec(),
                    plan.read_version(),
                    plan.options().clone(),
                    Some(payloads),
                )
            }
        };
        (plan_data, versions)
    };
    let (tasks, read_version, options, payloads) = plan_data;
    compaction_plan_into_java(
        env,
        &tasks,
        read_version,
        &options,
        &fragment_clustering_versions,
        payloads.as_deref(),
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_compaction_Compaction_commitCompactionNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    java_dataset: JObject,                    // Dataset
    rewrite_results: JObject,                 // List<RewriteResult>
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_commit_compaction(
            &mut env,
            java_dataset,
            rewrite_results,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap,
            compaction_mode,
            binary_copy_read_batch_bytes,
            max_source_fragments,
            max_source_rows,
            max_source_bytes,
            excluded_fragment_ids,
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_commit_compaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    rewrite_results: JObject,                 // List<RewriteResult>
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> Result<JObject<'local>> {
    let config = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset.inner.manifest.config.clone()
    };
    let compaction_request = build_compaction_request(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
        &compaction_mode,
        &binary_copy_read_batch_bytes,
        &max_source_fragments,
        &max_source_rows,
        &max_source_bytes,
        &excluded_fragment_ids,
        &config,
    )?;
    let completed_tasks: Vec<ImportedRewriteResult> =
        import_vec_to_rust(env, &rewrite_results, |env, rewrite_result| {
            rewrite_result.extract_object(env)
        })?;
    let committed_metrics = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        match compaction_request {
            CompactionRequest::Compaction(options) => {
                let completed_tasks = completed_tasks
                    .into_iter()
                    .map(|result| {
                        if result.new_fragment_clustering_version.is_some()
                            || result.clustering_result_payload.is_some()
                        {
                            return Err(crate::error::Error::input_error(
                                "ordinary compaction cannot commit clustering rewrite results"
                                    .to_string(),
                            ));
                        }
                        Ok(result.result)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let remap_options: Arc<dyn IndexRemapperOptions> =
                    Arc::new(DatasetIndexRemapperOptions {});
                block_on(commit_compaction(
                    &mut dataset.inner,
                    completed_tasks,
                    remap_options,
                    &options,
                ))?
            }
            CompactionRequest::Clustering(options) => {
                let completed_tasks = completed_tasks
                    .into_iter()
                    .map(|result| {
                        let payload = result.clustering_result_payload.ok_or_else(|| {
                            crate::error::Error::input_error(
                                "clustering compaction requires a tagged clustering result payload"
                                    .to_string(),
                            )
                        })?;
                        let tagged: ClusteringRewriteResult = serde_json::from_slice(&payload)
                            .map_err(|error| {
                                crate::error::Error::input_error(format!(
                                    "invalid tagged clustering result payload: {error}"
                                ))
                            })?;
                        if tagged.rewrite_result() != &result.result {
                            return Err(crate::error::Error::input_error(
                                "clustering result payload does not match Java RewriteResult"
                                    .to_string(),
                            ));
                        }
                        if result.new_fragment_clustering_version
                            != Some(tagged.clustering_version())
                        {
                            return Err(crate::error::Error::input_error(
                                "clustering result payload version does not match fragment metadata"
                                    .to_string(),
                            ));
                        }
                        Ok(tagged)
                    })
                    .collect::<Result<Vec<_>>>()?;
                block_on(commit_clustering_compaction(
                    &mut dataset.inner,
                    completed_tasks,
                    &options,
                ))?
            }
        }
    };
    committed_metrics.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_compaction_CompactionTask_nativeExecute<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,                            // CompactionTask itself
    java_dataset: JObject,                    // Dataset
    task_data: JObject,                       // TaskData
    clustering_task_payload: JObject,         // byte[] or null
    read_version: jlong,                      // readVersion
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_execute_task(
            &mut env,
            java_dataset,
            task_data,
            clustering_task_payload,
            read_version,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap,
            compaction_mode,
            binary_copy_read_batch_bytes,
            max_source_fragments,
            max_source_rows,
            max_source_bytes,
            excluded_fragment_ids
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_execute_task<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    task_data: JObject,                       // TaskData
    clustering_task_payload: JObject,         // byte[] or null
    read_version: jlong,                      // readVersion
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> Result<JObject<'local>> {
    let imported_task: ImportedTaskData = task_data.extract_object(env)?;
    let config = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset.inner.manifest.config.clone()
    };
    let compaction_request = build_compaction_request(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
        &compaction_mode,
        &binary_copy_read_batch_bytes,
        &max_source_fragments,
        &max_source_rows,
        &max_source_bytes,
        &excluded_fragment_ids,
        &config,
    )?;
    let read_version = u64::try_from(read_version).map_err(|_| {
        crate::error::Error::input_error(format!(
            "readVersion must be non-negative, got {read_version}"
        ))
    })?;
    let clustering_task_payload = if clustering_task_payload.is_null() {
        None
    } else {
        Some(env.convert_byte_array(JByteArray::from(clustering_task_payload))?)
    };
    let (rewrite_result, original_fragment_clustering_versions, clustering_result_payload) = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let execution_dataset = if dataset.inner.version().version == read_version {
            dataset.inner.clone()
        } else {
            block_on(dataset.inner.checkout_version(read_version))?
        };
        let original_fragment_clustering_versions = execution_dataset
            .manifest()
            .fragments
            .iter()
            .map(|fragment| {
                (
                    fragment.id,
                    execution_dataset
                        .manifest()
                        .fragment_clustering_version(fragment.id),
                )
            })
            .collect::<HashMap<_, _>>();
        match compaction_request {
            CompactionRequest::Compaction(options) => {
                if clustering_task_payload.is_some() {
                    return Err(crate::error::Error::input_error(
                        "ordinary compaction task cannot carry a clustering task payload"
                            .to_string(),
                    ));
                }
                let task = CompactionTask {
                    task: imported_task.task,
                    read_version,
                    options,
                };
                (
                    block_on(task.execute(&execution_dataset))?,
                    original_fragment_clustering_versions,
                    None,
                )
            }
            CompactionRequest::Clustering(options) => {
                let payload = clustering_task_payload.as_deref().ok_or_else(|| {
                    crate::error::Error::input_error(
                        "clustering compaction task requires a tagged clustering task payload"
                            .to_string(),
                    )
                })?;
                let tagged: ClusteringCompactionTask =
                    serde_json::from_slice(payload).map_err(|error| {
                        crate::error::Error::input_error(format!(
                            "invalid tagged clustering task payload: {error}"
                        ))
                    })?;
                if tagged.task_data() != &imported_task.task
                    || tagged.read_version() != read_version
                    || tagged.options() != &options
                {
                    return Err(crate::error::Error::input_error(
                        "clustering task payload does not match Java CompactionTask".to_string(),
                    ));
                }
                let result = block_on(tagged.execute(&execution_dataset))?;
                let payload = serde_json::to_vec(&result)?;
                (
                    result.rewrite_result().clone(),
                    original_fragment_clustering_versions,
                    Some((result.clustering_version(), payload)),
                )
            }
        }
    };
    rewrite_result_into_java(
        env,
        &rewrite_result,
        &original_fragment_clustering_versions,
        clustering_result_payload
            .as_ref()
            .map(|(version, _)| *version),
        clustering_result_payload
            .as_ref()
            .map(|(_, payload)| payload.as_slice()),
    )
}

const TASK_DATA_CLASS: &str = "org/lance/compaction/TaskData";
const COMPACTION_METRICS_CLASS: &str = "org/lance/compaction/CompactionMetrics";
const COMPACTION_METRICS_CONSTRUCTOR_SIG: &str = "(JJJJ)V";
const COMPACTION_PLAN_CLASS: &str = "org/lance/compaction/CompactionPlan";
const COMPACTION_PLAN_CONSTRUCTOR_SIG: &str =
    "(Ljava/util/List;JLorg/lance/compaction/CompactionOptions;)V";
const REWRITE_RESULT_CLASS: &str = "org/lance/compaction/RewriteResult";
const COMPACTION_OPTIONS_CLASS: &str = "org/lance/compaction/CompactionOptions";
const COMPACTION_MODE_CLASS: &str = "org/lance/compaction/CompactionMode";
const COMPACTION_OPTIONS_CONSTRUCTOR_SIG: &str = "(Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/List;)V";

fn task_data_into_java<'local>(
    env: &mut JNIEnv<'local>,
    task: &TaskData,
    fragment_clustering_versions: &HashMap<u64, Option<u64>>,
    clustering_task_payload: Option<&[u8]>,
) -> Result<JObject<'local>> {
    let fragments = export_fragments_with_clustering_versions(env, &task.fragments, |fragment| {
        fragment_clustering_versions
            .get(&fragment.id)
            .copied()
            .flatten()
    })?;
    let clustering_task_payload = match clustering_task_payload {
        Some(payload) => JObject::from(env.byte_array_from_slice(payload)?),
        None => JObject::null(),
    };
    if clustering_task_payload.is_null() {
        Ok(env.new_object(
            TASK_DATA_CLASS,
            "(Ljava/util/List;)V",
            &[JValueGen::Object(&fragments)],
        )?)
    } else {
        Ok(env.new_object(
            "org/lance/compaction/ClusteringTaskData",
            "(Ljava/util/List;[B)V",
            &[
                JValueGen::Object(&fragments),
                JValueGen::Object(&clustering_task_payload),
            ],
        )?)
    }
}

impl IntoJava for &CompactionMetrics {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        Ok(env.new_object(
            COMPACTION_METRICS_CLASS,
            COMPACTION_METRICS_CONSTRUCTOR_SIG,
            &[
                JValueGen::Long(self.fragments_removed as i64),
                JValueGen::Long(self.fragments_added as i64),
                JValueGen::Long(self.files_removed as i64),
                JValueGen::Long(self.files_added as i64),
            ],
        )?)
    }
}

fn compaction_options_into_java<'a>(
    env: &mut JNIEnv<'a>,
    options: &CompactionOptions,
    clustering: bool,
) -> Result<JObject<'a>> {
    let target_rows_per_fragment =
        to_java_long_obj(env, Some(options.target_rows_per_fragment as i64))?;
    let target_rows_per_fragment_opt = to_java_optional(env, target_rows_per_fragment)?;
    let max_rows_per_group = to_java_long_obj(env, Some(options.max_rows_per_group as i64))?;
    let max_rows_per_group_opt = to_java_optional(env, max_rows_per_group)?;
    let max_bytes_per_file = to_java_long_obj(env, options.max_bytes_per_file.map(|v| v as i64))?;
    let max_bytes_per_file_opt = to_java_optional(env, max_bytes_per_file)?;
    let materialize_deletions = to_java_boolean_obj(env, Some(options.materialize_deletions))?;
    let materialize_deletions_opt = to_java_optional(env, materialize_deletions)?;
    let materialize_deletions_threshold =
        to_java_float_obj(env, Some(options.materialize_deletions_threshold))?;
    let materialize_deletions_threshold_opt =
        to_java_optional(env, materialize_deletions_threshold)?;
    let num_threads = to_java_long_obj(env, options.num_threads.map(|v| v as i64))?;
    let num_threads_opt = to_java_optional(env, num_threads)?;
    let batch_size = to_java_long_obj(env, options.batch_size.map(|v| v as i64))?;
    let batch_size_opt = to_java_optional(env, batch_size)?;
    let defer_index_remap = to_java_boolean_obj(env, Some(options.defer_index_remap))?;
    let defer_index_remap_opt = to_java_optional(env, defer_index_remap)?;
    let compaction_mode_obj = if clustering {
        env.get_static_field(
            COMPACTION_MODE_CLASS,
            "CLUSTER",
            format!("L{};", COMPACTION_MODE_CLASS),
        )?
        .l()?
    } else {
        match options.compaction_mode {
            Some(mode) => {
                let name = match mode {
                    CompactionMode::Reencode => "REENCODE",
                    CompactionMode::TryBinaryCopy => "TRY_BINARY_COPY",
                    CompactionMode::ForceBinaryCopy => "FORCE_BINARY_COPY",
                };
                env.get_static_field(
                    COMPACTION_MODE_CLASS,
                    name,
                    format!("L{};", COMPACTION_MODE_CLASS),
                )?
                .l()?
            }
            None => JObject::null(),
        }
    };
    let compaction_mode_opt = to_java_optional(env, compaction_mode_obj)?;
    let binary_copy_read_batch_bytes =
        to_java_long_obj(env, options.binary_copy_read_batch_bytes.map(|v| v as i64))?;
    let binary_copy_read_batch_bytes_opt = to_java_optional(env, binary_copy_read_batch_bytes)?;
    let max_source_fragments =
        to_java_long_obj(env, options.max_source_fragments.map(|v| v as i64))?;
    let max_source_fragments_opt = to_java_optional(env, max_source_fragments)?;
    let max_source_rows = to_java_long_obj(env, options.max_source_rows.map(|v| v as i64))?;
    let max_source_rows_opt = to_java_optional(env, max_source_rows)?;
    let max_source_bytes = to_java_long_obj(env, options.max_source_bytes.map(|v| v as i64))?;
    let max_source_bytes_opt = to_java_optional(env, max_source_bytes)?;
    let excluded_fragment_ids = options
        .excluded_fragment_ids
        .iter()
        .map(|fragment_id| to_java_long_obj(env, Some(*fragment_id as i64)))
        .collect::<Result<Vec<_>>>()?;
    let excluded_fragment_ids = to_java_list(env, &excluded_fragment_ids)?;

    Ok(env.new_object(
        COMPACTION_OPTIONS_CLASS,
        COMPACTION_OPTIONS_CONSTRUCTOR_SIG,
        &[
            JValueGen::Object(&target_rows_per_fragment_opt),
            JValueGen::Object(&max_rows_per_group_opt),
            JValueGen::Object(&max_bytes_per_file_opt),
            JValueGen::Object(&materialize_deletions_opt),
            JValueGen::Object(&materialize_deletions_threshold_opt),
            JValueGen::Object(&num_threads_opt),
            JValueGen::Object(&batch_size_opt),
            JValueGen::Object(&defer_index_remap_opt),
            JValueGen::Object(&compaction_mode_opt),
            JValueGen::Object(&binary_copy_read_batch_bytes_opt),
            JValueGen::Object(&max_source_fragments_opt),
            JValueGen::Object(&max_source_rows_opt),
            JValueGen::Object(&max_source_bytes_opt),
            JValueGen::Object(&excluded_fragment_ids),
        ],
    )?)
}

impl IntoJava for &CompactionOptions {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        compaction_options_into_java(env, self, false)
    }
}

fn compaction_plan_into_java<'local>(
    env: &mut JNIEnv<'local>,
    tasks_data: &[TaskData],
    read_version: u64,
    options: &CompactionOptions,
    fragment_clustering_versions: &HashMap<u64, Option<u64>>,
    clustering_task_payloads: Option<&[Vec<u8>]>,
) -> Result<JObject<'local>> {
    let tasks = env.new_object("java/util/ArrayList", "()V", &[])?;
    for (position, task) in tasks_data.iter().enumerate() {
        let payload = clustering_task_payloads
            .and_then(|payloads| payloads.get(position))
            .map(Vec::as_slice);
        let task = task_data_into_java(env, task, fragment_clustering_versions, payload)?;
        env.call_method(
            &tasks,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValueGen::Object(&task)],
        )?;
    }
    let compaction_options =
        compaction_options_into_java(env, options, clustering_task_payloads.is_some())?;
    Ok(env.new_object(
        COMPACTION_PLAN_CLASS,
        COMPACTION_PLAN_CONSTRUCTOR_SIG,
        &[
            JValueGen::Object(&tasks),
            JValueGen::Long(read_version as i64),
            JValueGen::Object(&compaction_options),
        ],
    )?)
}

fn rewrite_result_into_java<'local>(
    env: &mut JNIEnv<'local>,
    result: &RewriteResult,
    original_fragment_clustering_versions: &HashMap<u64, Option<u64>>,
    new_fragment_clustering_version: Option<u64>,
    clustering_result_payload: Option<&[u8]>,
) -> Result<JObject<'local>> {
    let metrics = result.metrics.into_java(env)?;
    let new_fragments =
        export_fragments_with_clustering_versions(env, &result.new_fragments, |_| {
            new_fragment_clustering_version
        })?;
    let original_fragments =
        export_fragments_with_clustering_versions(env, &result.original_fragments, |fragment| {
            original_fragment_clustering_versions
                .get(&fragment.id)
                .copied()
                .flatten()
        })?;
    let row_addrs: JObject<'_> = if let Some(row_addrs) = &result.row_addrs {
        env.byte_array_from_slice(row_addrs)?.into()
    } else {
        JObject::null()
    };
    if let Some(payload) = clustering_result_payload {
        let payload = JObject::from(env.byte_array_from_slice(payload)?);
        Ok(env.new_object(
            "org/lance/compaction/ClusteringRewriteResult",
            "(Lorg/lance/compaction/CompactionMetrics;Ljava/util/List;Ljava/util/List;J[B[B)V",
            &[
                JValueGen::Object(&metrics),
                JValueGen::Object(&new_fragments),
                JValueGen::Object(&original_fragments),
                JValueGen::Long(result.read_version as i64),
                JValueGen::Object(&row_addrs),
                JValueGen::Object(&payload),
            ],
        )?)
    } else {
        Ok(env.new_object(
            REWRITE_RESULT_CLASS,
            "(Lorg/lance/compaction/CompactionMetrics;Ljava/util/List;Ljava/util/List;J[B)V",
            &[
                JValueGen::Object(&metrics),
                JValueGen::Object(&new_fragments),
                JValueGen::Object(&original_fragments),
                JValueGen::Long(result.read_version as i64),
                JValueGen::Object(&row_addrs),
            ],
        )?)
    }
}

impl FromJObjectWithEnv<CompactionMetrics> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<CompactionMetrics> {
        let fragments_removed = non_negative_usize_from_method(env, self, "getFragmentsRemoved")?;
        let fragments_added = non_negative_usize_from_method(env, self, "getFragmentsAdded")?;
        let files_removed = non_negative_usize_from_method(env, self, "getFilesRemoved")?;
        let files_added = non_negative_usize_from_method(env, self, "getFilesAdded")?;
        Ok(CompactionMetrics {
            fragments_removed,
            fragments_added,
            files_removed,
            files_added,
        })
    }
}

impl FromJObjectWithEnv<ImportedTaskData> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<ImportedTaskData> {
        let fragments: Vec<ImportedFragment> =
            import_vec_from_method(env, self, "getFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let fragments = fragments
            .into_iter()
            .map(|fragment| fragment.fragment)
            .collect();
        Ok(ImportedTaskData {
            task: TaskData { fragments },
        })
    }
}

impl FromJObjectWithEnv<ImportedRewriteResult> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<ImportedRewriteResult> {
        let metrics_obj = env
            .call_method(
                self,
                "getMetrics",
                "()Lorg/lance/compaction/CompactionMetrics;",
                &[],
            )?
            .l()?;
        let metrics = metrics_obj.extract_object(env)?;
        let new_fragments: Vec<ImportedFragment> =
            import_vec_from_method(env, self, "getNewFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let (new_fragments, new_fragment_clustering_version) =
            split_imported_fragments(new_fragments)?;
        let read_version = env.call_method(self, "getReadVersion", "()J", &[])?.j()?;
        let read_version = u64::try_from(read_version).map_err(|_| {
            crate::error::Error::input_error(format!(
                "readVersion must be non-negative, got {read_version}"
            ))
        })?;
        let original_fragments =
            import_vec_from_method(env, self, "getOriginalFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let row_addrs_obj: JByteArray<'_> = env
            .call_method(self, "getRowAddrs", "()[B", &[])?
            .l()?
            .into();
        let row_addrs = if row_addrs_obj.is_null() {
            None
        } else {
            Some(env.convert_byte_array(row_addrs_obj)?)
        };
        let clustering_result_class =
            env.find_class("org/lance/compaction/ClusteringRewriteResult")?;
        let clustering_result_payload = if env.is_instance_of(self, clustering_result_class)? {
            let payload = env
                .call_method(self, "getClusteringResultPayload", "()[B", &[])?
                .l()?;
            Some(env.convert_byte_array(JByteArray::from(payload))?)
        } else {
            None
        };
        Ok(ImportedRewriteResult {
            result: RewriteResult {
                metrics,
                new_fragments,
                read_version,
                original_fragments,
                row_addrs,
            },
            new_fragment_clustering_version,
            clustering_result_payload,
        })
    }
}

fn non_negative_usize_from_method(
    env: &mut JNIEnv<'_>,
    object: &JObject<'_>,
    method: &str,
) -> Result<usize> {
    let value = env.call_method(object, method, "()J", &[])?.j()?;
    usize::try_from(value).map_err(|_| {
        crate::error::Error::input_error(format!(
            "{method} must return a non-negative value that fits usize, got {value}"
        ))
    })
}
