// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use lance::dataset::{
    index::DatasetIndexRemapperOptions,
    optimize::{
        ClusteringCompactionPlan, ClusteringCompactionTask, ClusteringRewriteResult,
        CompactionMetrics, CompactionMode, CompactionOptions, CompactionPlan, CompactionRequest,
        CompactionTask, RewriteResult, commit_clustering_compaction, commit_compaction,
        compact_files, compact_files_with_clustering, plan_clustering_compaction, plan_compaction,
        resolve_compaction_options_from_dataset_config,
    },
};
use pyo3::{exceptions::PyNotImplementedError, pyclass::CompareOp, types::PyTuple};

use super::*;

type FragmentClusteringVersions = std::collections::HashMap<u64, u64>;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
enum CompactionPlanKind {
    Clustering(ClusteringCompactionPlan),
    Ordinary(CompactionPlan),
}

impl CompactionPlanKind {
    fn read_version(&self) -> u64 {
        match self {
            Self::Clustering(plan) => plan.read_version(),
            Self::Ordinary(plan) => plan.read_version(),
        }
    }

    fn num_tasks(&self) -> usize {
        match self {
            Self::Clustering(plan) => plan.num_tasks(),
            Self::Ordinary(plan) => plan.num_tasks(),
        }
    }

    fn task_data(&self) -> &[lance::dataset::optimize::TaskData] {
        match self {
            Self::Clustering(plan) => plan.tasks(),
            Self::Ordinary(plan) => &plan.tasks,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
enum CompactionTaskKind {
    Clustering(ClusteringCompactionTask),
    Ordinary(CompactionTask),
}

impl CompactionTaskKind {
    fn read_version(&self) -> u64 {
        match self {
            Self::Clustering(task) => task.read_version(),
            Self::Ordinary(task) => task.read_version,
        }
    }

    fn fragments(&self) -> &[Fragment] {
        match self {
            Self::Clustering(task) => task.fragments(),
            Self::Ordinary(task) => &task.task.fragments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
enum RewriteResultKind {
    Clustering(ClusteringRewriteResult),
    Ordinary(RewriteResult),
}

impl RewriteResultKind {
    fn rewrite_result(&self) -> &RewriteResult {
        match self {
            Self::Clustering(result) => result.rewrite_result(),
            Self::Ordinary(result) => result,
        }
    }

    fn clustering_version(&self) -> Option<u64> {
        match self {
            Self::Clustering(result) => Some(result.clustering_version()),
            Self::Ordinary(_) => None,
        }
    }
}

#[allow(deprecated)]
fn parse_compaction_options(
    options: &Bound<'_, PyDict>,
    config: &std::collections::HashMap<String, String>,
) -> PyResult<(CompactionOptions, bool)> {
    let mut config = config.clone();
    let explicit_mode = options
        .get_item("compaction_mode")?
        .map(|value| value.extract::<Option<String>>())
        .transpose()?
        .flatten();
    if explicit_mode.is_some() {
        config.remove("lance.compaction.compaction_mode");
    }
    let configured = resolve_compaction_options_from_dataset_config(&config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut is_clustering = matches!(configured, CompactionRequest::Clustering(_));
    let mut opts = configured.into_options();

    for (key, value) in options.into_iter() {
        let key: String = key.extract()?;

        match key.as_str() {
            "target_rows_per_fragment" => {
                opts.target_rows_per_fragment = value.extract()?;
            }
            "max_rows_per_group" => {
                opts.max_rows_per_group = value.extract()?;
            }
            "max_bytes_per_file" => {
                opts.max_bytes_per_file = value.extract()?;
            }
            "materialize_deletions" => {
                opts.materialize_deletions = value.extract()?;
            }
            "materialize_deletions_threshold" => {
                opts.materialize_deletions_threshold = value.extract()?;
            }
            "defer_index_remap" => {
                opts.defer_index_remap = value.extract()?;
            }
            "num_threads" => {
                opts.num_threads = value.extract()?;
            }
            "batch_size" => {
                opts.batch_size = value.extract()?;
            }
            "io_buffer_size" => {
                opts.io_buffer_size = value.extract()?;
            }
            "compaction_mode" => {
                let mode_str: Option<String> = value.extract()?;
                if let Some(mode_str) = mode_str {
                    is_clustering = mode_str.eq_ignore_ascii_case("cluster");
                    if is_clustering {
                        opts.compaction_mode = Some(CompactionMode::Reencode);
                        opts.enable_binary_copy = false;
                        opts.enable_binary_copy_force = false;
                    } else {
                        opts.compaction_mode = Some(
                            CompactionMode::try_from(mode_str.as_str())
                                .map_err(|e| PyValueError::new_err(e.to_string()))?,
                        );
                    }
                }
            }
            "binary_copy_read_batch_bytes" => {
                opts.binary_copy_read_batch_bytes = value.extract()?;
            }
            "max_source_fragments" => {
                opts.max_source_fragments = value.extract()?;
            }
            "max_source_rows" => {
                opts.max_source_rows = value.extract()?;
            }
            "max_source_bytes" => {
                opts.max_source_bytes = value.extract()?;
            }
            "excluded_fragment_ids" => {
                opts.excluded_fragment_ids =
                    value.extract::<Option<Vec<u32>>>()?.unwrap_or_default();
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid compaction option: {}",
                    key
                )));
            }
        }
    }

    Ok((opts, is_clustering))
}

fn unwrap_dataset(dataset: Bound<PyAny>) -> PyResult<Bound<Dataset>> {
    let ds = dataset.getattr("_ds")?;
    Ok(ds.cast::<Dataset>()?.clone())
}

fn wrap_fragment<'py>(
    py: Python<'py>,
    fragment: &Fragment,
    clustering_version: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::fragment::export_fragment_metadata(py, fragment, clustering_version)
}

#[derive(serde::Serialize)]
struct CompactionPlanState {
    #[serde(flatten)]
    plan: CompactionPlanKind,
    #[serde(default, skip_serializing_if = "FragmentClusteringVersions::is_empty")]
    fragment_clustering_versions: FragmentClusteringVersions,
}

#[derive(serde::Serialize)]
struct CompactionTaskState {
    #[serde(flatten)]
    task: CompactionTaskKind,
    #[serde(default, skip_serializing_if = "FragmentClusteringVersions::is_empty")]
    fragment_clustering_versions: FragmentClusteringVersions,
}

#[derive(serde::Serialize)]
struct RewriteResultState {
    #[serde(flatten)]
    result: RewriteResultKind,
    #[serde(default, skip_serializing_if = "FragmentClusteringVersions::is_empty")]
    original_fragment_clustering_versions: FragmentClusteringVersions,
}

fn split_serialized_state(
    json: &str,
    clustering_tag: &str,
    sidecar_key: &str,
) -> PyResult<(serde_json::Value, FragmentClusteringVersions, bool)> {
    let mut value: serde_json::Value =
        serde_json::from_str(json).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| PyValueError::new_err("compaction state must be a JSON object"))?;
    let sidecar = object
        .remove(sidecar_key)
        .map(serde_json::from_value)
        .transpose()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .unwrap_or_default();
    let is_clustering = object.contains_key(clustering_tag);
    if is_clustering && object.len() != 1 {
        return Err(PyValueError::new_err(
            "clustering compaction state cannot contain ordinary payload fields",
        ));
    }
    if !is_clustering
        && object
            .keys()
            .any(|key| key.starts_with("lance_clustering_"))
    {
        return Err(PyValueError::new_err(
            "unexpected clustering payload type in compaction state",
        ));
    }
    Ok((value, sidecar, is_clustering))
}

#[pyclass(name = "CompactionMetrics", module = "lance.optimize")]
pub struct PyCompactionMetrics {
    /// int : The number of fragments that have been overwritten.
    #[pyo3(get)]
    pub fragments_removed: usize,
    /// int : The number of new fragments that have been added.
    #[pyo3(get)]
    pub fragments_added: usize,
    /// int : The number of files that have been removed, including deletion files.
    #[pyo3(get)]
    pub files_removed: usize,
    /// int : The number of files that have been added, which is always equal to the
    /// number of fragments.
    #[pyo3(get)]
    pub files_added: usize,
}

#[pymethods]
impl PyCompactionMetrics {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CompactionMetrics(fragments_removed={}, fragments_added={}, files_removed={}, files_added={})",
            self.fragments_removed, self.fragments_added, self.files_removed, self.files_added
        ))
    }
}

impl From<CompactionMetrics> for PyCompactionMetrics {
    fn from(metrics: CompactionMetrics) -> Self {
        Self {
            fragments_removed: metrics.fragments_removed,
            fragments_added: metrics.fragments_added,
            files_removed: metrics.files_removed,
            files_added: metrics.files_added,
        }
    }
}

/// A plan to compact small dataset fragments into larger ones.
///
/// Created by :py:meth:`lance.optimize.Compaction.plan`.
#[pyclass(name = "CompactionPlan", module = "lance.optimize")]
pub struct PyCompactionPlan {
    plan: CompactionPlanKind,
    fragment_clustering_versions: FragmentClusteringVersions,
}

#[pymethods]
impl PyCompactionPlan {
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CompactionPlan(read_version={}, tasks=<{} compaction tasks>)",
            self.plan.read_version(),
            self.num_tasks()
        ))
    }

    /// int : The read version of the dataset that this plan was created from.
    #[getter]
    pub fn read_version(&self) -> u64 {
        self.plan.read_version()
    }

    /// int : The number of compaction tasks in the plan.
    pub fn num_tasks(&self) -> usize {
        self.plan.num_tasks()
    }

    /// List[CompactionTask] : The individual tasks in the plan.
    #[getter]
    pub fn tasks(&self) -> Vec<PyCompactionTask> {
        match &self.plan {
            CompactionPlanKind::Clustering(plan) => plan
                .compaction_tasks()
                .map(CompactionTaskKind::Clustering)
                .collect::<Vec<_>>(),
            CompactionPlanKind::Ordinary(plan) => plan
                .compaction_tasks()
                .map(CompactionTaskKind::Ordinary)
                .collect::<Vec<_>>(),
        }
        .into_iter()
        .map(|task| {
            let fragment_clustering_versions = task
                .fragments()
                .iter()
                .filter_map(|fragment| {
                    self.fragment_clustering_versions
                        .get(&fragment.id)
                        .map(|version| (fragment.id, *version))
                })
                .collect();
            PyCompactionTask {
                task,
                fragment_clustering_versions,
            }
        })
        .collect()
    }

    /// Get a JSON representation of the plan.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Warning
    /// -------
    /// The JSON representation is not guaranteed to be stable across versions.
    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&CompactionPlanState {
            plan: self.plan.clone(),
            fragment_clustering_versions: self.fragment_clustering_versions.clone(),
        })
        .map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump CompactionPlan due to error: {}",
                err
            ))
        })
    }

    /// Load a plan from a JSON representation.
    ///
    /// Parameters
    /// ----------
    /// json : str
    ///     The JSON representation of the plan.
    ///
    /// Returns
    /// -------
    /// CompactionPlan
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let (value, fragment_clustering_versions, is_clustering) = split_serialized_state(
            &json,
            "lance_clustering_compaction_plan_v1",
            "fragment_clustering_versions",
        )?;
        let plan = if is_clustering {
            CompactionPlanKind::Clustering(serde_json::from_value(value).map_err(|err| {
                PyValueError::new_err(format!(
                    "Could not load CompactionPlan due to error: {}",
                    err
                ))
            })?)
        } else {
            CompactionPlanKind::Ordinary(serde_json::from_value(value).map_err(|err| {
                PyValueError::new_err(format!(
                    "Could not load CompactionPlan due to error: {}",
                    err
                ))
            })?)
        };
        Ok(Self {
            plan,
            fragment_clustering_versions,
        })
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state])?.extract()?;
        let from_json = PyModule::import(py, "lance.optimize")?
            .getattr("CompactionPlan")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.plan == other.plan
                && self.fragment_clustering_versions == other.fragment_clustering_versions),
            CompareOp::Ne => Ok(self.plan != other.plan
                || self.fragment_clustering_versions != other.fragment_clustering_versions),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for CompactionTask",
            )),
        }
    }
}

#[pyclass(name = "CompactionTask", module = "lance.optimize", from_py_object)]
#[derive(Clone)]
pub struct PyCompactionTask {
    task: CompactionTaskKind,
    fragment_clustering_versions: FragmentClusteringVersions,
}

#[pymethods]
impl PyCompactionTask {
    pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let fragment_reprs: String = self
            .fragments(py)?
            .iter()
            .map(|f| f.call_method0("__repr__")?.extract())
            .collect::<PyResult<Vec<String>>>()?
            .join(", ");
        Ok(format!(
            "CompactionTask(read_version={}, fragments=[{}])",
            self.task.read_version(),
            fragment_reprs
        ))
    }

    /// int : The read version of the dataset that this task was created from.
    #[getter]
    pub fn read_version(&self) -> u64 {
        self.task.read_version()
    }

    /// List[lance.fragment.FragmentMetadata] : The fragments that will be compacted.
    #[getter]
    pub fn fragments<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.task
            .fragments()
            .iter()
            .map(|fragment| {
                wrap_fragment(
                    py,
                    fragment,
                    self.fragment_clustering_versions.get(&fragment.id).copied(),
                )
            })
            .collect()
    }

    /// Execute the compaction task and return the :py:class:`RewriteResult`.
    ///
    /// The rewrite result should be passed onto :py:meth:`lance.optimize.Compaction.commit`.
    pub fn execute(&self, dataset: Bound<PyAny>) -> PyResult<PyRewriteResult> {
        let dataset = unwrap_dataset(dataset)?;
        let dataset = dataset.borrow().clone();
        let result = match &self.task {
            CompactionTaskKind::Clustering(task) => RewriteResultKind::Clustering(
                rt().block_on(None, task.execute(dataset.ds.as_ref()))?
                    .infer_error()?,
            ),
            CompactionTaskKind::Ordinary(task) => RewriteResultKind::Ordinary(
                rt().block_on(None, task.execute(dataset.ds.as_ref()))?
                    .io_error()?,
            ),
        };

        Ok(PyRewriteResult {
            result,
            original_fragment_clustering_versions: self.fragment_clustering_versions.clone(),
        })
    }

    /// Get a JSON representation of the task.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Warning
    /// -------
    /// The JSON representation is not guaranteed to be stable across versions.
    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&CompactionTaskState {
            task: self.task.clone(),
            fragment_clustering_versions: self.fragment_clustering_versions.clone(),
        })
        .map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump CompactionTask due to error: {}",
                err
            ))
        })
    }

    /// Load a task from a JSON representation.
    ///
    /// Parameters
    /// ----------
    /// json : str
    ///     The JSON representation of the task.
    ///
    /// Returns
    /// -------
    /// CompactionTask
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let (value, fragment_clustering_versions, is_clustering) = split_serialized_state(
            &json,
            "lance_clustering_compaction_task_v1",
            "fragment_clustering_versions",
        )?;
        let task = if is_clustering {
            CompactionTaskKind::Clustering(serde_json::from_value(value).map_err(|err| {
                PyValueError::new_err(format!(
                    "Could not load CompactionTask due to error: {}",
                    err
                ))
            })?)
        } else {
            CompactionTaskKind::Ordinary(serde_json::from_value(value).map_err(|err| {
                PyValueError::new_err(format!(
                    "Could not load CompactionTask due to error: {}",
                    err
                ))
            })?)
        };
        Ok(Self {
            task,
            fragment_clustering_versions,
        })
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state])?.extract()?;
        let from_json = PyModule::import(py, "lance.optimize")?
            .getattr("CompactionTask")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.task == other.task
                && self.fragment_clustering_versions == other.fragment_clustering_versions),
            CompareOp::Ne => Ok(self.task != other.task
                || self.fragment_clustering_versions != other.fragment_clustering_versions),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for CompactionTask",
            )),
        }
    }
}

/// The result of a single compaction task.
///
/// Created by :py:meth:`lance.optimize.CompactionTask.execute`.
///
/// This result is pickle-able, so it can be serialized and sent back to the
/// main process to be passed to :py:meth:`lance.optimize.Compaction.commit`.
#[pyclass(name = "RewriteResult", module = "lance.optimize", from_py_object)]
#[derive(Clone)]
pub struct PyRewriteResult {
    result: RewriteResultKind,
    original_fragment_clustering_versions: FragmentClusteringVersions,
}

#[pymethods]
impl PyRewriteResult {
    pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let orig_fragment_reprs: String = self
            .original_fragments(py)?
            .iter()
            .map(|f| f.call_method0("__repr__")?.extract())
            .collect::<PyResult<Vec<String>>>()?
            .join(", ");
        let new_fragment_reprs: String = self
            .new_fragments(py)?
            .iter()
            .map(|f| f.call_method0("__repr__")?.extract())
            .collect::<PyResult<Vec<String>>>()?
            .join(", ");

        Ok(format!(
            "RewriteResult(read_version={}, new_fragments=[{}], old_fragments=[{}])",
            self.result.rewrite_result().read_version,
            new_fragment_reprs,
            orig_fragment_reprs,
        ))
    }

    /// int : The version of the dataset the optimize operation is based on.
    #[getter]
    pub fn read_version(&self) -> u64 {
        self.result.rewrite_result().read_version
    }

    /// List[lance.fragment.FragmentMetadata] : The metadata for fragments that are being replaced.
    #[getter]
    pub fn original_fragments<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.result
            .rewrite_result()
            .original_fragments
            .iter()
            .map(|fragment| {
                wrap_fragment(
                    py,
                    fragment,
                    self.original_fragment_clustering_versions
                        .get(&fragment.id)
                        .copied(),
                )
            })
            .collect()
    }

    /// List[lance.fragment.FragmentMetadata] : The metadata for fragments that are being added.
    #[getter]
    pub fn new_fragments<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.result
            .rewrite_result()
            .new_fragments
            .iter()
            .map(|fragment| wrap_fragment(py, fragment, self.result.clustering_version()))
            .collect()
    }

    /// Get a JSON representation of the result.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Warning
    /// -------
    /// The JSON representation is not guaranteed to be stable across versions.
    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&RewriteResultState {
            result: self.result.clone(),
            original_fragment_clustering_versions: self
                .original_fragment_clustering_versions
                .clone(),
        })
        .map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump RewriteResult due to error: {}",
                err
            ))
        })
    }

    /// Load a result from a JSON representation.
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let (value, original_fragment_clustering_versions, is_clustering) = split_serialized_state(
            &json,
            "lance_clustering_rewrite_result_v1",
            "original_fragment_clustering_versions",
        )?;
        let result = if is_clustering {
            RewriteResultKind::Clustering(serde_json::from_value(value).map_err(|err| {
                PyValueError::new_err(format!(
                    "Could not load RewriteResult due to error: {}",
                    err
                ))
            })?)
        } else {
            RewriteResultKind::Ordinary(serde_json::from_value(value).map_err(|err| {
                PyValueError::new_err(format!(
                    "Could not load RewriteResult due to error: {}",
                    err
                ))
            })?)
        };
        Ok(Self {
            result,
            original_fragment_clustering_versions,
        })
    }

    /// CompactionMetrics : The metrics from this compaction task.
    #[getter]
    pub fn metrics(&self) -> PyResult<PyCompactionMetrics> {
        Ok(self.result.rewrite_result().metrics.clone().into())
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state])?.extract()?;
        let from_json = PyModule::import(py, "lance.optimize")?
            .getattr("RewriteResult")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.result == other.result
                && self.original_fragment_clustering_versions
                    == other.original_fragment_clustering_versions),
            CompareOp::Ne => Ok(self.result != other.result
                || self.original_fragment_clustering_versions
                    != other.original_fragment_clustering_versions),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for RewriteResult",
            )),
        }
    }
}

/// File compaction operation.
///
/// To run with multiple threads in a single process, just use :py:meth:`execute()`.
///
/// To run with multiple processes, first use :py:meth:`plan()` to construct a
/// plan, then execute the tasks in parallel, and finally use :py:meth:`commit()`.
/// The :py:class:`CompactionPlan` contains many :py:class:`CompactionTask` objects,
/// which can be pickled and sent to other processes. The tasks produce
/// :py:class:`RewriteResult` objects, which can be pickled and sent back to the
/// main process to be passed to :py:meth:`commit()`.
#[pyclass(name = "Compaction", module = "lance.optimize")]
pub struct PyCompaction;

#[pymethods]
impl PyCompaction {
    /// Execute a full compaction operation.
    ///
    /// Parameters
    /// ----------
    /// dataset : lance.Dataset
    ///    The dataset to compact. The dataset instance will be updated to the
    ///    new version once complete.
    /// options : CompactionOptions
    ///    The compaction options.
    ///
    /// Returns
    /// -------
    /// CompactionMetrics
    ///     The metrics from the compaction operation.
    #[staticmethod]
    pub fn execute(dataset: Bound<PyAny>, options: Bound<PyAny>) -> PyResult<PyCompactionMetrics> {
        let dataset_ref = unwrap_dataset(dataset)?;
        let dataset = dataset_ref.borrow().clone();
        // Make sure we parse the options within a scoped GIL context, so we
        // aren't holding the GIL while blocking the thread on the operation.
        let options = options.cast::<PyDict>()?;
        let config = dataset.ds.manifest.config.clone();
        let (opts, is_clustering) = parse_compaction_options(options, &config)?;
        let mut new_ds = dataset.ds.as_ref().clone();
        let result = if is_clustering {
            rt().block_on(None, compact_files_with_clustering(&mut new_ds, opts))?
                .infer_error()?
        } else {
            rt().block_on(None, compact_files(&mut new_ds, opts, None))?
                .io_error()?
        };
        dataset_ref.borrow_mut().ds = Arc::new(new_ds);
        Ok(result.into())
    }

    /// Plan a compaction operation.
    ///
    /// This is intended for users who want to run compaction in a distributed
    /// fashion. For running on a single process, use :py:meth:`execute()`
    /// instead.
    ///
    /// Parameters
    /// ----------
    /// dataset : lance.Dataset
    ///   The dataset to compact.
    /// options : CompactionOptions
    ///   The compaction options.
    ///
    /// Returns
    /// -------
    /// CompactionPlan
    #[staticmethod]
    pub fn plan(dataset: Bound<PyAny>, options: Bound<PyAny>) -> PyResult<PyCompactionPlan> {
        let dataset = unwrap_dataset(dataset)?;
        let dataset = dataset.borrow().clone();
        // Make sure we parse the options within a scoped GIL context, so we
        // aren't holding the GIL while blocking the thread on the operation.
        let options = options.cast::<PyDict>()?;
        let config = dataset.ds.manifest.config.clone();
        let (opts, is_clustering) = parse_compaction_options(options, &config)?;
        let plan = if is_clustering {
            CompactionPlanKind::Clustering(
                rt().block_on(None, plan_clustering_compaction(dataset.ds.as_ref(), &opts))?
                    .infer_error()?,
            )
        } else {
            CompactionPlanKind::Ordinary(
                rt().block_on(None, plan_compaction(dataset.ds.as_ref(), &opts))?
                    .io_error()?,
            )
        };
        let planned_fragment_ids = plan
            .task_data()
            .iter()
            .flat_map(|task| task.fragments.iter().map(|fragment| fragment.id))
            .collect::<std::collections::HashSet<_>>();
        let fragment_clustering_versions = planned_fragment_ids
            .into_iter()
            .filter_map(|fragment_id| {
                dataset
                    .ds
                    .manifest()
                    .fragment_clustering_version(fragment_id)
                    .map(|version| (fragment_id, version))
            })
            .collect();
        Ok(PyCompactionPlan {
            plan,
            fragment_clustering_versions,
        })
    }

    /// Commit a compaction operation.
    ///
    /// Once tasks from :py:meth:`plan()` have been executed, the results can
    /// be passed to this method to commit the compaction. It is not required
    /// that all of the original tasks are passed. For example, if only a subset
    /// were successful or completed before a deadline, you can pass just those.
    ///
    /// Parameters
    /// ----------
    /// dataset : lance.Dataset
    ///     The dataset to compact. The dataset instance will be updated to the
    ///     new version once committed.
    /// rewrites : List[RewriteResult]
    ///     The results of the compaction tasks to include in the commit.
    /// options : dict, optional
    ///     Compaction options to apply at commit time.
    ///     When absent or ``None``, defaults to ``CompactionOptions::default()``.
    ///
    /// Returns
    /// -------
    /// CompactionMetrics
    #[staticmethod]
    #[pyo3(signature = (dataset, rewrites, options = None))]
    pub fn commit(
        dataset: Bound<PyAny>,
        rewrites: Vec<PyRewriteResult>,
        options: Option<Bound<PyDict>>,
    ) -> PyResult<PyCompactionMetrics> {
        let dataset_ref = unwrap_dataset(dataset)?;
        let dataset = dataset_ref.borrow().clone();
        let config = dataset.ds.manifest.config.clone();
        let explicit_clustering = options
            .as_ref()
            .and_then(|dict| dict.get_item("compaction_mode").ok().flatten())
            .and_then(|value| value.extract::<Option<String>>().ok().flatten())
            .map(|mode| mode.eq_ignore_ascii_case("cluster"));
        let (mut opts, configured_clustering) = match options {
            Some(ref dict) => parse_compaction_options(dict, &config)?,
            None => (CompactionOptions::default(), false),
        };
        let result_kind = rewrites.first().map(|rewrite| match rewrite.result {
            RewriteResultKind::Clustering(_) => true,
            RewriteResultKind::Ordinary(_) => false,
        });
        if rewrites
            .iter()
            .any(|rewrite| matches!(rewrite.result, RewriteResultKind::Clustering(_)))
            && rewrites
                .iter()
                .any(|rewrite| matches!(rewrite.result, RewriteResultKind::Ordinary(_)))
        {
            return Err(PyValueError::new_err(
                "cannot commit a mixture of ordinary and clustering RewriteResult values",
            ));
        }
        if let (Some(requested), Some(actual)) = (explicit_clustering, result_kind)
            && requested != actual
        {
            return Err(PyValueError::new_err(
                "compaction_mode cannot change the operation kind carried by RewriteResult",
            ));
        }
        let is_clustering = result_kind.unwrap_or(configured_clustering);
        if is_clustering {
            opts.compaction_mode = Some(CompactionMode::Reencode);
            #[allow(deprecated)]
            {
                opts.enable_binary_copy = false;
                opts.enable_binary_copy_force = false;
            }
        }
        let mut new_ds = dataset.ds.as_ref().clone();
        let result = if is_clustering {
            let rewrites = rewrites
                .into_iter()
                .map(|rewrite| match rewrite.result {
                    RewriteResultKind::Clustering(result) => Ok(result),
                    RewriteResultKind::Ordinary(_) => Err(PyValueError::new_err(
                        "ordinary RewriteResult cannot be committed as clustering compaction",
                    )),
                })
                .collect::<PyResult<Vec<_>>>()?;
            rt().block_on(
                None,
                commit_clustering_compaction(&mut new_ds, rewrites, &opts),
            )?
            .infer_error()?
        } else {
            let rewrites = rewrites
                .into_iter()
                .map(|rewrite| match rewrite.result {
                    RewriteResultKind::Ordinary(result) => Ok(result),
                    RewriteResultKind::Clustering(_) => Err(PyValueError::new_err(
                        "clustering RewriteResult cannot be committed as ordinary compaction",
                    )),
                })
                .collect::<PyResult<Vec<_>>>()?;
            rt().block_on(
                None,
                commit_compaction(
                    &mut new_ds,
                    rewrites,
                    Arc::new(DatasetIndexRemapperOptions::default()),
                    &opts,
                ),
            )?
            .io_error()?
        };
        dataset_ref.borrow_mut().ds = Arc::new(new_ds);
        Ok(result.into())
    }
}
