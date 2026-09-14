// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StructArray, UInt64Array};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi_and_data_type};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::RecordBatchReader;
use arrow_schema::DataType;
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObject, JString, JValueGen};
use jni::sys::jlong;
use lance::dataset::optimize::{
    ReclusterGroup, ReclusterPlan, ReclusterResult, commit_recluster, plan_recluster,
};
use lance_core::utils::address::RowAddress;
use lance_index::clustering::{ClusteringModel, PartialClusteringModel};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::block_on;
use crate::blocking_dataset::{
    BlockingDataset, NATIVE_DATASET, convert_java_compaction_options_to_rust,
};
use crate::error::{Error, Result};
use crate::traits::{FromJObjectWithEnv, FromJString, IntoJava, export_vec, import_vec};
use crate::utils::{to_java_list, to_java_long_obj};

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_clustering_Clustering_nativePlanRecluster<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    java_dataset: JObject<'local>,
    java_options: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_plan_recluster(&mut env, java_dataset, java_options),
        JObject::null()
    )
}

fn inner_plan_recluster<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject<'local>,
    java_options: JObject<'local>,
) -> Result<JObject<'local>> {
    let (dataset, config) = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        (dataset.inner.clone(), dataset.inner.manifest.config.clone())
    };
    let options = convert_java_compaction_options_to_rust(env, java_options, &config)?;
    let plan = block_on(plan_recluster(&dataset, &options))?;
    (&plan).into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_clustering_Clustering_nativeBuildPartialModel<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    plan: JByteArray<'local>,
    batch_array_addr: jlong,
    batch_schema_addr: jlong,
    row_address_array_addr: jlong,
    row_address_schema_addr: jlong,
) -> JByteArray<'local> {
    ok_or_throw_with_return!(
        env,
        inner_build_partial_model(
            &mut env,
            plan,
            batch_array_addr,
            batch_schema_addr,
            row_address_array_addr,
            row_address_schema_addr
        ),
        JByteArray::default()
    )
}

fn inner_build_partial_model<'local>(
    env: &mut JNIEnv<'local>,
    plan: JByteArray<'local>,
    batch_array_addr: jlong,
    batch_schema_addr: jlong,
    row_address_array_addr: jlong,
    row_address_schema_addr: jlong,
) -> Result<JByteArray<'local>> {
    let plan = ReclusterPlan::from_bytes(&env.convert_byte_array(plan)?)?;
    let batch = import_record_batch(batch_array_addr, batch_schema_addr)?;
    let row_addresses = import_row_addresses(row_address_array_addr, row_address_schema_addr)?;
    let planned_fragments: RoaringBitmap = plan
        .groups()
        .iter()
        .flat_map(ReclusterGroup::source_fragment_ids)
        .map(|fragment_id| fragment_id as u32)
        .collect();
    if let Some(row_address) = row_addresses.values().iter().copied().find(|row_address| {
        !planned_fragments.contains(RowAddress::from(*row_address).fragment_id())
    }) {
        return Err(Error::input_error(format!(
            "clustering row address {row_address} does not belong to a source fragment in this plan"
        )));
    }
    let payload = PartialClusteringModel::from_batch_bytes(
        plan.model_context().to_vec(),
        plan.columns().to_vec(),
        &batch,
        &row_addresses,
    )?;
    Ok(env.byte_array_from_slice(&payload)?)
}

fn import_row_addresses(
    row_address_array_addr: jlong,
    row_address_schema_addr: jlong,
) -> Result<UInt64Array> {
    let row_addresses = import_array(row_address_array_addr, row_address_schema_addr)?;
    match row_addresses.data_type() {
        DataType::UInt64 => {
            let values = row_addresses
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    Error::input_error("invalid UInt64 row address array".to_string())
                })?;
            if values.null_count() != 0 {
                return Err(Error::input_error(
                    "clustering row addresses must not contain nulls".to_string(),
                ));
            }
            Ok(values.clone())
        }
        DataType::Int64 => {
            let values = row_addresses
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::input_error("invalid Int64 row address array".to_string()))?;
            if values.null_count() != 0 {
                return Err(Error::input_error(
                    "clustering row addresses must not contain nulls".to_string(),
                ));
            }
            Ok(UInt64Array::from_iter_values(
                values.values().iter().map(|value| *value as u64),
            ))
        }
        data_type => Err(Error::input_error(format!(
            "clustering row addresses must be Int64 or UInt64, got {data_type:?}"
        ))),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_clustering_Clustering_nativeMergePartialModels<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    partials: JObject<'local>,
) -> JByteArray<'local> {
    ok_or_throw_with_return!(
        env,
        inner_merge_partial_models(&mut env, partials),
        JByteArray::default()
    )
}

fn inner_merge_partial_models<'local>(
    env: &mut JNIEnv<'local>,
    partials: JObject<'local>,
) -> Result<JByteArray<'local>> {
    let partials = import_partial_payloads(env, partials)?;
    let merged = PartialClusteringModel::merge_bytes(partials)?;
    Ok(env.byte_array_from_slice(&merged)?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_clustering_Clustering_nativeDigestRowAddresses<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    row_address_array_addr: jlong,
    row_address_schema_addr: jlong,
) -> JByteArray<'local> {
    ok_or_throw_with_return!(
        env,
        inner_digest_row_addresses(&mut env, row_address_array_addr, row_address_schema_addr),
        JByteArray::default()
    )
}

fn inner_digest_row_addresses<'local>(
    env: &mut JNIEnv<'local>,
    row_address_array_addr: jlong,
    row_address_schema_addr: jlong,
) -> Result<JByteArray<'local>> {
    let row_addresses = import_row_addresses(row_address_array_addr, row_address_schema_addr)?;
    let digest = lance_index::clustering::RowDigest::from_row_addresses(&row_addresses)?.to_bytes();
    Ok(env.byte_array_from_slice(&digest)?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_clustering_ClusteringModel_nativeMerge<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    partials: JObject<'local>,
) -> JByteArray<'local> {
    ok_or_throw_with_return!(
        env,
        inner_merge_models(&mut env, partials),
        JByteArray::default()
    )
}

fn inner_merge_models<'local>(
    env: &mut JNIEnv<'local>,
    partials: JObject<'local>,
) -> Result<JByteArray<'local>> {
    let partials = import_partial_payloads(env, partials)?;
    let model = PartialClusteringModel::merge_bytes_to_model(partials)?;
    Ok(env.byte_array_from_slice(&model.to_bytes())?)
}

fn import_partial_payloads(env: &mut JNIEnv, partials: JObject) -> Result<Vec<Vec<u8>>> {
    import_vec(env, &partials)?
        .into_iter()
        .map(|partial| {
            env.convert_byte_array(JByteArray::from(partial))
                .map_err(Into::into)
        })
        .collect()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_clustering_ClusteringModel_nativeEncode(
    mut env: JNIEnv,
    _class: JClass,
    model: JByteArray,
    batch_array_addr: jlong,
    batch_schema_addr: jlong,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_encode(
            &mut env,
            model,
            batch_array_addr,
            batch_schema_addr,
            stream_addr
        )
    );
}

fn inner_encode(
    env: &mut JNIEnv,
    model: JByteArray,
    batch_array_addr: jlong,
    batch_schema_addr: jlong,
    stream_addr: jlong,
) -> Result<()> {
    let model = ClusteringModel::from_bytes(&env.convert_byte_array(model)?)?;
    let batch = import_record_batch(batch_array_addr, batch_schema_addr)?;
    let keys = model.encode_batch(&batch)?;
    let result = RecordBatch::try_from_iter([("__lance_clustering_key", keys)])?;
    let schema = result.schema();
    let reader: Box<dyn RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new([Ok(result)].into_iter(), schema));
    let stream = FFI_ArrowArrayStream::new(reader);
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, stream) };
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_clustering_Clustering_nativeCreateResult<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    plan: JByteArray<'local>,
    group_id: JString<'local>,
    model: JByteArray<'local>,
    new_fragments: JObject<'local>,
    output_row_digests: JObject<'local>,
) -> JByteArray<'local> {
    ok_or_throw_with_return!(
        env,
        inner_create_result(
            &mut env,
            plan,
            group_id,
            model,
            new_fragments,
            output_row_digests
        ),
        JByteArray::default()
    )
}

fn inner_create_result<'local>(
    env: &mut JNIEnv<'local>,
    plan: JByteArray<'local>,
    group_id: JString<'local>,
    model: JByteArray<'local>,
    new_fragments: JObject<'local>,
    output_row_digests: JObject<'local>,
) -> Result<JByteArray<'local>> {
    let plan = ReclusterPlan::from_bytes(&env.convert_byte_array(plan)?)?;
    let group_id = Uuid::parse_str(&group_id.extract(env)?)
        .map_err(|error| Error::input_error(format!("invalid recluster group id: {error}")))?;
    let model = ClusteringModel::from_bytes(&env.convert_byte_array(model)?)?;
    let new_fragments = import_vec(env, &new_fragments)?
        .into_iter()
        .map(|fragment| fragment.extract_object(env))
        .collect::<Result<Vec<_>>>()?;
    let output_row_digests = import_vec(env, &output_row_digests)?
        .into_iter()
        .map(|digest| {
            env.convert_byte_array(JByteArray::from(digest))
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut output_row_digest = lance_index::clustering::RowDigest::default();
    for digest in output_row_digests {
        let digest: [u8; 32] = digest.try_into().map_err(|digest: Vec<u8>| {
            Error::input_error(format!(
                "clustering output row digest must contain 32 bytes, got {}",
                digest.len()
            ))
        })?;
        output_row_digest.merge(lance_index::clustering::RowDigest::from_bytes(digest))?;
    }
    let result = ReclusterResult::try_new(
        &plan,
        group_id,
        &model,
        new_fragments,
        output_row_digest.to_bytes(),
    )?;
    Ok(env.byte_array_from_slice(&result.to_bytes())?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_clustering_Clustering_nativeCommitRecluster<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    java_dataset: JObject<'local>,
    plan: JByteArray<'local>,
    model: JByteArray<'local>,
    results: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_commit_recluster(&mut env, java_dataset, plan, model, results),
        JObject::null()
    )
}

fn inner_commit_recluster<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject<'local>,
    plan: JByteArray<'local>,
    model: JByteArray<'local>,
    results: JObject<'local>,
) -> Result<JObject<'local>> {
    let plan = ReclusterPlan::from_bytes(&env.convert_byte_array(plan)?)?;
    let model = ClusteringModel::from_bytes(&env.convert_byte_array(model)?)?;
    let results = import_vec(env, &results)?
        .into_iter()
        .map(|result| {
            let payload = env
                .call_method(&result, "getPayload", "()[B", &[])
                .and_then(|value| value.l())?;
            let bytes = env.convert_byte_array(JByteArray::from(payload))?;
            ReclusterResult::from_bytes(&bytes).map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let metrics = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(commit_recluster(&mut dataset.inner, &plan, &model, results))?
    };
    (&metrics).into_java(env)
}

fn import_record_batch(array_addr: jlong, schema_addr: jlong) -> Result<RecordBatch> {
    let array = import_array(array_addr, schema_addr)?;
    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| Error::input_error("expected an Arrow struct array batch".to_string()))?;
    Ok(RecordBatch::from(struct_array.clone()))
}

fn import_array(array_addr: jlong, schema_addr: jlong) -> Result<arrow::array::ArrayRef> {
    let ffi_array =
        unsafe { std::ptr::replace(array_addr as *mut FFI_ArrowArray, FFI_ArrowArray::empty()) };
    let ffi_schema = unsafe {
        std::ptr::replace(
            schema_addr as *mut FFI_ArrowSchema,
            FFI_ArrowSchema::empty(),
        )
    };
    let data_type = DataType::try_from(&ffi_schema)?;
    let data = unsafe { from_ffi_and_data_type(ffi_array, data_type) }?;
    Ok(arrow::array::make_array(data))
}

impl IntoJava for &ReclusterGroup {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        let id = self.id().into_java(env)?;
        let source_ids = self
            .source_fragment_ids()
            .map(|id| to_java_long_obj(env, Some(id as i64)))
            .collect::<Result<Vec<_>>>()?;
        let source_ids = to_java_list(env, &source_ids)?;
        Ok(env.new_object(
            "org/lance/clustering/ReclusterGroup",
            "(Ljava/util/UUID;Ljava/util/List;J)V",
            &[
                JValueGen::Object(&id),
                JValueGen::Object(&source_ids),
                JValueGen::Long(self.expected_live_rows() as i64),
            ],
        )?)
    }
}

impl IntoJava for &ReclusterPlan {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        let payload = JObject::from(env.byte_array_from_slice(&self.to_bytes())?);
        let columns = self
            .columns()
            .iter()
            .map(|column| {
                env.new_string(column)
                    .map(JObject::from)
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        let columns = to_java_list(env, &columns)?;
        let groups = export_vec(env, self.groups())?;
        Ok(env.new_object(
            "org/lance/clustering/ReclusterPlan",
            "([BJJLjava/util/List;Ljava/util/List;)V",
            &[
                JValueGen::Object(&payload),
                JValueGen::Long(self.read_version() as i64),
                JValueGen::Long(self.clustering_version() as i64),
                JValueGen::Object(&columns),
                JValueGen::Object(&groups),
            ],
        )?)
    }
}
