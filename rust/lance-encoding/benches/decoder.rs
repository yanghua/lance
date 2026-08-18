// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
#![allow(clippy::print_stdout)]

use std::{collections::HashMap, hint::black_box, ops::Range, sync::Arc};

use arrow_array::{BinaryArray, RecordBatch, UInt32Array};
#[cfg(feature = "bitpacking")]
use arrow_buffer::ArrowNativeType;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use arrow_select::take::take;
#[cfg(feature = "bitpacking")]
use bytemuck::Pod;
use bytes::BytesMut;
use criterion::{Criterion, criterion_group, criterion_main};
use futures::StreamExt;
#[cfg(feature = "bitpacking")]
use lance_bitpacking::BitPacking;
use lance_core::cache::LanceCache;
use lance_datagen::ArrayGeneratorExt;
#[cfg(feature = "bitpacking")]
use lance_encoding::buffer::LanceBuffer;
#[cfg(feature = "bitpacking")]
use lance_encoding::compression::BlockDecompressor;
#[cfg(feature = "bitpacking")]
use lance_encoding::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
#[cfg(feature = "bitpacking")]
use lance_encoding::encodings::physical::bitpacking::{ELEMS_PER_CHUNK, InlineBitpacking};
use lance_encoding::format::pb21;
use lance_encoding::{
    decoder::{
        ColumnInfo, DecodeBatchScheduler, DecoderConfig, DecoderPlugins, EncodedBatchLayout,
        FilterExpression, PageInfo, create_decode_stream,
    },
    encoder::{
        BatchEncoder, EncodedBatch, EncodedPage, EncodingOptions, OutOfLineBuffers, encode_batch,
    },
    repdef::RepDefBuilder,
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::unbounded_channel;

use rand::Rng;

pub mod common;
use common::{BenchEncoding, encoding_strategy};

const PRIMITIVE_TYPES: &[DataType] = &[
    DataType::Date32,
    DataType::Date64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::Decimal128(10, 10),
    DataType::Decimal256(10, 10),
    DataType::Timestamp(TimeUnit::Nanosecond, None),
    DataType::Time32(TimeUnit::Second),
    DataType::Time64(TimeUnit::Nanosecond),
    DataType::Duration(TimeUnit::Second),
    // The Interval type is supported by the reader but the writer works with Lance schema
    // at the moment and Lance schema can't parse interval
    // DataType::Interval(IntervalUnit::DayTime),
];

// Some types are supported by the encoder/decoder but Lance
// schema doesn't yet parse them in the context of a fixed size list.
const PRIMITIVE_TYPES_FOR_FSL: &[DataType] = &[DataType::Int8, DataType::Float32];

fn encoded_batch_layout(encoding: BenchEncoding) -> EncodedBatchLayout {
    match encoding {
        BenchEncoding::Array => EncodedBatchLayout::Array,
        BenchEncoding::StructuralU16 | BenchEncoding::StructuralU32 => {
            EncodedBatchLayout::Structural
        }
    }
}

fn bench_decode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");
    const NUM_BYTES: u64 = 1024 * 1024 * 128;
    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));
    for data_type in PRIMITIVE_TYPES {
        let func_name = format!("{:?}", data_type).to_lowercase();
        let num_rows = NUM_BYTES / data_type.primitive_width().unwrap() as u64;
        group.bench_function(func_name, |b| {
            let data = lance_datagen::gen_batch()
                .anon_col(lance_datagen::array::rand_type(data_type))
                .into_batch_rows(lance_datagen::RowCount::from(num_rows))
                .unwrap();
            let lance_schema =
                Arc::new(lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap());
            let encoding_strategy = encoding_strategy(BenchEncoding::StructuralU16);
            let encoded = rt
                .block_on(encode_batch(
                    &data,
                    lance_schema,
                    encoding_strategy.as_ref(),
                    &EncodingOptions::default(),
                ))
                .unwrap();

            b.iter(|| {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        EncodedBatchLayout::Structural,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data.num_rows(), batch.num_rows());
            })
        });
    }
}

fn bench_decode_fsl(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_fsl");
    const NUM_BYTES: u64 = 1024 * 1024 * 128;
    for encoding in [
        BenchEncoding::Array,
        BenchEncoding::StructuralU16,
        BenchEncoding::StructuralU32,
    ] {
        for data_type in PRIMITIVE_TYPES_FOR_FSL {
            for dimension in [4, 16, 32, 64, 128] {
                let nullable_choices: &[bool] = if encoding == BenchEncoding::Array {
                    &[false]
                } else {
                    &[false, true]
                };
                for nullable in nullable_choices {
                    let func_name = format!(
                        "{:?}_{}_v{}_null{}",
                        data_type, dimension, encoding, nullable
                    )
                    .to_lowercase();
                    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));
                    group.bench_function(func_name, |b| {
                        let num_rows =
                            NUM_BYTES / (dimension * data_type.primitive_width().unwrap() as u64);
                        let mut arraygen =
                            lance_datagen::array::rand_type(&DataType::FixedSizeList(
                                Arc::new(Field::new("item", data_type.clone(), true)),
                                dimension as i32,
                            ));
                        if *nullable {
                            arraygen = arraygen.with_random_nulls(0.5);
                        }
                        let data = lance_datagen::gen_batch()
                            .anon_col(arraygen)
                            .into_batch_rows(lance_datagen::RowCount::from(num_rows))
                            .unwrap();
                        let lance_schema = Arc::new(
                            lance_core::datatypes::Schema::try_from(data.schema().as_ref())
                                .unwrap(),
                        );
                        let encoding_strategy = encoding_strategy(encoding);
                        let encoded = rt
                            .block_on(encode_batch(
                                &data,
                                lance_schema,
                                encoding_strategy.as_ref(),
                                &EncodingOptions::default(),
                            ))
                            .unwrap();
                        b.iter(|| {
                            let batch = rt
                                .block_on(lance_encoding::decoder::decode_batch(
                                    &encoded,
                                    &FilterExpression::no_filter(),
                                    Arc::<DecoderPlugins>::default(),
                                    false,
                                    encoded_batch_layout(encoding),
                                    Some(Arc::new(LanceCache::no_cache())),
                                ))
                                .unwrap();
                            assert_eq!(data.num_rows(), batch.num_rows());
                        })
                    });
                }
            }
        }
    }
}

fn bench_decode_str_with_dict_encoding(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");
    const NUM_ROWS: u64 = 100000;

    let data_type = DataType::Utf8;
    // generate string column with 20 rows
    let string_data = lance_datagen::gen_batch()
        .anon_col(lance_datagen::array::rand_type(&DataType::Utf8))
        .into_batch_rows(lance_datagen::RowCount::from(20))
        .unwrap();

    group.throughput(criterion::Throughput::Bytes(
        NUM_ROWS * std::mem::size_of::<u32>() as u64 + string_data.get_array_memory_size() as u64,
    ));

    let func_name = format!("{:?}", data_type).to_lowercase();
    group.bench_function(func_name, |b| {
        let string_array = string_data.column(0);

        // generate random int column with 100000 rows
        let mut rng = rand::rng();
        let integer_arr: Vec<u32> = (0..100_000).map(|_| rng.random_range(0..20)).collect();
        let integer_array = UInt32Array::from(integer_arr);

        let mapped_strings = take(string_array, &integer_array, None).unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "string",
            DataType::Utf8,
            false,
        )]));

        let data = RecordBatch::try_new(schema, vec![Arc::new(mapped_strings)]).unwrap();

        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap());
        let encoding_strategy = encoding_strategy(BenchEncoding::StructuralU16);
        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();
        b.iter(|| {
            let batch = rt
                .block_on(lance_encoding::decoder::decode_batch(
                    &encoded,
                    &FilterExpression::no_filter(),
                    Arc::<DecoderPlugins>::default(),
                    false,
                    EncodedBatchLayout::Structural,
                    Some(Arc::new(LanceCache::no_cache())),
                ))
                .unwrap();
            assert_eq!(data.num_rows(), batch.num_rows());
        })
    });
}

fn bench_decode_packed_struct(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");

    const NUM_ROWS: u64 = 10000;
    let size_bytes =
        ((6 * std::mem::size_of::<i32>() as u64) + std::mem::size_of::<f32>() as u64) * NUM_ROWS;
    group.throughput(criterion::Throughput::Bytes(size_bytes));

    let func_name = "struct";
    group.bench_function(func_name, |b| {
        let fields = vec![
            Arc::new(Field::new("int_field", DataType::Int32, false)),
            Arc::new(Field::new("float_field", DataType::Float32, false)),
            Arc::new(Field::new(
                "fsl_field",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 5),
                false,
            )),
        ]
        .into();

        // generate struct column with 1M rows
        let data = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::rand_type(&DataType::Struct(fields)))
            .into_batch_rows(lance_datagen::RowCount::from(NUM_ROWS))
            .unwrap();

        let schema = data.schema();
        let new_fields: Vec<Arc<Field>> = schema
            .fields()
            .iter()
            .map(|field| {
                if matches!(field.data_type(), &DataType::Struct(_)) {
                    let mut metadata = HashMap::new();
                    metadata.insert("packed".to_string(), "true".to_string());
                    let field =
                        Field::new(field.name(), field.data_type().clone(), field.is_nullable());
                    Arc::new(field.with_metadata(metadata))
                } else {
                    field.clone()
                }
            })
            .collect();

        let new_schema = Schema::new(new_fields);
        let data =
            RecordBatch::try_new(Arc::new(new_schema.clone()), data.columns().to_vec()).unwrap();

        let lance_schema = Arc::new(lance_core::datatypes::Schema::try_from(&new_schema).unwrap());
        let encoding_strategy = encoding_strategy(BenchEncoding::StructuralU32);
        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();

        b.iter(|| {
            let batch = rt
                .block_on(lance_encoding::decoder::decode_batch(
                    &encoded,
                    &FilterExpression::no_filter(),
                    Arc::<DecoderPlugins>::default(),
                    false,
                    EncodedBatchLayout::Structural,
                    Some(Arc::new(LanceCache::no_cache())),
                ))
                .unwrap();
            assert_eq!(data.num_rows(), batch.num_rows());
        })
    });
}

#[cfg(target_os = "linux")]
fn bench_decode_str_with_fixed_size_binary_encoding(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");

    const NUM_ROWS: u64 = 10000;
    // Randomly generated strings are always 12 characters (at the moment)
    // Plus we need 4 bytes for the offset
    const NUM_BYTES: u64 = NUM_ROWS * 16;
    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));

    let func_name = "fixed-utf8".to_string();
    group.bench_function(func_name, |b| {
        // generate string column with 10k rows
        // Currently the generator generates fixed size strings by default
        // This function will need to be updated once that changes.
        let string_data = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::rand_type(&DataType::Utf8))
            .into_batch_rows(lance_datagen::RowCount::from(10000))
            .unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "string",
            DataType::Utf8,
            false,
        )]));

        let data = RecordBatch::try_new(schema, string_data.columns().to_vec()).unwrap();

        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap());
        let encoding_strategy = encoding_strategy(BenchEncoding::StructuralU16);
        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();
        b.iter(|| {
            let batch = rt
                .block_on(lance_encoding::decoder::decode_batch(
                    &encoded,
                    &FilterExpression::no_filter(),
                    Arc::<DecoderPlugins>::default(),
                    false,
                    EncodedBatchLayout::Structural,
                    Some(Arc::new(LanceCache::no_cache())),
                ))
                .unwrap();
            assert_eq!(data.num_rows(), batch.num_rows());
        })
    });
}

fn bench_decode_compressed(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_compressed");

    const NUM_ROWS: usize = 5_000_000;
    const NUM_COLUMNS: usize = 10;

    // Generate compressible string data - high cardinality but compressible
    // (unique values to avoid dictionary encoding, repeated prefix for compression)
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..NUM_ROWS).map(|i| format!("prefix_that_compresses_well_{}", i)),
    ));

    for compression in ["zstd", "lz4"] {
        let mut metadata = HashMap::new();
        metadata.insert(
            "lance-encoding:compression".to_string(),
            compression.to_string(),
        );
        // Disable dictionary encoding to ensure we hit the compression path
        metadata.insert(
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        );
        // Force miniblock encoding (the path that benefits from compressor caching)
        metadata.insert(
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        );
        let fields: Vec<Field> = (0..NUM_COLUMNS)
            .map(|i| {
                Field::new(format!("s{}", i), DataType::Utf8, false).with_metadata(metadata.clone())
            })
            .collect();
        let columns: Vec<Arc<dyn arrow_array::Array>> =
            (0..NUM_COLUMNS).map(|_| array.clone()).collect();
        let schema = Arc::new(Schema::new(fields));
        let data = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
        // V2_2+ required for general compression
        let encoding_strategy = encoding_strategy(BenchEncoding::StructuralU32);

        // Encode once during setup
        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();

        group.throughput(criterion::Throughput::Elements(
            (NUM_ROWS * NUM_COLUMNS) as u64,
        ));
        group.bench_function(
            format!("{}_strings_{}cols", compression, NUM_COLUMNS),
            |b| {
                b.iter(|| {
                    let batch = rt
                        .block_on(lance_encoding::decoder::decode_batch(
                            &encoded,
                            &FilterExpression::no_filter(),
                            Arc::<DecoderPlugins>::default(),
                            false,
                            EncodedBatchLayout::Structural,
                            Some(Arc::new(LanceCache::no_cache())),
                        ))
                        .unwrap();
                    assert_eq!(data.num_rows(), batch.num_rows());
                })
            },
        );
    }
}

/// Benchmark parallel decoding with multiple concurrent batch decode tasks.
/// This creates contention on the shared decompressor mutex when multiple
/// batches from the same page are decoded in parallel.
fn bench_decode_compressed_parallel(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_compressed_parallel");

    const NUM_ROWS: u64 = 1_000_000;
    const NUM_COLUMNS: usize = 10;
    // Small batch size to create many batches that will contend on the same decompressor
    const BATCH_SIZE: u32 = 100_000;

    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..NUM_ROWS as usize).map(|i| format!("prefix_that_compresses_well_{}", i)),
    ));

    for compression in ["zstd", "lz4"] {
        let mut metadata = HashMap::new();
        metadata.insert(
            "lance-encoding:compression".to_string(),
            compression.to_string(),
        );
        metadata.insert(
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        );
        metadata.insert(
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        );
        let fields: Vec<Field> = (0..NUM_COLUMNS)
            .map(|i| {
                Field::new(format!("s{}", i), DataType::Utf8, false).with_metadata(metadata.clone())
            })
            .collect();
        let columns: Vec<Arc<dyn arrow_array::Array>> =
            (0..NUM_COLUMNS).map(|_| array.clone()).collect();
        let schema = Arc::new(Schema::new(fields));
        let data = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
        let encoding_strategy = encoding_strategy(BenchEncoding::StructuralU32);

        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();

        let encoded = Arc::new(encoded);

        // Test with different parallelism levels to see impact of mutex contention
        // parallelism=1 is sequential (no contention), higher values cause contention
        for parallelism in [1, 8] {
            group.throughput(criterion::Throughput::Elements(
                NUM_ROWS * NUM_COLUMNS as u64,
            ));
            group.bench_function(
                format!(
                    "{}_{}cols_parallel_{}",
                    compression, NUM_COLUMNS, parallelism
                ),
                |b| {
                    b.iter(|| {
                        rt.block_on(async {
                            let io_scheduler = Arc::new(lance_encoding::BufferScheduler::new(
                                encoded.data.clone(),
                            ))
                                as Arc<dyn lance_encoding::EncodingsIo>;
                            let cache = Arc::new(LanceCache::no_cache());
                            let filter = FilterExpression::no_filter();

                            let mut decode_scheduler = DecodeBatchScheduler::try_new(
                                encoded.schema.as_ref(),
                                &encoded.top_level_columns,
                                &encoded.page_table,
                                &vec![],
                                encoded.num_rows,
                                Arc::<DecoderPlugins>::default(),
                                io_scheduler.clone(),
                                cache,
                                &filter,
                                &DecoderConfig::default(),
                            )
                            .await
                            .unwrap();

                            let (tx, rx) = unbounded_channel();
                            decode_scheduler.schedule_range(
                                0..encoded.num_rows,
                                &filter,
                                tx,
                                io_scheduler,
                            );

                            let decode_stream = create_decode_stream(
                                &encoded.schema,
                                encoded.num_rows,
                                BATCH_SIZE,
                                true, // is_structural for V2_2
                                false,
                                false,
                                rx,
                                None,
                            )
                            .unwrap();

                            // Buffer multiple batch decodes in parallel - this causes contention
                            let batches: Vec<_> = decode_stream
                                .map(|task| task.task)
                                .buffered(parallelism)
                                .collect()
                                .await;

                            let total_rows: usize =
                                batches.iter().map(|b| b.as_ref().unwrap().num_rows()).sum();
                            assert_eq!(total_rows, NUM_ROWS as usize);
                        })
                    })
                },
            );
        }
    }
}

#[cfg(feature = "bitpacking")]
fn make_inline_bitpacking_chunk<T>(bit_width: usize) -> LanceBuffer
where
    T: ArrowNativeType + BitPacking + Pod,
{
    let value_range = 1_usize << bit_width;
    let values: Vec<T> = (0..ELEMS_PER_CHUNK as usize)
        .map(|i| T::from_usize((i * 31 + 7) % value_range).unwrap())
        .collect();
    let packed_words = ELEMS_PER_CHUNK as usize * bit_width / (std::mem::size_of::<T>() * 8);

    let mut chunk = Vec::with_capacity(1 + packed_words);
    chunk.push(T::from_usize(bit_width).unwrap());
    let payload_start = chunk.len();
    chunk.resize(payload_start + packed_words, T::from_usize(0).unwrap());
    unsafe {
        BitPacking::unchecked_pack(bit_width, &values, &mut chunk[payload_start..]);
    }

    LanceBuffer::reinterpret_vec(chunk)
}

#[cfg(feature = "bitpacking")]
fn read_little_endian_header<T>(bytes: &[u8]) -> usize {
    bytes[..std::mem::size_of::<T>()]
        .iter()
        .enumerate()
        .fold(0_u64, |value, (idx, byte)| {
            value | ((*byte as u64) << (idx * 8))
        }) as usize
}

#[cfg(feature = "bitpacking")]
fn legacy_copy_unchunk<T>(data: LanceBuffer, num_values: u64) -> DataBlock
where
    T: ArrowNativeType + BitPacking + Pod,
{
    assert!(data.len() >= std::mem::size_of::<T>());
    assert!(num_values <= ELEMS_PER_CHUNK);

    let chunk_in_u8 = data.to_vec();
    let bit_width_value = read_little_endian_header::<T>(&chunk_in_u8);
    let chunk = bytemuck::cast_slice(&chunk_in_u8[std::mem::size_of::<T>()..]);
    assert!(std::mem::size_of_val(chunk) == bit_width_value * ELEMS_PER_CHUNK as usize / 8);

    let mut decompressed = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];
    unsafe {
        BitPacking::unchecked_unpack(bit_width_value, chunk, &mut decompressed);
    }

    decompressed.truncate(num_values as usize);
    DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(decompressed),
        bits_per_value: (std::mem::size_of::<T>() * 8) as u64,
        num_values,
        block_info: BlockInfo::new(),
    })
}

#[cfg(feature = "bitpacking")]
fn typed_view_unchunk(buffer: LanceBuffer, uncompressed_bits: u64, num_values: u64) -> DataBlock {
    InlineBitpacking::new(uncompressed_bits)
        .decompress(buffer, num_values)
        .unwrap()
}

#[cfg(feature = "bitpacking")]
fn assert_same_fixed_width_payloads(legacy: &DataBlock, typed_view: &DataBlock) {
    let legacy = legacy.as_fixed_width_ref().unwrap();
    let typed_view = typed_view.as_fixed_width_ref().unwrap();

    assert_eq!(legacy.num_values, typed_view.num_values);
    assert_eq!(legacy.bits_per_value, typed_view.bits_per_value);
    assert_eq!(legacy.data.as_ref(), typed_view.data.as_ref());
}

#[cfg(feature = "bitpacking")]
fn bench_inline_bitpacking_case<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    bit_width: usize,
) where
    T: ArrowNativeType + BitPacking + Pod,
{
    let buffer = make_inline_bitpacking_chunk::<T>(bit_width);
    let compressed_bytes = buffer.len() as u64;
    let uncompressed_bits = (std::mem::size_of::<T>() * 8) as u64;
    group.throughput(criterion::Throughput::Bytes(compressed_bytes));

    let legacy = legacy_copy_unchunk::<T>(buffer.clone(), ELEMS_PER_CHUNK);
    let typed_view = typed_view_unchunk(buffer.clone(), uncompressed_bits, ELEMS_PER_CHUNK);
    assert_same_fixed_width_payloads(&legacy, &typed_view);

    group.bench_function(format!("{name}/legacy_copy/compressed_bytes"), |b| {
        b.iter(|| {
            let decoded =
                legacy_copy_unchunk::<T>(black_box(buffer.clone()), black_box(ELEMS_PER_CHUNK));
            let fixed = decoded.as_fixed_width().unwrap();
            black_box(fixed.data.as_ref());
        })
    });

    group.bench_function(format!("{name}/typed_view/compressed_bytes"), |b| {
        b.iter(|| {
            let decoded = typed_view_unchunk(
                black_box(buffer.clone()),
                black_box(uncompressed_bits),
                black_box(ELEMS_PER_CHUNK),
            );
            let fixed = decoded.as_fixed_width().unwrap();
            black_box(fixed.data.as_ref());
        })
    });
}

#[cfg(feature = "bitpacking")]
fn bench_decode_inline_bitpacking_unchunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_inline_bitpacking_unchunk");
    bench_inline_bitpacking_case::<u32>(&mut group, "u32_bw12_1024", 12);
    bench_inline_bitpacking_case::<u64>(&mut group, "u64_bw23_1024", 23);
    group.finish();
}

const PAGE_INIT_BENCH_ROWS: usize = 32 * 1024;
const PAGE_INIT_BENCH_PAYLOAD_BYTES: usize = 512;
const PAGE_INIT_BENCH_RANDOM_ROWS: usize = 512;

fn page_init_bench_data() -> RecordBatch {
    let payload = BinaryArray::from_iter_values((0..PAGE_INIT_BENCH_ROWS).map(|row| {
        let bytes = row.to_le_bytes();
        bytes
            .into_iter()
            .cycle()
            .take(PAGE_INIT_BENCH_PAYLOAD_BYTES)
            .collect::<Vec<_>>()
    }));
    RecordBatch::try_from_iter([("payload", Arc::new(payload) as _)])
        .expect("failed to create page initialization benchmark batch")
}

async fn page_init_bench_encoded() -> lance_encoding::encoder::EncodedBatch {
    let batch = page_init_bench_data();
    let schema = Arc::new(
        lance_core::datatypes::Schema::try_from(batch.schema().as_ref())
            .expect("failed to convert benchmark schema"),
    );
    let options = EncodingOptions {
        cache_bytes_per_column: 64 * 1024,
        max_page_bytes: 64 * 1024,
        ..Default::default()
    };
    let strategy = encoding_strategy(BenchEncoding::StructuralU16);
    let mut batch_encoder = BatchEncoder::try_new(schema.as_ref(), strategy.as_ref(), &options)
        .expect("failed to create benchmark encoder");
    let mut encoder = batch_encoder.field_encoders.remove(0);
    let mut pages = Vec::new();
    let mut external_buffers = OutOfLineBuffers::new(0, options.buffer_alignment);
    for start in (0..PAGE_INIT_BENCH_ROWS).step_by(128) {
        let len = 128.min(PAGE_INIT_BENCH_ROWS - start);
        let tasks = encoder
            .maybe_encode(
                batch.column(0).slice(start, len),
                &mut external_buffers,
                RepDefBuilder::default(),
                start as u64,
                len as u64,
            )
            .expect("failed to buffer benchmark page");
        assert!(
            external_buffers.take_buffers().is_empty(),
            "benchmark data should not use external buffers"
        );
        external_buffers = OutOfLineBuffers::new(0, options.buffer_alignment);
        for task in tasks {
            pages.push(task.await.expect("failed to encode benchmark page"));
        }
    }
    for task in encoder
        .flush(&mut external_buffers)
        .expect("failed to flush benchmark pages")
    {
        pages.push(task.await.expect("failed to encode flushed page"));
    }
    assert!(
        external_buffers.take_buffers().is_empty(),
        "benchmark data should not use external buffers"
    );

    let mut column_buffers = OutOfLineBuffers::new(0, options.buffer_alignment);
    let mut columns = encoder
        .finish(&mut column_buffers)
        .await
        .expect("failed to finish benchmark encoder");
    assert!(column_buffers.take_buffers().is_empty());
    assert_eq!(columns.len(), 1);
    let column = columns.remove(0);
    assert!(column.column_buffers.is_empty());
    pages.extend(column.final_pages);

    let mut encoded_data = BytesMut::new();
    let page_infos = pages
        .into_iter()
        .map(|page| append_benchmark_page(page, &mut encoded_data))
        .collect::<Vec<_>>();
    EncodedBatch {
        data: encoded_data.freeze(),
        page_table: vec![Arc::new(ColumnInfo::new(
            0,
            page_infos.into(),
            Vec::new(),
            column.encoding,
        ))],
        schema,
        top_level_columns: vec![0],
        num_rows: PAGE_INIT_BENCH_ROWS as u64,
    }
}

fn append_benchmark_page(page: EncodedPage, data: &mut BytesMut) -> PageInfo {
    let mut buffers = Vec::with_capacity(page.data.len());
    for buffer in page.data {
        let start = data.len() as u64;
        data.extend_from_slice(&buffer);
        buffers.push((start, data.len() as u64 - start));
    }
    PageInfo {
        num_rows: page.num_rows,
        priority: page.row_number,
        encoding: page.description,
        buffer_offsets_and_sizes: buffers.into(),
    }
}

async fn run_page_init_read(
    encoded: &lance_encoding::encoder::EncodedBatch,
    ranges: &[Range<u64>],
    selective: bool,
    cache_repetition_index: bool,
) {
    let io = Arc::new(lance_encoding::BufferScheduler::new(encoded.data.clone()))
        as Arc<dyn lance_encoding::EncodingsIo>;
    let cache = Arc::new(LanceCache::no_cache());
    let filter = FilterExpression::no_filter();
    let config = DecoderConfig {
        cache_repetition_index,
        ..Default::default()
    };
    let file_buffers = Vec::new();
    let covers_all_rows =
        ranges.len() == 1 && ranges[0].start == 0 && ranges[0].end == encoded.num_rows;
    let mut scheduler = if selective && !covers_all_rows {
        DecodeBatchScheduler::try_new_with_ranges(
            encoded.schema.as_ref(),
            &encoded.top_level_columns,
            &encoded.page_table,
            &file_buffers,
            encoded.num_rows,
            Arc::<DecoderPlugins>::default(),
            io.clone(),
            cache,
            ranges,
            &filter,
            &config,
        )
        .await
    } else {
        DecodeBatchScheduler::try_new(
            encoded.schema.as_ref(),
            &encoded.top_level_columns,
            &encoded.page_table,
            &file_buffers,
            encoded.num_rows,
            Arc::<DecoderPlugins>::default(),
            io.clone(),
            cache,
            &filter,
            &config,
        )
        .await
    }
    .expect("failed to initialize benchmark decoder");

    let requested_rows = ranges.iter().map(|range| range.end - range.start).sum();
    let (tx, rx) = unbounded_channel();
    scheduler.schedule_ranges(ranges, &filter, tx, io);
    let mut stream = create_decode_stream(
        encoded.schema.as_ref(),
        requested_rows,
        requested_rows.max(1) as u32,
        true,
        false,
        false,
        rx,
        None,
    )
    .expect("failed to create benchmark decode stream");
    let mut decoded_rows = 0;
    while let Some(task) = stream.next().await {
        decoded_rows += task
            .task
            .await
            .expect("failed to decode benchmark batch")
            .num_rows();
    }
    assert_eq!(decoded_rows as u64, requested_rows);
}

fn bench_selective_page_initialization(_c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("failed to create benchmark runtime");
    let encoded = Arc::new(runtime.block_on(page_init_bench_encoded()));
    let page_count = encoded.page_table[0].page_infos.len();
    assert!(page_count > 100, "benchmark requires many pages");
    assert!(encoded.page_table[0].page_infos.iter().all(|page| matches!(
        &page.encoding,
        lance_encoding::decoder::PageEncoding::Structural(pb21::PageLayout {
            layout: Some(pb21::page_layout::Layout::FullZipLayout(_))
        })
    )));

    let mut rng = rand::rng();
    let random_rows = Arc::new(
        (0..PAGE_INIT_BENCH_RANDOM_ROWS)
            .map(|_| rng.random_range(0..PAGE_INIT_BENCH_ROWS as u64))
            .collect::<Vec<_>>(),
    );

    report_paired_page_init_measurements(&runtime, encoded.as_ref(), random_rows.as_ref());
}

fn report_paired_page_init_measurements(
    runtime: &tokio::runtime::Runtime,
    encoded: &EncodedBatch,
    random_rows: &[u64],
) {
    runtime.block_on(async {
        for row in random_rows.iter().take(16) {
            let range = [*row..*row + 1];
            run_page_init_read(encoded, &range, false, false).await;
            run_page_init_read(encoded, &range, true, false).await;
        }
        let full_range = [0..encoded.num_rows];
        for _ in 0..2 {
            run_page_init_read(encoded, &full_range, false, false).await;
            run_page_init_read(encoded, &full_range, true, false).await;
        }
    });

    for cache_repetition_index in [false, true] {
        for round in 0..5 {
            let (eager, selective) = runtime.block_on(measure_paired_random_reads(
                encoded,
                random_rows,
                cache_repetition_index,
                round * 400,
                400,
            ));
            print_paired_result(
                "random",
                cache_repetition_index,
                round,
                400,
                eager,
                selective,
            );
        }
    }

    let full_range = [0..encoded.num_rows];
    for round in 0..5 {
        let (eager, selective) = runtime.block_on(measure_paired_reads(
            encoded,
            &full_range,
            false,
            10,
            round % 2 == 0,
        ));
        print_paired_result("full_scan", false, round, 10, eager, selective);
    }
}

async fn measure_paired_random_reads(
    encoded: &EncodedBatch,
    random_rows: &[u64],
    cache_repetition_index: bool,
    start: usize,
    count: usize,
) -> (Duration, Duration) {
    let mut eager = Duration::ZERO;
    let mut selective = Duration::ZERO;
    for offset in 0..count {
        let row = random_rows[(start + offset) % random_rows.len()];
        let ranges = [row..row + 1];
        let (eager_elapsed, selective_elapsed) =
            measure_paired_reads(encoded, &ranges, cache_repetition_index, 1, offset % 2 == 0)
                .await;
        eager += eager_elapsed;
        selective += selective_elapsed;
    }
    (eager, selective)
}

async fn measure_paired_reads(
    encoded: &EncodedBatch,
    ranges: &[Range<u64>],
    cache_repetition_index: bool,
    count: usize,
    eager_first: bool,
) -> (Duration, Duration) {
    let mut eager = Duration::ZERO;
    let mut selective = Duration::ZERO;
    for iteration in 0..count {
        let this_eager_first = if iteration % 2 == 0 {
            eager_first
        } else {
            !eager_first
        };
        if this_eager_first {
            let started = Instant::now();
            run_page_init_read(encoded, ranges, false, cache_repetition_index).await;
            eager += started.elapsed();
            let started = Instant::now();
            run_page_init_read(encoded, ranges, true, cache_repetition_index).await;
            selective += started.elapsed();
        } else {
            let started = Instant::now();
            run_page_init_read(encoded, ranges, true, cache_repetition_index).await;
            selective += started.elapsed();
            let started = Instant::now();
            run_page_init_read(encoded, ranges, false, cache_repetition_index).await;
            eager += started.elapsed();
        }
    }
    (eager, selective)
}

fn print_paired_result(
    workload: &str,
    cache_repetition_index: bool,
    round: usize,
    count: usize,
    eager: Duration,
    selective: Duration,
) {
    let eager_us = eager.as_secs_f64() * 1_000_000.0 / count as f64;
    let selective_us = selective.as_secs_f64() * 1_000_000.0 / count as f64;
    let change = (selective_us / eager_us - 1.0) * 100.0;
    println!(
        "PAIRED workload={workload} cache_repetition_index={cache_repetition_index} round={round} eager_us={eager_us:.3} selective_us={selective_us:.3} change_pct={change:.2}"
    );
}

#[cfg(not(feature = "bitpacking"))]
fn bench_decode_inline_bitpacking_unchunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_inline_bitpacking_unchunk");
    group.bench_function("bitpacking_feature_disabled", |b| b.iter(|| black_box(())));
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(lance_testing::pprof::PProfProfiler::new(100, lance_testing::pprof::Output::Flamegraph(None)));
    targets = bench_decode, bench_decode_fsl, bench_decode_str_with_dict_encoding, bench_decode_packed_struct,
                bench_decode_str_with_fixed_size_binary_encoding, bench_decode_compressed,
                bench_decode_compressed_parallel, bench_decode_inline_bitpacking_unchunk,
                bench_selective_page_initialization);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_decode, bench_decode_fsl, bench_decode_str_with_dict_encoding, bench_decode_packed_struct,
                bench_decode_compressed, bench_decode_compressed_parallel, bench_decode_inline_bitpacking_unchunk,
                bench_selective_page_initialization);
criterion_main!(benches);
