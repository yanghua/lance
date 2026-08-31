/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance;

import org.lance.clustering.ClusteringCurve;
import org.lance.clustering.ClusteringSpec;
import org.lance.operation.Append;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.math.BigInteger;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Tests for the liquid-clustering declaration API. */
public class ClusteringTest {
  private static final BigInteger ABOVE_LONG_MAX =
      BigInteger.valueOf(Long.MAX_VALUE).add(BigInteger.ONE);

  @Test
  void testClusteringSpecRejectsInvalidParameters() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new ClusteringSpec(Arrays.asList("id"), ClusteringCurve.HILBERT, 0, 16));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ClusteringSpec(Arrays.asList("id"), ClusteringCurve.HILBERT, -1, 16));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ClusteringSpec(Arrays.asList("id"), ClusteringCurve.HILBERT, 1, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ClusteringSpec(Arrays.asList("id"), ClusteringCurve.HILBERT, 1, 65));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new ClusteringSpec(
                Arrays.asList("id", "name", "other"), ClusteringCurve.HILBERT, 1, 64));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ClusteringSpec(Arrays.asList("id", "id"), ClusteringCurve.HILBERT, 1, 16));
    assertThrows(
        NullPointerException.class,
        () -> new ClusteringSpec(Arrays.asList("id", null), ClusteringCurve.HILBERT, 1, 16));
  }

  @Test
  void testWriteParamsRejectsEmptyClusterBy() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new WriteParams.Builder().withClusterBy(Collections.emptyList()));
  }

  @Test
  void testWriteDatasetBuilderUriForwardsClusterBy(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("clustered_builder_write").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        VectorSchemaRoot root =
            VectorSchemaRoot.create(
                new TestUtils.SimpleTestDataset(allocator, datasetPath).getSchema(), allocator);
        ArrowReader reader = new SingleBatchReader(allocator, root)) {
      root.allocateNew();
      ((IntVector) root.getVector("id")).setSafe(0, 1);
      ((VarCharVector) root.getVector("name")).setSafe(0, new byte[] {'a'});
      root.setRowCount(1);

      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  Dataset.write()
                      .allocator(allocator)
                      .reader(reader)
                      .uri(datasetPath)
                      .clusterBy(Arrays.asList("missing"))
                      .execute());
      assertTrue(error.getMessage().contains("clustering column \"missing\""));
    }
  }

  @Test
  void testWriteDatasetBuilderRejectsClusterByForSchemaOnlyDataset() {
    Schema schema =
        new Schema(
            Collections.singletonList(
                new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null)));

    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      IllegalStateException error =
          assertThrows(
              IllegalStateException.class,
              () ->
                  new WriteDatasetBuilder()
                      .allocator(allocator)
                      .schema(schema)
                      .uri("unused")
                      .clusterBy(Collections.singletonList("id"))
                      .execute());

      assertEquals(
          "clusterBy() cannot be used with schema-only dataset creation because there are no rows "
              + "to cluster. Provide data via reader() or stream().",
          error.getMessage());
    }
  }

  @Test
  @SuppressWarnings("deprecation")
  void testDeprecatedSchemaOnlyCreateRejectsClusterBy(@TempDir Path tempDir) {
    Schema schema =
        new Schema(
            Collections.singletonList(
                new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null)));
    WriteParams params =
        new WriteParams.Builder().withClusterBy(Collections.singletonList("id")).build();

    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  Dataset.create(
                      allocator, tempDir.resolve("deprecated").toString(), schema, params));
      assertTrue(error.getMessage().contains("schema-only dataset"));
    }
  }

  @Test
  void testWriteFragmentBuilderRejectsWriteParamsAndClusterByInBothOrders() {
    WriteParams params = new WriteParams.Builder().build();
    assertThrows(
        IllegalStateException.class,
        () -> Fragment.write().writeParams(params).clusterBy(Collections.singletonList("id")));
    assertThrows(
        IllegalStateException.class,
        () -> Fragment.write().clusterBy(Collections.singletonList("id")).writeParams(params));
  }

  @Test
  void testWriteDatasetInheritsWideDeclaredClusteringSpec(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("wide_clustered_append").toString();
    List<String> columns = Arrays.asList("k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8");
    Schema schema =
        new Schema(
            columns.stream()
                .map(
                    column ->
                        new Field(column, FieldType.nullable(new ArrowType.Int(32, true)), null))
                .collect(Collectors.toList()));

    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        Dataset dataset =
            Dataset.write().allocator(allocator).schema(schema).uri(datasetPath).execute()) {
      ClusteringSpec declared = new ClusteringSpec(columns, ClusteringCurve.ZORDER, 7, 8);
      dataset.setClustering(declared);

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
          ArrowReader reader = new SingleBatchReader(allocator, root)) {
        root.allocateNew();
        for (String column : columns) {
          IntVector vector = (IntVector) root.getVector(column);
          vector.setSafe(0, 3);
          vector.setSafe(1, 1);
          vector.setSafe(2, 2);
        }
        root.setRowCount(3);

        try (Dataset appended =
            Dataset.write()
                .allocator(allocator)
                .reader(reader)
                .uri(datasetPath)
                .mode(WriteParams.WriteMode.APPEND)
                .clusterBy(columns)
                .execute()) {
          assertEquals(declared, appended.getClusteringSpec().orElseThrow(AssertionError::new));
          Fragment last = appended.getFragments().get(appended.getFragments().size() - 1);
          assertEquals(Long.valueOf(7), last.metadata().getClusteringVersion());
        }
      }
    }
  }

  @Test
  void testSetReadAndClearClustering(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("clustering_declare").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("value", FieldType.nullable(new ArrowType.Int(32, true)), null)));
      try (Dataset dataset =
          Dataset.write().allocator(allocator).schema(schema).uri(datasetPath).execute()) {
        assertFalse(dataset.getClusteringSpec().isPresent());

        ClusteringSpec spec =
            new ClusteringSpec(Arrays.asList("id", "value"), ClusteringCurve.ZORDER, 1, 20);
        dataset.setClustering(spec);

        Optional<ClusteringSpec> readBack = dataset.getClusteringSpec();
        assertTrue(readBack.isPresent());
        assertEquals(spec, readBack.get());

        // Any layout change is allowed when the version is bumped.
        dataset.setClustering(
            new ClusteringSpec(Arrays.asList("id"), ClusteringCurve.HILBERT, 2, 32));
        assertEquals(
            new ClusteringSpec(Arrays.asList("id"), ClusteringCurve.HILBERT, 2, 32),
            dataset.getClusteringSpec().orElseThrow(AssertionError::new));
        // Changing the layout without increasing the version is rejected.
        assertThrows(
            RuntimeException.class,
            () ->
                dataset.setClustering(
                    new ClusteringSpec(Arrays.asList("value"), ClusteringCurve.ZORDER, 2, 20)));

        // Clearing removes the complete declaration.
        dataset.clearClustering();
        assertFalse(dataset.getClusteringSpec().isPresent());
      }
    }
  }

  @Test
  void testClusteringVersionAboveLongMaxRoundTrips(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("clustering_u64_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        ClusteringSpec spec =
            new ClusteringSpec(
                Collections.singletonList("id"), ClusteringCurve.HILBERT, ABOVE_LONG_MAX, 16);
        dataset.setClustering(spec);
        assertEquals(
            ABOVE_LONG_MAX,
            dataset.getClusteringSpec().orElseThrow(AssertionError::new).getVersionUnsigned());
        assertThrows(
            ArithmeticException.class,
            () -> dataset.getClusteringSpec().orElseThrow().getVersion());

        FragmentMetadata rawFragment = testDataset.createNewFragment(1);
        FragmentMetadata stampedFragment =
            FragmentMetadata.withClusteringVersionUnsigned(
                rawFragment.getId(),
                rawFragment.getFiles(),
                rawFragment.getPhysicalRows(),
                rawFragment.getDeletionFile(),
                rawFragment.getRowIdMeta(),
                rawFragment.getCreatedAtVersionMeta(),
                rawFragment.getLastUpdatedAtVersionMeta(),
                ABOVE_LONG_MAX);
        try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        Append.builder()
                            .fragments(Collections.singletonList(stampedFragment))
                            .build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(transaction)) {
          assertEquals(
              ABOVE_LONG_MAX,
              committed.getFragments().get(0).metadata().getClusteringVersionUnsigned());
          Transaction readTransaction =
              committed.readTransaction().orElseThrow(AssertionError::new);
          Append append = (Append) readTransaction.operation();
          assertEquals(ABOVE_LONG_MAX, append.fragments().get(0).getClusteringVersionUnsigned());
        }
      }
    }
  }

  @Test
  void testSetClusteringRejectsUnknownColumn(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("clustering_bad_column").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertThrows(
            RuntimeException.class,
            () ->
                dataset.setClustering(
                    new ClusteringSpec(Arrays.asList("missing"), ClusteringCurve.HILBERT)));
        assertFalse(dataset.getClusteringSpec().isPresent());
      }
    }
  }

  private static class SingleBatchReader extends ArrowReader {
    private final VectorSchemaRoot root;
    private boolean batchLoaded;

    private SingleBatchReader(RootAllocator allocator, VectorSchemaRoot root) {
      super(allocator);
      this.root = root;
    }

    @Override
    public boolean loadNextBatch() {
      if (batchLoaded) {
        return false;
      }
      batchLoaded = true;
      return true;
    }

    @Override
    public VectorSchemaRoot getVectorSchemaRoot() {
      return root;
    }

    @Override
    public long bytesRead() {
      return root.getFieldVectors().stream().mapToLong(vector -> vector.getBufferSize()).sum();
    }

    @Override
    protected void closeReadSource() {}

    @Override
    protected org.apache.arrow.vector.types.pojo.Schema readSchema() {
      return root.getSchema();
    }
  }
}
