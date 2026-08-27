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

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Tests for the liquid-clustering declaration API. */
public class ClusteringTest {
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
