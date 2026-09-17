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

import org.lance.clustering.Clustering;
import org.lance.clustering.ClusteringModel;
import org.lance.clustering.ReclusterPlan;
import org.lance.clustering.ReclusterResult;
import org.lance.compaction.CompactionOptions;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FixedSizeBinaryVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ClusteringTest {
  @Test
  void testDeclareReclusterAndClear(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("clustering").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        assertFalse(dataset.getClusteringColumns().isPresent());
        dataset.setClustering(Arrays.asList("id", "name"));
        assertEquals(Arrays.asList("id", "name"), dataset.getClusteringColumns().orElseThrow());

        dataset.recluster(
            CompactionOptions.builder()
                .withTargetRowsPerFragment(100)
                .withMaxRowsPerGroup(100)
                .withNumThreads(1)
                .build());
        assertEquals(20, dataset.countRows());
        assertEquals(1, dataset.getFragments().size());

        dataset.clearClustering();
        assertFalse(dataset.getClusteringColumns().isPresent());
        assertThrows(RuntimeException.class, () -> dataset.setClustering(Collections.emptyList()));
      }
    }
  }

  @Test
  void testDistributedPrimitivesUseOpaquePlanAndArrowKernels(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("distributed-clustering").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 4)) {
        dataset.setClustering(Collections.singletonList("id"));
        ReclusterPlan plan =
            roundTrip(
                Clustering.planRecluster(
                    dataset, CompactionOptions.builder().withTargetRowsPerFragment(10).build()));
        assertEquals(1, plan.getGroups().size());
        assertTrue(plan.getClusteringGeneration() > 0);
        assertEquals(Collections.singletonList(0L), plan.getGroups().get(0).getSourceFragmentIds());
        assertEquals(4, plan.getGroups().get(0).getExpectedLiveRows());

        Schema schema =
            new Schema(
                Collections.singletonList(Field.notNullable("id", new ArrowType.Int(32, true))));
        try (VectorSchemaRoot left = VectorSchemaRoot.create(schema, allocator);
            VectorSchemaRoot right = VectorSchemaRoot.create(schema, allocator);
            UInt8Vector leftIds = new UInt8Vector("row_address", allocator);
            UInt8Vector rightIds = new UInt8Vector("row_address", allocator)) {
          fillIds(left, new int[] {30, 10});
          fillIds(right, new int[] {40, 20});
          fillRowIds(leftIds, new long[] {0, 1});
          fillRowIds(rightIds, new long[] {2, 3});
          byte[] first = Clustering.buildPartialModel(allocator, left, leftIds, plan);
          byte[] second = Clustering.buildPartialModel(allocator, right, rightIds, plan);
          byte[] merged = Clustering.mergePartialModels(Arrays.asList(first, second));
          ClusteringModel model =
              roundTrip(ClusteringModel.merge(Collections.singletonList(merged)));

          try (ArrowReader encoded = model.encode(allocator, left)) {
            assertEquals(
                "__lance_clustering_key",
                encoded.getVectorSchemaRoot().getSchema().getFields().get(0).getName());
            assertTrue(encoded.loadNextBatch());
            FixedSizeBinaryVector keys =
                (FixedSizeBinaryVector) encoded.getVectorSchemaRoot().getVector(0);
            assertEquals(2, keys.getValueCount());
            assertFalse(Arrays.equals(keys.get(0), keys.get(1)));
            assertFalse(encoded.loadNextBatch());
          }

          List<FragmentMetadata> newFragments =
              Collections.singletonList(testDataset.createNewFragment(4));
          byte[] outputRowDigest = Clustering.digestRowAddresses(allocator, leftIds);
          byte[] secondOutputRowDigest = Clustering.digestRowAddresses(allocator, rightIds);
          ReclusterResult result =
              roundTrip(
                  Clustering.createResult(
                      plan,
                      plan.getGroups().get(0).getId(),
                      model,
                      newFragments,
                      Arrays.asList(outputRowDigest, secondOutputRowDigest)));
          Clustering.commitRecluster(dataset, plan, model, Collections.singletonList(result));
          assertEquals(4, dataset.countRows());
          assertTrue(
              Clustering.planRecluster(dataset, CompactionOptions.builder().build())
                  .getGroups()
                  .isEmpty());
        }
      }
    }
  }

  private static void fillIds(VectorSchemaRoot root, int[] values) {
    root.allocateNew();
    IntVector ids = (IntVector) root.getVector("id");
    for (int index = 0; index < values.length; index++) {
      ids.setSafe(index, values[index]);
    }
    root.setRowCount(values.length);
  }

  private static void fillRowIds(UInt8Vector rowIds, long[] values) {
    rowIds.allocateNew(values.length);
    for (int index = 0; index < values.length; index++) {
      rowIds.setSafe(index, values[index]);
    }
    rowIds.setValueCount(values.length);
  }

  @SuppressWarnings("unchecked")
  private static <T> T roundTrip(T value) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
      output.writeObject(value);
    }
    try (ObjectInputStream input =
        new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
      return (T) input.readObject();
    }
  }
}
