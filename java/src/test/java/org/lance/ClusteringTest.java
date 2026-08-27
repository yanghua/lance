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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Tests for the liquid-clustering declaration API. */
public class ClusteringTest {

  @Test
  void testSetReadAndClearClustering(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("clustering_declare").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertFalse(dataset.getClusteringSpec().isPresent());

        ClusteringSpec spec =
            new ClusteringSpec(Arrays.asList("id", "name"), ClusteringCurve.ZORDER, 1, 20);
        dataset.setClustering(spec);

        Optional<ClusteringSpec> readBack = dataset.getClusteringSpec();
        assertTrue(readBack.isPresent());
        assertEquals(spec, readBack.get());

        // The column set is immutable, but the version can be bumped.
        dataset.setClustering(
            new ClusteringSpec(Arrays.asList("id", "name"), ClusteringCurve.ZORDER, 2, 20));
        assertEquals(2, dataset.getClusteringSpec().get().getVersion());
        assertThrows(
            RuntimeException.class,
            () ->
                dataset.setClustering(
                    new ClusteringSpec(Arrays.asList("id"), ClusteringCurve.ZORDER, 3, 20)));

        // Clearing drops the tuning config, but the immutable column markers
        // remain, so the spec falls back to the defaults over those columns.
        dataset.clearClustering();
        ClusteringSpec afterClear = dataset.getClusteringSpec().orElseThrow(AssertionError::new);
        assertEquals(Arrays.asList("id", "name"), afterClear.getColumns());
        assertEquals(ClusteringCurve.HILBERT, afterClear.getCurve());
        assertEquals(1, afterClear.getVersion());
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
}
