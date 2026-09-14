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
package org.lance.clustering;

import org.lance.Dataset;
import org.lance.JniLoader;
import org.lance.LockManager;
import org.lance.compaction.CompactionMetrics;
import org.lance.compaction.CompactionOptions;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.List;
import java.util.UUID;

/** Rust-backed primitives used to coordinate distributed liquid clustering. */
public final class Clustering {
  static {
    JniLoader.ensureLoaded();
  }

  private Clustering() {}

  /** Plan source-fragment groups against the dataset's current clustering declaration. */
  public static ReclusterPlan planRecluster(Dataset dataset, CompactionOptions options) {
    Preconditions.checkNotNull(dataset, "dataset must not be null");
    Preconditions.checkNotNull(options, "options must not be null");
    try (LockManager.ReadLock ignored = dataset.acquireReadLock()) {
      return nativePlanRecluster(dataset, options);
    }
  }

  /** Build a mergeable partial model from stable physical row addresses. */
  public static byte[] buildPartialModel(
      BufferAllocator allocator,
      VectorSchemaRoot root,
      FieldVector rowAddresses,
      ReclusterPlan plan) {
    Preconditions.checkNotNull(allocator, "allocator must not be null");
    Preconditions.checkNotNull(root, "root must not be null");
    Preconditions.checkNotNull(rowAddresses, "rowAddresses must not be null");
    Preconditions.checkNotNull(plan, "plan must not be null");
    Preconditions.checkArgument(
        root.getRowCount() == rowAddresses.getValueCount(),
        "root and rowAddresses must contain the same number of rows");
    try (ArrowSchema batchSchema = ArrowSchema.allocateNew(allocator);
        ArrowArray batchArray = ArrowArray.allocateNew(allocator);
        ArrowSchema rowAddressSchema = ArrowSchema.allocateNew(allocator);
        ArrowArray rowAddressArray = ArrowArray.allocateNew(allocator)) {
      Data.exportVectorSchemaRoot(allocator, root, null, batchArray, batchSchema);
      Data.exportVector(allocator, rowAddresses, null, rowAddressArray, rowAddressSchema);
      return nativeBuildPartialModel(
          plan.getPayload(),
          batchArray.memoryAddress(),
          batchSchema.memoryAddress(),
          rowAddressArray.memoryAddress(),
          rowAddressSchema.memoryAddress());
    }
  }

  /**
   * Combine partial model payloads without finalizing their empirical distributions.
   *
   * <p>This operation is associative and may be used as a Spark tree-aggregation combiner.
   */
  public static byte[] mergePartialModels(List<byte[]> partialModels) {
    Preconditions.checkNotNull(partialModels, "partialModels must not be null");
    return nativeMergePartialModels(partialModels);
  }

  /** Build an order-independent digest for row addresses carried through one output task. */
  public static byte[] digestRowAddresses(BufferAllocator allocator, FieldVector rowAddresses) {
    Preconditions.checkNotNull(allocator, "allocator must not be null");
    Preconditions.checkNotNull(rowAddresses, "rowAddresses must not be null");
    try (ArrowSchema schema = ArrowSchema.allocateNew(allocator);
        ArrowArray array = ArrowArray.allocateNew(allocator)) {
      Data.exportVector(allocator, rowAddresses, null, array, schema);
      return nativeDigestRowAddresses(array.memoryAddress(), schema.memoryAddress());
    }
  }

  /** Bind staged worker fragments to one plan group and the exact model used to order them. */
  public static ReclusterResult createResult(
      ReclusterPlan plan,
      UUID groupId,
      ClusteringModel model,
      List<org.lance.FragmentMetadata> newFragments,
      List<byte[]> outputRowDigests) {
    Preconditions.checkNotNull(plan, "plan must not be null");
    Preconditions.checkNotNull(groupId, "groupId must not be null");
    Preconditions.checkNotNull(model, "model must not be null");
    Preconditions.checkNotNull(newFragments, "newFragments must not be null");
    Preconditions.checkNotNull(outputRowDigests, "outputRowDigests must not be null");
    return new ReclusterResult(
        nativeCreateResult(
            plan.getPayload(),
            groupId.toString(),
            model.getPayload(),
            newFragments,
            outputRowDigests));
  }

  /** Validate worker results and atomically commit the completed groups. */
  public static CompactionMetrics commitRecluster(
      Dataset dataset, ReclusterPlan plan, ClusteringModel model, List<ReclusterResult> results) {
    Preconditions.checkNotNull(dataset, "dataset must not be null");
    Preconditions.checkNotNull(plan, "plan must not be null");
    Preconditions.checkNotNull(model, "model must not be null");
    Preconditions.checkNotNull(results, "results must not be null");
    try (LockManager.ReadLock ignored = dataset.acquireReadLock()) {
      return nativeCommitRecluster(dataset, plan.getPayload(), model.getPayload(), results);
    }
  }

  private static native ReclusterPlan nativePlanRecluster(
      Dataset dataset, CompactionOptions options);

  private static native byte[] nativeBuildPartialModel(
      byte[] plan,
      long batchArrayAddress,
      long batchSchemaAddress,
      long rowAddressArrayAddress,
      long rowAddressSchemaAddress);

  private static native byte[] nativeMergePartialModels(List<byte[]> partialModels);

  private static native byte[] nativeDigestRowAddresses(
      long rowAddressArray, long rowAddressSchema);

  private static native byte[] nativeCreateResult(
      byte[] plan,
      String groupId,
      byte[] model,
      List<org.lance.FragmentMetadata> newFragments,
      List<byte[]> outputRowDigests);

  private static native CompactionMetrics nativeCommitRecluster(
      Dataset dataset, byte[] plan, byte[] model, List<ReclusterResult> results);
}
