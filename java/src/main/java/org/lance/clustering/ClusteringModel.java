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

import org.lance.JniLoader;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.io.Serializable;
import java.util.List;

/** Opaque clustering model shared by all workers for one reclustering plan. */
public final class ClusteringModel implements Serializable {
  private static final long serialVersionUID = 1L;

  static {
    JniLoader.ensureLoaded();
  }

  private final byte[] payload;

  ClusteringModel(byte[] payload) {
    this.payload = payload.clone();
  }

  byte[] getPayload() {
    return payload.clone();
  }

  /** Finalize deterministic partial models produced by independent workers. */
  public static ClusteringModel merge(List<byte[]> partialModels) {
    Preconditions.checkNotNull(partialModels, "partialModels must not be null");
    return new ClusteringModel(nativeMerge(partialModels));
  }

  /** Encode the clustering key for every row in {@code root}. */
  public ArrowReader encode(BufferAllocator allocator, VectorSchemaRoot root) {
    Preconditions.checkNotNull(allocator, "allocator must not be null");
    Preconditions.checkNotNull(root, "root must not be null");
    try (ArrowSchema schema = ArrowSchema.allocateNew(allocator);
        ArrowArray array = ArrowArray.allocateNew(allocator);
        ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
      Data.exportVectorSchemaRoot(allocator, root, null, array, schema);
      nativeEncode(payload, array.memoryAddress(), schema.memoryAddress(), stream.memoryAddress());
      return Data.importArrayStream(allocator, stream);
    }
  }

  private static native byte[] nativeMerge(List<byte[]> partialModels);

  private static native void nativeEncode(
      byte[] model, long arrowArrayAddress, long arrowSchemaAddress, long streamAddress);

  private Object readResolve() {
    return new ClusteringModel(payload);
  }
}
