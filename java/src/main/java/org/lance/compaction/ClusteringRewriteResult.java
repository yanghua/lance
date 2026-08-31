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
package org.lance.compaction;

import org.lance.FragmentMetadata;

import java.util.List;

/**
 * A clustering rewrite result carrying the core's tagged provenance payload.
 *
 * <p>The distinct class descriptor makes rolling-version deserialization fail closed on clients
 * that do not understand clustering compaction.
 */
public final class ClusteringRewriteResult extends RewriteResult {
  private static final long serialVersionUID = 1L;

  private final byte[] clusteringResultPayload;

  public ClusteringRewriteResult(
      CompactionMetrics metrics,
      List<FragmentMetadata> newFragments,
      List<FragmentMetadata> originalFragments,
      long readVersion,
      byte[] rowAddrs,
      byte[] clusteringResultPayload) {
    super(metrics, newFragments, originalFragments, readVersion, rowAddrs);
    if (clusteringResultPayload == null) {
      throw new IllegalArgumentException("clusteringResultPayload cannot be null");
    }
    this.clusteringResultPayload = clusteringResultPayload;
  }

  public byte[] getClusteringResultPayload() {
    return clusteringResultPayload;
  }
}
