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

import java.io.Serializable;
import java.util.List;

/**
 * Opaque, versioned plan produced by the Rust clustering planner.
 *
 * <p>Instances are serializable so a Spark driver can distribute the read version, clustering
 * declaration, and group identities without reconstructing trusted metadata in Java.
 */
public final class ReclusterPlan implements Serializable {
  private static final long serialVersionUID = 1L;

  private final byte[] payload;
  private final long readVersion;
  private final long clusteringGeneration;
  private final List<String> columns;
  private final List<ReclusterGroup> groups;

  ReclusterPlan(
      byte[] payload,
      long readVersion,
      long clusteringGeneration,
      List<String> columns,
      List<ReclusterGroup> groups) {
    this.payload = payload.clone();
    this.readVersion = readVersion;
    this.clusteringGeneration = clusteringGeneration;
    this.columns = List.copyOf(columns);
    this.groups = List.copyOf(groups);
  }

  byte[] getPayload() {
    return payload.clone();
  }

  public long getReadVersion() {
    return readVersion;
  }

  public long getClusteringGeneration() {
    return clusteringGeneration;
  }

  /**
   * @deprecated Use {@link #getClusteringGeneration()} instead.
   */
  @Deprecated
  public long getClusteringVersion() {
    return getClusteringGeneration();
  }

  public List<String> getColumns() {
    return columns;
  }

  public List<ReclusterGroup> getGroups() {
    return groups;
  }

  private Object readResolve() {
    return new ReclusterPlan(payload, readVersion, clusteringGeneration, columns, groups);
  }
}
