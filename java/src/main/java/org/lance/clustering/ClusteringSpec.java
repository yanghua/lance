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

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;

/**
 * Describes how a dataset is clustered ("liquid clustering").
 *
 * <p>The clustering key lays data out along a multi-column space-filling curve so that zone-map
 * data skipping is effective across every key column at once. The column set is persisted as schema
 * markers and the tuning parameters (curve, version, bits-per-dimension) in the dataset config; see
 * {@code rust/lance-index/src/clustering/mod.rs} for the layout.
 *
 * <p>Mirrors the Rust {@code ClusteringSpec} and the Python {@code clustering_spec} shape.
 */
public class ClusteringSpec implements Serializable {
  private static final long serialVersionUID = 1L;

  /** Default per-column quantization bit width, matching the Rust core. */
  public static final int DEFAULT_BITS_PER_DIM = 16;

  private final List<String> columns;
  private final ClusteringCurve curve;
  private final long version;
  private final int bitsPerDim;

  /**
   * Constructs a clustering spec.
   *
   * @param columns the clustering-key columns, in priority order; must be non-empty
   * @param curve the space-filling curve used to order key values
   * @param version the clustering layout version; bump it to mark existing data under-clustered
   * @param bitsPerDim the per-column quantization bit width; {@code columns.size() * bitsPerDim}
   *     must not exceed 128
   */
  public ClusteringSpec(List<String> columns, ClusteringCurve curve, long version, int bitsPerDim) {
    Objects.requireNonNull(columns, "columns");
    Objects.requireNonNull(curve, "curve");
    if (columns.isEmpty()) {
      throw new IllegalArgumentException("clustering spec must have at least one column");
    }
    this.columns = List.copyOf(columns);
    this.curve = curve;
    this.version = version;
    this.bitsPerDim = bitsPerDim;
  }

  /**
   * Constructs a clustering spec at version 1 with the default bit width.
   *
   * @param columns the clustering-key columns, in priority order
   * @param curve the space-filling curve used to order key values
   */
  public ClusteringSpec(List<String> columns, ClusteringCurve curve) {
    this(columns, curve, 1, DEFAULT_BITS_PER_DIM);
  }

  public List<String> getColumns() {
    return columns;
  }

  public ClusteringCurve getCurve() {
    return curve;
  }

  public long getVersion() {
    return version;
  }

  public int getBitsPerDim() {
    return bitsPerDim;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof ClusteringSpec)) {
      return false;
    }
    ClusteringSpec that = (ClusteringSpec) o;
    return version == that.version
        && bitsPerDim == that.bitsPerDim
        && columns.equals(that.columns)
        && curve == that.curve;
  }

  @Override
  public int hashCode() {
    return Objects.hash(columns, curve, version, bitsPerDim);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("columns", columns)
        .add("curve", curve)
        .add("version", version)
        .add("bitsPerDim", bitsPerDim)
        .toString();
  }
}
