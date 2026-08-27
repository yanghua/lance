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

/** The space-filling curve used to order clustering-key values. */
public enum ClusteringCurve {
  /** Hilbert curve: better locality, no large jumps. Default. */
  HILBERT("hilbert"),
  /** Z-order (Morton) curve: interleave the bits of each column; cheaper to compute. */
  ZORDER("zorder");

  private final String value;

  ClusteringCurve(String value) {
    this.value = value;
  }

  /** Returns the canonical lowercase name understood by the native layer. */
  public String getValue() {
    return value;
  }

  /**
   * Parses a curve from its canonical lowercase name.
   *
   * @param value one of {@code "hilbert"} or {@code "zorder"}
   * @return the matching curve
   * @throws IllegalArgumentException if the name is not recognized
   */
  public static ClusteringCurve fromValue(String value) {
    for (ClusteringCurve curve : values()) {
      if (curve.value.equals(value)) {
        return curve;
      }
    }
    throw new IllegalArgumentException(
        "unknown clustering curve \"" + value + "\"; expected \"hilbert\" or \"zorder\"");
  }
}
