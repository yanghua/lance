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
import java.util.UUID;

/** One atomic source-fragment replacement in a distributed reclustering plan. */
public final class ReclusterGroup implements Serializable {
  private static final long serialVersionUID = 1L;

  private final UUID id;
  private final List<Long> sourceFragmentIds;
  private final long expectedLiveRows;

  ReclusterGroup(UUID id, List<Long> sourceFragmentIds, long expectedLiveRows) {
    this.id = id;
    this.sourceFragmentIds = List.copyOf(sourceFragmentIds);
    this.expectedLiveRows = expectedLiveRows;
  }

  public UUID getId() {
    return id;
  }

  public List<Long> getSourceFragmentIds() {
    return sourceFragmentIds;
  }

  public long getExpectedLiveRows() {
    return expectedLiveRows;
  }

  private Object readResolve() {
    return new ReclusterGroup(id, sourceFragmentIds, expectedLiveRows);
  }
}
