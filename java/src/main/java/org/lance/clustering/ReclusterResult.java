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

/**
 * Opaque worker result accepted by the Rust reclustering commit validator.
 *
 * <p>Construct instances with {@link Clustering#createResult}; the payload binds source and output
 * fragments to one plan, group, and clustering model.
 */
public final class ReclusterResult implements Serializable {
  private static final long serialVersionUID = 1L;

  private final byte[] payload;

  ReclusterResult(byte[] payload) {
    this.payload = payload.clone();
  }

  byte[] getPayload() {
    return payload.clone();
  }

  private Object readResolve() {
    return new ReclusterResult(payload);
  }
}
