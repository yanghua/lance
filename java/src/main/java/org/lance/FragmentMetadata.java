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

import org.lance.fragment.DataFile;
import org.lance.fragment.DeletionFile;
import org.lance.fragment.RowIdMeta;
import org.lance.fragment.VersionMeta;

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.List;
import java.util.Objects;

/** Metadata of a Fragment in the dataset. Matching to lance Fragment. */
public class FragmentMetadata implements Serializable {
  private static final long serialVersionUID = -5886811251944130460L;
  private static final BigInteger MAX_U64 = new BigInteger("18446744073709551615");
  private final int id;
  private final List<DataFile> files;
  private final long physicalRows;
  private final DeletionFile deletionFile;
  private final RowIdMeta rowIdMeta;
  private final VersionMeta createdAtVersionMeta;
  private final VersionMeta lastUpdatedAtVersionMeta;
  private final BigInteger clusteringVersion;

  public FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta) {
    this(id, files, physicalRows, deletionFile, rowIdMeta, null, null, null);
  }

  public FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta,
      VersionMeta createdAtVersionMeta,
      VersionMeta lastUpdatedAtVersionMeta) {
    this(
        id,
        files,
        physicalRows,
        deletionFile,
        rowIdMeta,
        createdAtVersionMeta,
        lastUpdatedAtVersionMeta,
        (Long) null);
  }

  /** Preserves the existing nullable-Long constructor for source compatibility. */
  public FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta,
      VersionMeta createdAtVersionMeta,
      VersionMeta lastUpdatedAtVersionMeta,
      Long clusteringVersion) {
    if (clusteringVersion != null && clusteringVersion <= 0) {
      throw new IllegalArgumentException(
          "clusteringVersion must be positive, got " + clusteringVersion);
    }
    this.id = id;
    this.files = files;
    this.physicalRows = physicalRows;
    this.deletionFile = deletionFile;
    this.rowIdMeta = rowIdMeta;
    this.createdAtVersionMeta = createdAtVersionMeta;
    this.lastUpdatedAtVersionMeta = lastUpdatedAtVersionMeta;
    this.clusteringVersion =
        clusteringVersion == null ? null : BigInteger.valueOf(clusteringVersion);
  }

  /** Creates metadata with a clustering version over the complete unsigned 64-bit range. */
  public static FragmentMetadata withClusteringVersionUnsigned(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta,
      VersionMeta createdAtVersionMeta,
      VersionMeta lastUpdatedAtVersionMeta,
      BigInteger clusteringVersion) {
    if (clusteringVersion != null
        && (clusteringVersion.signum() <= 0 || clusteringVersion.compareTo(MAX_U64) > 0)) {
      throw new IllegalArgumentException(
          "clusteringVersion must be in 1..=2^64-1, got " + clusteringVersion);
    }
    return new FragmentMetadata(
        id,
        files,
        physicalRows,
        deletionFile,
        rowIdMeta,
        createdAtVersionMeta,
        lastUpdatedAtVersionMeta,
        clusteringVersion,
        true);
  }

  private FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta,
      VersionMeta createdAtVersionMeta,
      VersionMeta lastUpdatedAtVersionMeta,
      BigInteger clusteringVersion,
      @SuppressWarnings("unused") boolean unsignedVersion) {
    this.id = id;
    this.files = files;
    this.physicalRows = physicalRows;
    this.deletionFile = deletionFile;
    this.rowIdMeta = rowIdMeta;
    this.createdAtVersionMeta = createdAtVersionMeta;
    this.lastUpdatedAtVersionMeta = lastUpdatedAtVersionMeta;
    this.clusteringVersion = clusteringVersion;
  }

  public int getId() {
    return id;
  }

  public List<DataFile> getFiles() {
    return files;
  }

  public long getPhysicalRows() {
    return physicalRows;
  }

  public DeletionFile getDeletionFile() {
    return deletionFile;
  }

  public long getNumDeletions() {
    if (deletionFile == null) {
      return 0;
    }
    Long deleted = deletionFile.getNumDeletedRows();
    if (deleted == null) {
      return 0;
    }
    return deleted;
  }

  public long getNumRows() {
    return getPhysicalRows() - getNumDeletions();
  }

  public RowIdMeta getRowIdMeta() {
    return rowIdMeta;
  }

  public VersionMeta getCreatedAtVersionMeta() {
    return createdAtVersionMeta;
  }

  public VersionMeta getLastUpdatedAtVersionMeta() {
    return lastUpdatedAtVersionMeta;
  }

  /**
   * Returns the clustering layout version under which this fragment was written.
   *
   * @throws ArithmeticException if the unsigned 64-bit version exceeds {@link Long#MAX_VALUE}; use
   *     {@link #getClusteringVersionUnsigned()} for the complete range
   * @return the clustering version, or null when the fragment is unstamped
   */
  public Long getClusteringVersion() {
    if (clusteringVersion == null) {
      return null;
    }
    try {
      return clusteringVersion.longValueExact();
    } catch (ArithmeticException error) {
      throw new ArithmeticException(
          "clusteringVersion "
              + clusteringVersion
              + " exceeds Long.MAX_VALUE; use getClusteringVersionUnsigned()");
    }
  }

  /** Returns the clustering version over the complete unsigned 64-bit range, or null if unset. */
  public BigInteger getClusteringVersionUnsigned() {
    return clusteringVersion;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    FragmentMetadata that = (FragmentMetadata) o;
    return id == that.id
        && physicalRows == that.physicalRows
        && Objects.equals(this.files, that.files)
        && Objects.equals(deletionFile, that.deletionFile)
        && Objects.equals(rowIdMeta, that.rowIdMeta)
        && Objects.equals(createdAtVersionMeta, that.createdAtVersionMeta)
        && Objects.equals(lastUpdatedAtVersionMeta, that.lastUpdatedAtVersionMeta)
        && Objects.equals(clusteringVersion, that.clusteringVersion);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        id,
        physicalRows,
        files,
        deletionFile,
        rowIdMeta,
        createdAtVersionMeta,
        lastUpdatedAtVersionMeta,
        clusteringVersion);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("id", id)
        .add("physicalRows", physicalRows)
        .add("files", files)
        .add("deletionFile", deletionFile)
        .add("rowIdMeta", rowIdMeta)
        .add("createdAtVersionMeta", createdAtVersionMeta)
        .add("lastUpdatedAtVersionMeta", lastUpdatedAtVersionMeta)
        .add("clusteringVersion", clusteringVersion)
        .toString();
  }
}
