// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Strongly typed liquid-clustering identifiers.

use lance_core::deepsize::DeepSizeOf;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Identity shared by all fragments produced by one clustering rewrite group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, DeepSizeOf, Serialize, Deserialize)]
pub struct ClusteringGroupId([u8; 16]);

impl ClusteringGroupId {
    /// Whether this is the nil UUID, which is not a valid clustering group.
    pub fn is_nil(self) -> bool {
        self.0 == [0; 16]
    }
}

impl From<Uuid> for ClusteringGroupId {
    fn from(value: Uuid) -> Self {
        Self(value.into_bytes())
    }
}

impl From<ClusteringGroupId> for Uuid {
    fn from(value: ClusteringGroupId) -> Self {
        Self::from_bytes(value.0)
    }
}

impl std::fmt::Display for ClusteringGroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Uuid::from(*self).fmt(f)
    }
}
