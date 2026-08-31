// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The transaction itself: an operation plus the version it was based on.

use crate::transaction::Operation;
use lance_core::deepsize::DeepSizeOf;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Reserved internal transport property carrying the clustering layout version
/// claimed for newly written fragments until their final IDs are assigned.
///
/// Commit validates the operation and declaration version, but advanced callers
/// that construct or deserialize [`Transaction`] directly remain responsible
/// for the truth of reserved metadata claims.
pub const NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY: &str =
    "__lance_new_fragment_clustering_version";

/// A change to a dataset that can be retried
///
/// This contains enough information to be able to build the next manifest,
/// given the current manifest.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct Transaction {
    /// The version of the table this transaction is based off of. If this is
    /// the first transaction, this should be 0.
    pub read_version: u64,
    pub uuid: String,
    pub operation: Operation,
    pub tag: Option<String>,
    /// Caller metadata and reserved internal transport properties.
    ///
    /// Direct construction is an advanced, trusted metadata boundary. Reserved
    /// properties are not attestations that can prove how referenced files were
    /// produced. Prefer [`TransactionBuilder`] for ordinary construction.
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
}

/// Add TransactionBuilder for flexibly setting option without using `mut`
pub struct TransactionBuilder {
    read_version: u64,
    // uuid is optional for builder since it can autogenerate
    uuid: Option<String>,
    operation: Operation,
    tag: Option<String>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
    new_fragment_clustering_version: Option<u64>,
}

impl TransactionBuilder {
    pub fn new(read_version: u64, operation: Operation) -> Self {
        Self {
            read_version,
            uuid: None,
            operation,
            tag: None,
            transaction_properties: None,
            new_fragment_clustering_version: None,
        }
    }

    pub fn uuid(mut self, uuid: String) -> Self {
        self.uuid = Some(uuid);
        self
    }

    pub fn tag(mut self, tag: Option<String>) -> Self {
        self.tag = tag;
        self
    }

    pub fn transaction_properties(
        mut self,
        transaction_properties: Option<Arc<HashMap<String, String>>>,
    ) -> Self {
        self.transaction_properties = transaction_properties.map(|properties| {
            let mut properties = properties.as_ref().clone();
            properties.remove(NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY);
            Arc::new(properties)
        });
        self
    }

    /// Carry the clustering declaration version used to produce new fragments.
    ///
    /// This is reserved internal transport metadata. Manifest construction
    /// checks its operation and declaration version, but does not re-read data
    /// files to verify their physical order.
    #[doc(hidden)]
    pub fn new_fragment_clustering_version(mut self, version: u64) -> Self {
        self.new_fragment_clustering_version = Some(version);
        self
    }

    pub fn build(self) -> Transaction {
        let uuid = self
            .uuid
            .unwrap_or_else(|| Uuid::new_v4().hyphenated().to_string());
        let mut transaction_properties = self
            .transaction_properties
            .as_deref()
            .cloned()
            .unwrap_or_default();
        // Keep ordinary caller properties from accidentally shadowing the
        // reserved transport marker, regardless of builder call order.
        transaction_properties.remove(NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY);
        if let Some(version) = self.new_fragment_clustering_version {
            transaction_properties.insert(
                NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY.to_string(),
                version.to_string(),
            );
        }
        let transaction_properties =
            (!transaction_properties.is_empty()).then(|| Arc::new(transaction_properties));
        Transaction {
            read_version: self.read_version,
            uuid,
            operation: self.operation,
            tag: self.tag,
            transaction_properties,
        }
    }
}

impl Transaction {
    /// Return the reserved clustering version marker carried by this transaction.
    ///
    /// This accessor lets language bindings preserve the internal transport
    /// marker while converting an uncommitted transaction.
    #[doc(hidden)]
    pub fn new_fragment_clustering_version(&self) -> lance_core::Result<Option<u64>> {
        let Some(value) = self
            .transaction_properties
            .as_deref()
            .and_then(|properties| properties.get(NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY))
        else {
            return Ok(None);
        };
        let version = value.parse::<u64>().map_err(|error| {
            lance_core::Error::invalid_input(format!(
                "invalid reserved transaction property \"{NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY}\" value {value:?}: expected a positive u64: {error}"
            ))
        })?;
        if version == 0 {
            return Err(lance_core::Error::invalid_input(format!(
                "invalid reserved transaction property \"{NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY}\": version must be positive"
            )));
        }
        Ok(Some(version))
    }

    pub fn new_from_version(read_version: u64, operation: Operation) -> Self {
        TransactionBuilder::new(read_version, operation).build()
    }

    pub fn new(read_version: u64, operation: Operation, tag: Option<String>) -> Self {
        TransactionBuilder::new(read_version, operation)
            .tag(tag)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caller_properties(marker: &str) -> Arc<HashMap<String, String>> {
        Arc::new(HashMap::from([
            ("application.key".to_string(), "value".to_string()),
            (
                NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY.to_string(),
                marker.to_string(),
            ),
        ]))
    }

    #[test]
    fn caller_properties_cannot_inject_reserved_clustering_marker() {
        let transaction = TransactionBuilder::new(0, Operation::Append { fragments: vec![] })
            .transaction_properties(Some(caller_properties("not-a-version")))
            .build();

        assert_eq!(transaction.new_fragment_clustering_version().unwrap(), None);
        let properties = transaction.transaction_properties.as_deref().unwrap();
        assert_eq!(
            properties.get("application.key").map(String::as_str),
            Some("value")
        );
        assert!(!properties.contains_key(NEW_FRAGMENT_CLUSTERING_VERSION_PROPERTY));
    }

    #[test]
    fn internal_clustering_marker_wins_regardless_of_builder_call_order() {
        let build = |properties_first| {
            let builder = TransactionBuilder::new(0, Operation::Append { fragments: vec![] });
            if properties_first {
                builder
                    .transaction_properties(Some(caller_properties("99")))
                    .new_fragment_clustering_version(7)
                    .build()
            } else {
                builder
                    .new_fragment_clustering_version(7)
                    .transaction_properties(Some(caller_properties("99")))
                    .build()
            }
        };

        for transaction in [build(true), build(false)] {
            assert_eq!(
                transaction.new_fragment_clustering_version().unwrap(),
                Some(7)
            );
            assert_eq!(
                transaction
                    .transaction_properties
                    .as_deref()
                    .unwrap()
                    .get("application.key")
                    .map(String::as_str),
                Some("value")
            );
        }
    }
}
