use crate::signature::verify_signature_hex;
use paraoxidizer_core::error::Result;
use paraoxidizer_format::PoxFile;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub container_valid: bool,
    pub tensor_integrity_valid: bool,
    pub manifest_valid: bool,
    pub signature_valid: Option<bool>,
    pub signer_pubkey: Option<String>,
    pub source_hash: String,
    pub tensor_count: usize,
    pub details: Vec<String>,
}

impl VerificationReport {
    pub fn is_trusted(&self, require_signature: bool) -> bool {
        let base_ok = self.container_valid && self.tensor_integrity_valid && self.manifest_valid;
        if require_signature {
            base_ok && self.signature_valid == Some(true)
        } else {
            base_ok
        }
    }
}

pub fn verify_pox_file(
    file: &PoxFile,
    trusted_public_key: Option<&str>,
) -> Result<VerificationReport> {
    let mut details = Vec::new();
    let container_valid = file.header.magic == *paraoxidizer_format::POX_MAGIC;
    if container_valid {
        details.push("Container header and magic verified".into());
    }

    // Verify all tensor internal SHA-256 hashes
    let tensor_integrity_valid = match file.verify_integrity() {
        Ok(()) => {
            details.push(format!(
                "All {} tensors verified against cryptographic SHA-256 manifest",
                file.tensors.len()
            ));
            true
        }
        Err(e) => {
            details.push(format!("Tensor integrity verification failed: {e}"));
            false
        }
    };

    let manifest_valid = !file.manifest.artifact_sha256.is_empty();
    if manifest_valid {
        details.push(format!(
            "Artifact root hash present: {}",
            file.manifest.artifact_sha256
        ));
    }

    // Verify Ed25519 signature if present
    let (signature_valid, signer_pubkey) = if let Some(sig) = &file.signature {
        let pubkey = trusted_public_key.unwrap_or(&sig.public_key_hex);
        let artifact_hash_bytes = file.manifest.artifact_sha256.as_bytes();
        match verify_signature_hex(artifact_hash_bytes, pubkey, &sig.signature_hex) {
            Ok(true) => {
                details.push(format!(
                    "Ed25519 signature verified for public key {}",
                    &pubkey[..pubkey.len().min(16)]
                ));
                (Some(true), Some(pubkey.to_string()))
            }
            Ok(false) => {
                details.push("Ed25519 signature is INVALID".into());
                (Some(false), Some(pubkey.to_string()))
            }
            Err(e) => {
                details.push(format!("Signature verification error: {e}"));
                (Some(false), None)
            }
        }
    } else {
        details.push("No digital signature block in artifact".into());
        (None, None)
    };

    Ok(VerificationReport {
        container_valid,
        tensor_integrity_valid,
        manifest_valid,
        signature_valid,
        signer_pubkey,
        source_hash: file.manifest.source_model_sha256.clone(),
        tensor_count: file.tensors.len(),
        details,
    })
}
