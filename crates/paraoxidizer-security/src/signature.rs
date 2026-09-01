use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use paraoxidizer_core::error::{PoxError, Result};
use rand::rngs::OsRng;

pub struct KeyPair {
    pub signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

impl KeyPair {
    pub fn generate() -> Self {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    pub fn from_private_hex(hex_str: &str) -> Result<Self> {
        let bytes = hex::decode(hex_str.trim())
            .map_err(|e| PoxError::Security(format!("Invalid private key hex: {e}")))?;
        if bytes.len() != 32 {
            return Err(PoxError::Security("Private key must be exactly 32 bytes".into()));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        let signing_key = SigningKey::from_bytes(&arr);
        let verifying_key = signing_key.verifying_key();
        Ok(Self {
            signing_key,
            verifying_key,
        })
    }

    pub fn public_key_hex(&self) -> String {
        hex::encode(self.verifying_key.as_bytes())
    }

    pub fn private_key_hex(&self) -> String {
        hex::encode(self.signing_key.to_bytes())
    }

    pub fn sign_message(&self, message: &[u8]) -> String {
        let signature = self.signing_key.sign(message);
        hex::encode(signature.to_bytes())
    }
}

pub fn verify_signature_hex(
    message: &[u8],
    public_key_hex: &str,
    signature_hex: &str,
) -> Result<bool> {
    let pub_bytes = hex::decode(public_key_hex.trim())
        .map_err(|e| PoxError::Security(format!("Invalid public key hex: {e}")))?;
    if pub_bytes.len() != 32 {
        return Err(PoxError::Security("Public key must be 32 bytes".into()));
    }
    let mut pub_arr = [0u8; 32];
    pub_arr.copy_from_slice(&pub_bytes);
    let verifying_key = VerifyingKey::from_bytes(&pub_arr)
        .map_err(|e| PoxError::Security(format!("Invalid verifying key: {e}")))?;

    let sig_bytes = hex::decode(signature_hex.trim())
        .map_err(|e| PoxError::Security(format!("Invalid signature hex: {e}")))?;
    if sig_bytes.len() != 64 {
        return Err(PoxError::Security("Signature must be 64 bytes".into()));
    }
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);
    let signature = Signature::from_bytes(&sig_arr);

    match verifying_key.verify(message, &signature) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}
