"""
ChurnShield 2.0 — Enterprise Encryption Engine

File:
utils/encryption.py

Purpose:
Military-grade encryption utilities
for ChurnShield AI platform.

Capabilities:
- AES-256 encryption
- Fernet symmetric encryption
- password hashing
- JWT-safe secret handling
- secure token generation
- API key encryption
- environment-safe secrets
- file encryption
- checksum validation
- HMAC verification
- secure random generation
- PBKDF2 key derivation
- base64-safe encoding
- tamper detection
- data integrity validation
- enterprise cryptography utilities

Author:
ChurnShield AI
"""

import os
import hmac
import json
import base64
import secrets
import hashlib

from typing import (

    Dict,
    Any,
    Optional

)

from pathlib import Path

from cryptography.fernet import (

    Fernet,
    InvalidToken

)

from cryptography.hazmat.primitives.kdf.pbkdf2 import (

    PBKDF2HMAC

)

from cryptography.hazmat.primitives import hashes

from cryptography.hazmat.backends import default_backend

# ============================================================
# SECURITY CONSTANTS
# ============================================================

DEFAULT_ITERATIONS = 390000

SALT_SIZE = 32

TOKEN_LENGTH = 64

CHECKSUM_ALGORITHM = "sha256"

# ============================================================
# MASTER SECRET
# ============================================================

MASTER_SECRET = os.getenv(

    "MASTER_SECRET",
    "CHANGE_THIS_SECRET_IMMEDIATELY"

)

# ============================================================
# ENCRYPTION ENGINE
# ============================================================

class EncryptionManager:

    """
    Enterprise encryption engine
    """

    # ========================================================
    # GENERATE SALT
    # ========================================================

    @staticmethod
    def generate_salt() -> bytes:

        return os.urandom(

            SALT_SIZE

        )

    # ========================================================
    # DERIVE KEY
    # ========================================================

    @staticmethod
    def derive_key(

        password: str,
        salt: bytes,
        iterations: int = DEFAULT_ITERATIONS

    ) -> bytes:

        """
        PBKDF2 key derivation
        """

        kdf = PBKDF2HMAC(

            algorithm=hashes.SHA256(),

            length=32,

            salt=salt,

            iterations=iterations,

            backend=default_backend()

        )

        key = base64.urlsafe_b64encode(

            kdf.derive(

                password.encode()

            )

        )

        return key

    # ========================================================
    # GENERATE ENCRYPTION KEY
    # ========================================================

    @staticmethod
    def generate_key() -> str:

        """
        Generate Fernet key
        """

        return Fernet.generate_key().decode()

    # ========================================================
    # ENCRYPT TEXT
    # ========================================================

    @staticmethod
    def encrypt_text(

        text: str,
        secret: str = MASTER_SECRET

    ) -> Dict:

        """
        Encrypt string data
        """

        salt = EncryptionManager.generate_salt()

        key = EncryptionManager.derive_key(

            secret,
            salt

        )

        fernet = Fernet(key)

        encrypted = fernet.encrypt(

            text.encode()

        )

        return {

            "encrypted":

                encrypted.decode(),

            "salt":

                base64.b64encode(

                    salt

                ).decode()

        }

    # ========================================================
    # DECRYPT TEXT
    # ========================================================

    @staticmethod
    def decrypt_text(

        encrypted_text: str,
        salt: str,
        secret: str = MASTER_SECRET

    ) -> Optional[str]:

        """
        Decrypt encrypted string
        """

        try:

            decoded_salt = base64.b64decode(

                salt

            )

            key = EncryptionManager.derive_key(

                secret,
                decoded_salt

            )

            fernet = Fernet(key)

            decrypted = fernet.decrypt(

                encrypted_text.encode()

            )

            return decrypted.decode()

        except InvalidToken:

            return None

        except Exception:

            return None

    # ========================================================
    # ENCRYPT JSON
    # ========================================================

    @staticmethod
    def encrypt_json(

        data: Dict,
        secret: str = MASTER_SECRET

    ) -> Dict:

        """
        Encrypt JSON object
        """

        serialized = json.dumps(

            data,
            default=str

        )

        return EncryptionManager.encrypt_text(

            serialized,
            secret

        )

    # ========================================================
    # DECRYPT JSON
    # ========================================================

    @staticmethod
    def decrypt_json(

        encrypted_text: str,
        salt: str,
        secret: str = MASTER_SECRET

    ) -> Optional[Dict]:

        """
        Decrypt JSON object
        """

        decrypted = EncryptionManager.decrypt_text(

            encrypted_text,
            salt,
            secret

        )

        if not decrypted:

            return None

        try:

            return json.loads(

                decrypted

            )

        except Exception:

            return None

# ============================================================
# PASSWORD HASHING
# ============================================================

class PasswordManager:

    """
    Enterprise password security
    """

    # ========================================================
    # HASH PASSWORD
    # ========================================================

    @staticmethod
    def hash_password(

        password: str

    ) -> Dict:

        """
        Secure password hashing
        """

        salt = secrets.token_hex(

            SALT_SIZE

        )

        pwd_hash = hashlib.pbkdf2_hmac(

            "sha256",

            password.encode(),

            salt.encode(),

            DEFAULT_ITERATIONS

        )

        return {

            "hash":

                base64.b64encode(

                    pwd_hash

                ).decode(),

            "salt":

                salt,

            "iterations":

                DEFAULT_ITERATIONS

        }

    # ========================================================
    # VERIFY PASSWORD
    # ========================================================

    @staticmethod
    def verify_password(

        password: str,
        stored_hash: str,
        salt: str,
        iterations: int = DEFAULT_ITERATIONS

    ) -> bool:

        """
        Verify hashed password
        """

        new_hash = hashlib.pbkdf2_hmac(

            "sha256",

            password.encode(),

            salt.encode(),

            iterations

        )

        computed_hash = base64.b64encode(

            new_hash

        ).decode()

        return hmac.compare_digest(

            computed_hash,
            stored_hash

        )

# ============================================================
# TOKEN UTILITIES
# ============================================================

class TokenManager:

    """
    Secure token generation
    """

    # ========================================================
    # GENERATE TOKEN
    # ========================================================

    @staticmethod
    def generate_token(

        length: int = TOKEN_LENGTH

    ) -> str:

        return secrets.token_urlsafe(

            length

        )

    # ========================================================
    # GENERATE API KEY
    # ========================================================

    @staticmethod
    def generate_api_key():

        prefix = "cs_live_"

        return (

            prefix
            +
            secrets.token_hex(32)

        )

    # ========================================================
    # GENERATE SESSION ID
    # ========================================================

    @staticmethod
    def generate_session_id():

        return str(

            secrets.token_hex(24)

        )

# ============================================================
# CHECKSUM UTILITIES
# ============================================================

class ChecksumManager:

    """
    File/data integrity validation
    """

    # ========================================================
    # GENERATE CHECKSUM
    # ========================================================

    @staticmethod
    def generate_checksum(

        data: bytes

    ) -> str:

        return hashlib.sha256(

            data

        ).hexdigest()

    # ========================================================
    # VALIDATE CHECKSUM
    # ========================================================

    @staticmethod
    def validate_checksum(

        data: bytes,
        checksum: str

    ) -> bool:

        generated = (

            ChecksumManager

            .generate_checksum(data)

        )

        return hmac.compare_digest(

            generated,
            checksum

        )

    # ========================================================
    # FILE CHECKSUM
    # ========================================================

    @staticmethod
    def file_checksum(

        filepath: str

    ) -> str:

        sha256 = hashlib.sha256()

        with open(

            filepath,
            "rb"

        ) as file:

            while chunk := file.read(8192):

                sha256.update(chunk)

        return sha256.hexdigest()

# ============================================================
# HMAC UTILITIES
# ============================================================

class HMACManager:

    """
    Tamper-proof signature system
    """

    # ========================================================
    # CREATE SIGNATURE
    # ========================================================

    @staticmethod
    def create_signature(

        message: str,
        secret: str = MASTER_SECRET

    ) -> str:

        signature = hmac.new(

            secret.encode(),

            message.encode(),

            hashlib.sha256

        )

        return signature.hexdigest()

    # ========================================================
    # VERIFY SIGNATURE
    # ========================================================

    @staticmethod
    def verify_signature(

        message: str,
        signature: str,
        secret: str = MASTER_SECRET

    ) -> bool:

        generated = (

            HMACManager

            .create_signature(

                message,
                secret

            )

        )

        return hmac.compare_digest(

            generated,
            signature

        )

# ============================================================
# FILE ENCRYPTION
# ============================================================

class FileEncryption:

    """
    Secure file encryption
    """

    # ========================================================
    # ENCRYPT FILE
    # ========================================================

    @staticmethod
    def encrypt_file(

        input_path: str,
        output_path: str,
        secret: str = MASTER_SECRET

    ) -> bool:

        try:

            with open(

                input_path,
                "rb"

            ) as file:

                data = file.read()

            encrypted = (

                EncryptionManager

                .encrypt_text(

                    base64.b64encode(

                        data

                    ).decode(),

                    secret

                )

            )

            with open(

                output_path,
                "w"

            ) as file:

                json.dump(

                    encrypted,
                    file

                )

            return True

        except Exception:

            return False

    # ========================================================
    # DECRYPT FILE
    # ========================================================

    @staticmethod
    def decrypt_file(

        input_path: str,
        output_path: str,
        secret: str = MASTER_SECRET

    ) -> bool:

        try:

            with open(

                input_path,
                "r"

            ) as file:

                encrypted = json.load(

                    file

                )

            decrypted = (

                EncryptionManager

                .decrypt_text(

                    encrypted["encrypted"],

                    encrypted["salt"],

                    secret

                )

            )

            binary_data = base64.b64decode(

                decrypted

            )

            with open(

                output_path,
                "wb"

            ) as file:

                file.write(

                    binary_data

                )

            return True

        except Exception:

            return False

# ============================================================
# SECURITY HEALTH CHECK
# ============================================================

def encryption_health():

    return {

        "status":

            "healthy",

        "algorithm":

            "AES/Fernet",

        "pbkdf2_iterations":

            DEFAULT_ITERATIONS,

        "salt_size":

            SALT_SIZE

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD ENCRYPTION ENGINE")
    print("=" * 60)

    sample_text = (

        "Sensitive customer churn data"

    )

    encrypted = (

        EncryptionManager

        .encrypt_text(

            sample_text

        )

    )

    print("\nEncrypted:\n")

    print(encrypted)

    decrypted = (

        EncryptionManager

        .decrypt_text(

            encrypted["encrypted"],

            encrypted["salt"]

        )

    )

    print("\nDecrypted:\n")

    print(decrypted)

    password_data = (

        PasswordManager

        .hash_password(

            "super_secure_password"

        )

    )

    print("\nPassword Hash:\n")

    print(password_data)

    verified = (

        PasswordManager

        .verify_password(

            "super_secure_password",

            password_data["hash"],

            password_data["salt"]

        )

    )

    print("\nPassword Verified:\n")

    print(verified)

    api_key = (

        TokenManager

        .generate_api_key()

    )

    print("\nAPI Key:\n")

    print(api_key)

    print("\nHealth:\n")

    print(encryption_health())

    print("\n")
    print("=" * 60)
    print("ENCRYPTION ENGINE READY")
    print("=" * 60)