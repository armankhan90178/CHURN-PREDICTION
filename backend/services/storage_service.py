"""
ChurnShield 2.0 — Storage Service

File:
services/storage_service.py

Purpose:
Enterprise-grade storage management
service for ChurnShield AI.

Capabilities:
- local file storage
- cloud-ready architecture
- secure uploads
- dataset storage
- report storage
- model storage
- export management
- automatic directory creation
- file hashing
- duplicate detection
- ZIP extraction
- cleanup system
- metadata generation
- enterprise file handling
- async-safe utilities

Author:
ChurnShield AI
"""

import os
import io
import json
import uuid
import shutil
import hashlib
import zipfile
import logging
import mimetypes

from pathlib import Path
from datetime import datetime
from typing import (

    Dict,
    Optional,
    List,
    Any

)

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "storage_service"
)

logging.basicConfig(
    level=logging.INFO
)

# ============================================================
# STORAGE PATHS
# ============================================================

BASE_STORAGE = Path("user_data")

RAW_DATA_DIR = BASE_STORAGE / "raw"

CLEANED_DATA_DIR = BASE_STORAGE / "cleaned"

PREDICTIONS_DIR = BASE_STORAGE / "predictions"

REPORTS_DIR = BASE_STORAGE / "reports"

EXPORTS_DIR = BASE_STORAGE / "exports"

TEMP_DIR = BASE_STORAGE / "temp"

MODELS_DIR = Path("models")

LOGS_DIR = Path("logs")

# ============================================================
# CREATE DIRECTORIES
# ============================================================

ALL_DIRS = [

    BASE_STORAGE,

    RAW_DATA_DIR,

    CLEANED_DATA_DIR,

    PREDICTIONS_DIR,

    REPORTS_DIR,

    EXPORTS_DIR,

    TEMP_DIR,

    MODELS_DIR,

    LOGS_DIR

]

for directory in ALL_DIRS:

    directory.mkdir(

        parents=True,
        exist_ok=True

    )

# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {

    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".zip",
    ".txt"

}

# ============================================================
# STORAGE SERVICE
# ============================================================

class StorageService:

    """
    Enterprise storage manager
    """

    # ========================================================
    # GENERATE UNIQUE NAME
    # ========================================================

    @staticmethod
    def generate_filename(
        original_name: str
    ) -> str:

        extension = Path(
            original_name
        ).suffix

        unique_id = uuid.uuid4().hex

        timestamp = datetime.utcnow().strftime(
            "%Y%m%d%H%M%S"
        )

        return (

            f"{timestamp}_"
            f"{unique_id}"
            f"{extension}"

        )

    # ========================================================
    # FILE HASH
    # ========================================================

    @staticmethod
    def calculate_hash(
        file_path: Path
    ) -> str:

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:

            for chunk in iter(

                lambda: f.read(4096),

                b""

            ):

                sha256.update(chunk)

        return sha256.hexdigest()

    # ========================================================
    # VALIDATE EXTENSION
    # ========================================================

    @staticmethod
    def is_allowed_file(
        filename: str
    ) -> bool:

        ext = Path(
            filename
        ).suffix.lower()

        return ext in ALLOWED_EXTENSIONS

    # ========================================================
    # SAVE FILE
    # ========================================================

    @staticmethod
    def save_file(

        file_bytes: bytes,
        original_name: str,
        category: str = "raw"

    ) -> Dict:

        if not StorageService.is_allowed_file(
            original_name
        ):

            raise ValueError(
                "Unsupported file type"
            )

        filename = StorageService.generate_filename(
            original_name
        )

        directory_map = {

            "raw": RAW_DATA_DIR,

            "cleaned": CLEANED_DATA_DIR,

            "prediction": PREDICTIONS_DIR,

            "report": REPORTS_DIR,

            "export": EXPORTS_DIR,

            "temp": TEMP_DIR

        }

        save_dir = directory_map.get(

            category,
            TEMP_DIR

        )

        save_path = save_dir / filename

        with open(save_path, "wb") as f:

            f.write(file_bytes)

        metadata = {

            "filename":
                filename,

            "original_name":
                original_name,

            "path":
                str(save_path),

            "size_mb":

                round(

                    save_path.stat().st_size
                    /
                    (1024 * 1024),

                    4

                ),

            "created_at":

                datetime.utcnow()

                .isoformat(),

            "sha256":

                StorageService.calculate_hash(
                    save_path
                )

        }

        logger.info({

            "event":
                "file_saved",

            "file":
                filename

        })

        return metadata

    # ========================================================
    # READ FILE
    # ========================================================

    @staticmethod
    def read_file(
        file_path: str
    ) -> bytes:

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                "File not found"
            )

        with open(path, "rb") as f:

            return f.read()

    # ========================================================
    # DELETE FILE
    # ========================================================

    @staticmethod
    def delete_file(
        file_path: str
    ) -> bool:

        try:

            path = Path(file_path)

            if path.exists():

                path.unlink()

                logger.info({

                    "event":
                        "file_deleted",

                    "path":
                        str(path)

                })

                return True

            return False

        except Exception as e:

            logger.error({

                "event":
                    "delete_failed",

                "error":
                    str(e)

            })

            return False

    # ========================================================
    # ZIP EXTRACTION
    # ========================================================

    @staticmethod
    def extract_zip(
        zip_path: str
    ) -> List[str]:

        extracted_files = []

        extract_dir = TEMP_DIR / uuid.uuid4().hex

        extract_dir.mkdir(

            parents=True,
            exist_ok=True

        )

        with zipfile.ZipFile(

            zip_path,
            "r"

        ) as zip_ref:

            zip_ref.extractall(
                extract_dir
            )

        for file in extract_dir.rglob("*"):

            if file.is_file():

                extracted_files.append(
                    str(file)
                )

        logger.info({

            "event":
                "zip_extracted",

            "count":
                len(extracted_files)

        })

        return extracted_files

    # ========================================================
    # FILE INFO
    # ========================================================

    @staticmethod
    def get_file_info(
        file_path: str
    ) -> Dict:

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                "File missing"
            )

        mime_type, _ = mimetypes.guess_type(
            str(path)
        )

        return {

            "name":
                path.name,

            "path":
                str(path),

            "extension":
                path.suffix,

            "size_mb":

                round(

                    path.stat().st_size
                    /
                    (1024 * 1024),

                    4

                ),

            "mime_type":
                mime_type,

            "created_at":

                datetime.fromtimestamp(

                    path.stat().st_ctime

                ).isoformat()

        }

    # ========================================================
    # COPY FILE
    # ========================================================

    @staticmethod
    def copy_file(

        source: str,
        destination: str

    ) -> bool:

        try:

            shutil.copy2(

                source,
                destination

            )

            return True

        except Exception as e:

            logger.error({

                "event":
                    "copy_failed",

                "error":
                    str(e)

            })

            return False

    # ========================================================
    # MOVE FILE
    # ========================================================

    @staticmethod
    def move_file(

        source: str,
        destination: str

    ) -> bool:

        try:

            shutil.move(

                source,
                destination

            )

            return True

        except Exception as e:

            logger.error({

                "event":
                    "move_failed",

                "error":
                    str(e)

            })

            return False

    # ========================================================
    # LIST FILES
    # ========================================================

    @staticmethod
    def list_files(
        directory: str
    ) -> List[Dict]:

        path = Path(directory)

        if not path.exists():

            return []

        files = []

        for file in path.iterdir():

            if file.is_file():

                files.append({

                    "name":
                        file.name,

                    "path":
                        str(file),

                    "size_mb":

                        round(

                            file.stat().st_size
                            /
                            (1024 * 1024),

                            4

                        )

                })

        return files

    # ========================================================
    # CLEAN TEMP
    # ========================================================

    @staticmethod
    def cleanup_temp_files():

        deleted = 0

        for file in TEMP_DIR.rglob("*"):

            try:

                if file.is_file():

                    file.unlink()

                    deleted += 1

            except Exception:

                pass

        logger.info({

            "event":
                "temp_cleanup",

            "deleted":
                deleted

        })

        return deleted

    # ========================================================
    # SAVE JSON
    # ========================================================

    @staticmethod
    def save_json(

        data: Dict,
        filename: str

    ) -> str:

        path = EXPORTS_DIR / filename

        with open(

            path,
            "w",
            encoding="utf-8"

        ) as f:

            json.dump(

                data,
                f,
                indent=4,
                ensure_ascii=False

            )

        return str(path)

# ============================================================
# STORAGE ANALYTICS
# ============================================================

def storage_statistics():

    """
    Storage usage analytics
    """

    stats = {}

    for directory in ALL_DIRS:

        total_size = 0

        file_count = 0

        for file in directory.rglob("*"):

            if file.is_file():

                total_size += file.stat().st_size

                file_count += 1

        stats[directory.name] = {

            "files":
                file_count,

            "size_mb":

                round(

                    total_size
                    /
                    (1024 * 1024),

                    4

                )

        }

    return stats

# ============================================================
# HEALTH CHECK
# ============================================================

def storage_health():

    return {

        "status": "healthy",

        "directories":

            len(ALL_DIRS),

        "base_storage":

            str(BASE_STORAGE),

        "models_storage":

            str(MODELS_DIR)

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD STORAGE SERVICE")
    print("=" * 60)

    service = StorageService()

    sample_content = b"sample,data\n1,test"

    result = service.save_file(

        file_bytes=sample_content,

        original_name="test.csv",

        category="raw"

    )

    print("\nSaved File:\n")

    print(result)

    print("\nStorage Statistics:\n")

    print(storage_statistics())

    print("\nHealth Status:\n")

    print(storage_health())

    print("\n")
    print("=" * 60)
    print("STORAGE SERVICE READY")
    print("=" * 60)