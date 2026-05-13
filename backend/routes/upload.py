"""
ChurnShield 2.0 — Upload Routes
Enterprise Upload + AI Processing Pipeline
"""

import os
import uuid
import shutil
import logging
import traceback
import pandas as pd

from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Form,
)

from fastapi.responses import FileResponse

from db.database import get_database_connection

from config import (
    USER_DATA_DIR,
    MAX_UPLOAD_ROWS,
    MAX_UPLOAD_COLUMNS,
    ALLOWED_UPLOAD_EXTENSIONS,
)

from upload.file_handler import (
    load_uploaded_file,
)

from upload.label_generator import (
    generate_churn_labels,
)

from upload.template_builder import (
    build_template_dataset,
)

from data.validator import (
    validate_dataset,
)

from data.cleaner import (
    clean_dataset,
)

from data.standardizer import (
    standardize_dataset,
)

from data.fetcher import (
    fetch_data_for_field,
)

from ml.feature_engineer import (
    engineer_features,
)

logger = logging.getLogger(
    "churnshield.routes.upload"
)

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)


# ─────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────

def validate_upload_file(
    upload_file: UploadFile,
):

    extension = (
        Path(upload_file.filename)
        .suffix
        .lower()
    )

    if extension not in ALLOWED_UPLOAD_EXTENSIONS:

        raise HTTPException(

            status_code=400,

            detail=f"""
            Unsupported file format.

            Allowed:
            {ALLOWED_UPLOAD_EXTENSIONS}
            """,
        )

    return True


# ─────────────────────────────────────────────
# SAVE FILE
# ─────────────────────────────────────────────

def save_uploaded_file(
    upload_file: UploadFile,
    user_id: str,
):

    try:

        user_dir = (
            USER_DATA_DIR / user_id
        )

        user_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        extension = (
            Path(upload_file.filename)
            .suffix
            .lower()
        )

        file_path = (
            user_dir /
            f"raw_upload{extension}"
        )

        with open(
            file_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                upload_file.file,
                buffer,
            )

        logger.info(
            f"Saved uploaded file: {file_path}"
        )

        return file_path

    except Exception as e:

        logger.error(
            f"File save failed: {e}"
        )

        raise


# ─────────────────────────────────────────────
# MAIN DATASET UPLOAD
# ─────────────────────────────────────────────

@router.post("/")
async def upload_dataset(

    file: UploadFile = File(...),

    industry: str = Form("general"),

    auto_clean: bool = Form(True),

    auto_standardize: bool = Form(True),

    auto_engineer: bool = Form(True),

    auto_label: bool = Form(True),
):

    try:

        logger.info(
            f"Upload started: {file.filename}"
        )

        validate_upload_file(file)

        user_id = uuid.uuid4().hex

        # SAVE FILE
        file_path = save_uploaded_file(
            file,
            user_id,
        )

        # LOAD DATASET
        df = load_uploaded_file(
            str(file_path)
        )

        if len(df) == 0:

            raise HTTPException(

                status_code=400,

                detail="Uploaded dataset is empty",
            )

        if len(df) > MAX_UPLOAD_ROWS:

            raise HTTPException(

                status_code=400,

                detail=f"""
                Dataset exceeds max rows:
                {MAX_UPLOAD_ROWS}
                """,
            )

        if len(df.columns) > MAX_UPLOAD_COLUMNS:

            raise HTTPException(

                status_code=400,

                detail=f"""
                Dataset exceeds max columns:
                {MAX_UPLOAD_COLUMNS}
                """,
            )

        processing_steps = []

        # VALIDATION
        validation_report = validate_dataset(df)

        processing_steps.append(
            "validation_completed"
        )

        # CLEANING
        if auto_clean:

            # FIX: clean_dataset() returns (df, metadata) — unpack the tuple
            df, _clean_meta = clean_dataset(df)

            processing_steps.append(
                "cleaning_completed"
            )

        # STANDARDIZATION
        if auto_standardize:

            df = standardize_dataset(df)

            processing_steps.append(
                "standardization_completed"
            )

        # LABEL GENERATION
        if auto_label:

            df = generate_churn_labels(df)

            processing_steps.append(
                "label_generation_completed"
            )

        # FEATURE ENGINEERING
        if auto_engineer:

            df = engineer_features(df)

            processing_steps.append(
                "feature_engineering_completed"
            )

        # SAVE CLEANED FILE
        cleaned_path = (
            USER_DATA_DIR /
            user_id /
            "cleaned_data.csv"
        )

        df.to_csv(
            cleaned_path,
            index=False,
        )

        # SAVE TO DATABASE
        conn = get_database_connection()

        df.to_sql(

            "customers",

            conn,

            if_exists="replace",

            index=False,
        )

        conn.commit()

        conn.close()

        logger.info(
            f"Dataset stored successfully ({len(df)} rows)"
        )

        return {

            "status":
                "success",

            "user_id":
                user_id,

            "industry":
                industry,

            "rows":
                int(len(df)),

            "columns":
                int(len(df.columns)),

            "processing_steps":
                processing_steps,

            # FIX: convert ValidationReport dataclass to dict so FastAPI
            # can JSON-serialise it; returning the dataclass directly crashes
            # the response encoder.
            "validation_report":
                asdict(validation_report),

            "saved_path":
                str(cleaned_path),

            "preview":

                df.head(10)
                .fillna("")
                .to_dict(
                    orient="records"
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Upload pipeline failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail=f"""
            Upload failed:
            {str(e)}
            """,
        )


# ─────────────────────────────────────────────
# GENERATE INDUSTRY DATASET
# ─────────────────────────────────────────────

@router.post("/generate")
async def generate_dataset(
    industry: str = Form(...),
):

    try:

        logger.info(
            f"Generating dataset for: {industry}"
        )

        df = fetch_data_for_field(
            industry
        )

        if len(df) == 0:

            raise HTTPException(

                status_code=500,

                detail="Dataset generation failed",
            )

        # AI PIPELINE
        # FIX: unpack (df, metadata) tuple from clean_dataset()
        df, _clean_meta = clean_dataset(df)

        df = standardize_dataset(df)

        df = generate_churn_labels(df)

        df = engineer_features(df)

        # SAVE USER DATA
        user_id = uuid.uuid4().hex

        user_dir = (
            USER_DATA_DIR / user_id
        )

        user_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_path = (
            user_dir /
            "generated_dataset.csv"
        )

        df.to_csv(
            dataset_path,
            index=False,
        )

        # DATABASE SAVE
        conn = get_database_connection()

        df.to_sql(

            "customers",

            conn,

            if_exists="replace",

            index=False,
        )

        conn.commit()

        conn.close()

        return {

            "status":
                "success",

            "industry":
                industry,

            "user_id":
                user_id,

            "rows":
                int(len(df)),

            "columns":
                int(len(df.columns)),

            "dataset_path":
                str(dataset_path),

            "preview":

                df.head(10)
                .fillna("")
                .to_dict(
                    orient="records"
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Dataset generation failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail=str(e),
        )


# ─────────────────────────────────────────────
# TEMPLATE DOWNLOAD
# ─────────────────────────────────────────────

@router.get("/template")
async def download_template():

    try:

        logger.info(
            "Generating upload template..."
        )

        df = build_template_dataset()

        if df is None or len(df) == 0:

            raise HTTPException(

                status_code=500,

                detail="Template generation failed",
            )

        template_dir = (
            USER_DATA_DIR / "templates"
        )

        template_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        template_filename = (
            f"churnshield_template_{timestamp}.csv"
        )

        template_path = (
            template_dir / template_filename
        )

        df.to_csv(
            template_path,
            index=False,
            encoding="utf-8",
        )

        if not template_path.exists():

            raise HTTPException(

                status_code=500,

                detail="Template file creation failed",
            )

        return FileResponse(

            path=str(template_path),

            filename=template_filename,

            media_type="text/csv",
        )

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Template download failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail=str(e),
        )


# ─────────────────────────────────────────────
# DATASET PREVIEW
# ─────────────────────────────────────────────

@router.post("/preview")
async def preview_dataset(
    file: UploadFile = File(...),
):

    try:

        # FIX: re-raise HTTPException before the generic handler so a bad file
        # extension returns 400 instead of being swallowed and re-raised as 500.
        validate_upload_file(file)

    except HTTPException:

        raise

    try:

        temp_id = uuid.uuid4().hex

        file_path = save_uploaded_file(
            file,
            temp_id,
        )

        df = load_uploaded_file(
            str(file_path)
        )

        summary = {

            "rows":
                int(len(df)),

            "columns":
                int(len(df.columns)),

            "column_names":
                list(df.columns),

            "missing_values":

                df.isnull()
                .sum()
                .astype(int)
                .to_dict(),

            "data_types":

                df.dtypes
                .astype(str)
                .to_dict(),
        }

        return {

            "status":
                "success",

            "summary":
                summary,

            "preview":

                df.head(15)
                .fillna("")
                .to_dict(
                    orient="records"
                ),
        }

    except Exception as e:

        logger.error(
            f"Preview failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail=str(e),
        )


# ─────────────────────────────────────────────
# LIST DATASETS
# ─────────────────────────────────────────────

@router.get("/datasets")
async def list_uploaded_datasets():

    try:

        datasets = []

        if not USER_DATA_DIR.exists():

            USER_DATA_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

        for folder in USER_DATA_DIR.iterdir():

            # FIX: skip the reserved "templates" directory so it is never
            # surfaced as a user dataset entry in the listing.
            if not folder.is_dir() or folder.name == "templates":
                continue

            files = list(
                folder.glob("*")
            )

            datasets.append({

                "user_id":
                    folder.name,

                "files":
                    [f.name for f in files],

                "created_at":

                    datetime.fromtimestamp(
                        folder.stat().st_ctime
                    ).isoformat(),
            })

        return {

            "status":
                "success",

            "datasets":
                datasets,
        }

    except Exception as e:

        logger.error(
            f"Dataset listing failed: {e}"
        )

        raise HTTPException(

            status_code=500,

            detail=str(e),
        )


# ─────────────────────────────────────────────
# DELETE DATASET
# ─────────────────────────────────────────────

@router.delete("/{user_id}")
async def delete_dataset(
    user_id: str,
):

    try:

        # FIX: resolve the path and verify it sits inside USER_DATA_DIR before
        # deleting — prevents path traversal attacks (e.g. user_id="../sibling").
        user_dir = (USER_DATA_DIR / user_id).resolve()
        base_dir = USER_DATA_DIR.resolve()

        if not str(user_dir).startswith(str(base_dir) + os.sep):

            raise HTTPException(

                status_code=400,

                detail="Invalid dataset ID",
            )

        if not user_dir.exists():

            raise HTTPException(

                status_code=404,

                detail="Dataset not found",
            )

        shutil.rmtree(user_dir)

        logger.info(
            f"Deleted dataset: {user_id}"
        )

        return {

            "status":
                "success",

            "deleted_user_id":
                user_id,
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Dataset deletion failed: {e}"
        )

        raise HTTPException(

            status_code=500,

            detail=str(e),
        )