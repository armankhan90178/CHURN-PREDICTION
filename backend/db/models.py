"""
ChurnShield 2.0 — Database Models

File:
db/models.py

Purpose:
Enterprise-grade SQLAlchemy ORM models
for ChurnShield AI platform.

Capabilities:
- user management
- customer storage
- churn predictions
- uploaded datasets
- analytics history
- audit logs
- API key management
- ML model registry
- export history
- scheduler tracking
- notifications
- enterprise analytics
- role permissions
- activity tracking
- realtime metrics
- SaaS-ready architecture

Author:
ChurnShield AI
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey
)

from sqlalchemy.orm import (
    relationship,
    declarative_base
)

# ============================================================
# BASE
# ============================================================

Base = declarative_base()

# ============================================================
# USER MODEL
# ============================================================

class User(Base):

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    full_name = Column(
        String(255),
        nullable=False
    )

    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = Column(
        String(500),
        nullable=False
    )

    company = Column(
        String(255)
    )

    role = Column(
        String(100),
        default="user"
    )

    is_active = Column(
        Boolean,
        default=True
    )

    is_admin = Column(
        Boolean,
        default=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    last_login = Column(
        DateTime
    )

    # Relationships
    datasets = relationship(
        "Dataset",
        back_populates="user"
    )

    predictions = relationship(
        "Prediction",
        back_populates="user"
    )

    exports = relationship(
        "ExportHistory",
        back_populates="user"
    )

    api_keys = relationship(
        "ApiKey",
        back_populates="user"
    )

# ============================================================
# DATASET MODEL
# ============================================================

class Dataset(Base):

    __tablename__ = "datasets"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    file_name = Column(
        String(255),
        nullable=False
    )

    file_path = Column(
        String(1000)
    )

    rows_count = Column(
        Integer,
        default=0
    )

    columns_count = Column(
        Integer,
        default=0
    )

    file_size_mb = Column(
        Float,
        default=0.0
    )

    industry = Column(
        String(255)
    )

    upload_status = Column(
        String(100),
        default="uploaded"
    )

    validation_status = Column(
        String(100),
        default="pending"
    )

    schema_info = Column(
        JSON
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    # Relationships
    user = relationship(
        "User",
        back_populates="datasets"
    )

# ============================================================
# CUSTOMER MODEL
# ============================================================

class Customer(Base):

    __tablename__ = "customers"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    customer_uid = Column(
        String(255),
        unique=True,
        index=True
    )

    full_name = Column(
        String(255)
    )

    email = Column(
        String(255)
    )

    phone = Column(
        String(50)
    )

    region = Column(
        String(255)
    )

    industry = Column(
        String(255)
    )

    monthly_spend = Column(
        Float,
        default=0.0
    )

    tenure = Column(
        Integer,
        default=0
    )

    churn_probability = Column(
        Float,
        default=0.0
    )

    churn_prediction = Column(
        Boolean,
        default=False
    )

    risk_level = Column(
        String(100),
        default="low"
    )

    persona = Column(
        String(255)
    )

    sentiment = Column(
        String(100)
    )

    metadata_json = Column(
        JSON
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# ============================================================
# PREDICTION MODEL
# ============================================================

class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    customer_id = Column(
        Integer,
        ForeignKey("customers.id")
    )

    model_name = Column(
        String(255)
    )

    prediction_value = Column(
        Integer
    )

    churn_probability = Column(
        Float
    )

    confidence_score = Column(
        Float
    )

    risk_level = Column(
        String(100)
    )

    explanation = Column(
        JSON
    )

    recommendations = Column(
        JSON
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    # Relationships
    user = relationship(
        "User",
        back_populates="predictions"
    )

# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry(Base):

    __tablename__ = "model_registry"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    model_name = Column(
        String(255),
        unique=True
    )

    model_version = Column(
        String(100)
    )

    model_path = Column(
        String(1000)
    )

    industry = Column(
        String(255)
    )

    accuracy = Column(
        Float
    )

    precision = Column(
        Float
    )

    recall = Column(
        Float
    )

    f1_score = Column(
        Float
    )

    roc_auc = Column(
        Float
    )

    training_rows = Column(
        Integer
    )

    features_used = Column(
        JSON
    )

    is_active = Column(
        Boolean,
        default=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# ============================================================
# EXPORT HISTORY
# ============================================================

class ExportHistory(Base):

    __tablename__ = "export_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    export_type = Column(
        String(100)
    )

    file_name = Column(
        String(255)
    )

    file_path = Column(
        String(1000)
    )

    download_count = Column(
        Integer,
        default=0
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    # Relationships
    user = relationship(
        "User",
        back_populates="exports"
    )

# ============================================================
# API KEY MODEL
# ============================================================

class ApiKey(Base):

    __tablename__ = "api_keys"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    api_key = Column(
        String(500),
        unique=True
    )

    is_active = Column(
        Boolean,
        default=True
    )

    requests_count = Column(
        Integer,
        default=0
    )

    last_used = Column(
        DateTime
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    # Relationships
    user = relationship(
        "User",
        back_populates="api_keys"
    )

# ============================================================
# AUDIT LOG MODEL
# ============================================================

class AuditLog(Base):

    __tablename__ = "audit_logs"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer
    )

    action = Column(
        String(255)
    )

    endpoint = Column(
        String(500)
    )

    method = Column(
        String(50)
    )

    ip_address = Column(
        String(100)
    )

    status = Column(
        String(100)
    )

    metadata_json = Column(
        JSON
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# ============================================================
# SCHEDULER JOB MODEL
# ============================================================

class SchedulerJob(Base):

    __tablename__ = "scheduler_jobs"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    job_name = Column(
        String(255)
    )

    job_type = Column(
        String(255)
    )

    status = Column(
        String(100),
        default="pending"
    )

    last_run = Column(
        DateTime
    )

    next_run = Column(
        DateTime
    )

    execution_time = Column(
        Float
    )

    logs = Column(
        Text
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# ============================================================
# NOTIFICATION MODEL
# ============================================================

class Notification(Base):

    __tablename__ = "notifications"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer
    )

    title = Column(
        String(255)
    )

    message = Column(
        Text
    )

    notification_type = Column(
        String(100)
    )

    is_read = Column(
        Boolean,
        default=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# ============================================================
# ANALYTICS CACHE
# ============================================================

class AnalyticsCache(Base):

    __tablename__ = "analytics_cache"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    cache_key = Column(
        String(255),
        unique=True
    )

    cache_value = Column(
        JSON
    )

    expires_at = Column(
        DateTime
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# ============================================================
# SYSTEM METRICS
# ============================================================

class SystemMetric(Base):

    __tablename__ = "system_metrics"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    cpu_usage = Column(
        Float
    )

    memory_usage = Column(
        Float
    )

    disk_usage = Column(
        Float
    )

    active_users = Column(
        Integer
    )

    active_models = Column(
        Integer
    )

    api_requests = Column(
        Integer
    )

    prediction_latency = Column(
        Float
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# ============================================================
# ROLE PERMISSIONS
# ============================================================

class RolePermission(Base):

    __tablename__ = "role_permissions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    role_name = Column(
        String(255)
    )

    permission_name = Column(
        String(255)
    )

    can_create = Column(
        Boolean,
        default=False
    )

    can_read = Column(
        Boolean,
        default=True
    )

    can_update = Column(
        Boolean,
        default=False
    )

    can_delete = Column(
        Boolean,
        default=False
    )

# ============================================================
# ACTIVITY TRACKER
# ============================================================

class ActivityTracker(Base):

    __tablename__ = "activity_tracker"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer
    )

    activity_type = Column(
        String(255)
    )

    activity_data = Column(
        JSON
    )

    device = Column(
        String(255)
    )

    browser = Column(
        String(255)
    )

    location = Column(
        String(255)
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD DATABASE MODELS")
    print("=" * 60)

    print("\nAvailable Tables:\n")

    for table in Base.metadata.tables:

        print(f"✔ {table}")