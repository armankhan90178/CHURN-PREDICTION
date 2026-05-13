"""
ChurnShield 2.0 — Repository Layer

File:
db/repositories.py

Purpose:
Enterprise-grade repository layer
for database operations in ChurnShield AI.

Capabilities:
- reusable CRUD operations
- optimized DB access layer
- pagination support
- filtering/search
- bulk inserts
- analytics queries
- prediction storage
- customer management
- audit logging
- model registry management
- API key management
- scheduler management
- export history
- enterprise-ready architecture

Author:
ChurnShield AI
"""

from datetime import datetime
from typing import (
    Optional,
    List,
    Dict,
    Any
)

from sqlalchemy.orm import Session
from sqlalchemy import desc

from db.models import (

    User,
    Dataset,
    Customer,
    Prediction,
    ModelRegistry,
    ExportHistory,
    ApiKey,
    AuditLog,
    SchedulerJob,
    Notification,
    AnalyticsCache,
    SystemMetric,
    RolePermission,
    ActivityTracker

)

# ============================================================
# BASE REPOSITORY
# ============================================================

class BaseRepository:

    def __init__(
        self,
        db: Session
    ):

        self.db = db

    # ========================================================
    # COMMIT
    # ========================================================

    def commit(self):

        self.db.commit()

    # ========================================================
    # ROLLBACK
    # ========================================================

    def rollback(self):

        self.db.rollback()

    # ========================================================
    # REFRESH
    # ========================================================

    def refresh(
        self,
        instance
    ):

        self.db.refresh(instance)

# ============================================================
# USER REPOSITORY
# ============================================================

class UserRepository(
    BaseRepository
):

    # ========================================================
    # CREATE USER
    # ========================================================

    def create_user(
        self,
        data: Dict
    ):

        user = User(**data)

        self.db.add(user)

        self.commit()

        self.refresh(user)

        return user

    # ========================================================
    # GET USER BY ID
    # ========================================================

    def get_user_by_id(
        self,
        user_id: int
    ):

        return (

            self.db.query(User)

            .filter(
                User.id == user_id
            )

            .first()

        )

    # ========================================================
    # GET USER BY EMAIL
    # ========================================================

    def get_user_by_email(
        self,
        email: str
    ):

        return (

            self.db.query(User)

            .filter(
                User.email == email
            )

            .first()

        )

    # ========================================================
    # GET ALL USERS
    # ========================================================

    def get_all_users(
        self,
        skip: int = 0,
        limit: int = 100
    ):

        return (

            self.db.query(User)

            .offset(skip)

            .limit(limit)

            .all()

        )

    # ========================================================
    # DELETE USER
    # ========================================================

    def delete_user(
        self,
        user_id: int
    ):

        user = self.get_user_by_id(
            user_id
        )

        if user:

            self.db.delete(user)

            self.commit()

        return user

# ============================================================
# CUSTOMER REPOSITORY
# ============================================================

class CustomerRepository(
    BaseRepository
):

    # ========================================================
    # CREATE CUSTOMER
    # ========================================================

    def create_customer(
        self,
        data: Dict
    ):

        customer = Customer(**data)

        self.db.add(customer)

        self.commit()

        self.refresh(customer)

        return customer

    # ========================================================
    # BULK INSERT
    # ========================================================

    def bulk_create_customers(
        self,
        customers: List[Dict]
    ):

        objs = [

            Customer(**item)

            for item in customers

        ]

        self.db.bulk_save_objects(
            objs
        )

        self.commit()

        return len(objs)

    # ========================================================
    # GET CUSTOMER
    # ========================================================

    def get_customer(
        self,
        customer_id: int
    ):

        return (

            self.db.query(Customer)

            .filter(
                Customer.id == customer_id
            )

            .first()

        )

    # ========================================================
    # SEARCH CUSTOMERS
    # ========================================================

    def search_customers(
        self,
        query: str
    ):

        return (

            self.db.query(Customer)

            .filter(

                Customer.full_name.ilike(
                    f"%{query}%"
                )

            )

            .all()

        )

    # ========================================================
    # HIGH RISK CUSTOMERS
    # ========================================================

    def high_risk_customers(

        self,
        threshold: float = 0.80

    ):

        return (

            self.db.query(Customer)

            .filter(

                Customer.churn_probability
                >= threshold

            )

            .all()

        )

    # ========================================================
    # UPDATE CHURN
    # ========================================================

    def update_churn_prediction(

        self,
        customer_id: int,
        probability: float

    ):

        customer = self.get_customer(
            customer_id
        )

        if not customer:

            return None

        customer.churn_probability = (
            probability
        )

        customer.churn_prediction = (
            probability > 0.5
        )

        self.commit()

        return customer

# ============================================================
# PREDICTION REPOSITORY
# ============================================================

class PredictionRepository(
    BaseRepository
):

    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    def save_prediction(
        self,
        data: Dict
    ):

        prediction = Prediction(
            **data
        )

        self.db.add(prediction)

        self.commit()

        self.refresh(prediction)

        return prediction

    # ========================================================
    # GET USER PREDICTIONS
    # ========================================================

    def get_user_predictions(
        self,
        user_id: int
    ):

        return (

            self.db.query(Prediction)

            .filter(
                Prediction.user_id
                == user_id
            )

            .order_by(
                desc(
                    Prediction.created_at
                )
            )

            .all()

        )

    # ========================================================
    # RECENT PREDICTIONS
    # ========================================================

    def recent_predictions(
        self,
        limit: int = 20
    ):

        return (

            self.db.query(Prediction)

            .order_by(
                desc(
                    Prediction.created_at
                )
            )

            .limit(limit)

            .all()

        )

# ============================================================
# DATASET REPOSITORY
# ============================================================

class DatasetRepository(
    BaseRepository
):

    # ========================================================
    # CREATE DATASET
    # ========================================================

    def create_dataset(
        self,
        data: Dict
    ):

        dataset = Dataset(**data)

        self.db.add(dataset)

        self.commit()

        self.refresh(dataset)

        return dataset

    # ========================================================
    # GET USER DATASETS
    # ========================================================

    def get_user_datasets(
        self,
        user_id: int
    ):

        return (

            self.db.query(Dataset)

            .filter(
                Dataset.user_id
                == user_id
            )

            .all()

        )

# ============================================================
# MODEL REGISTRY REPOSITORY
# ============================================================

class ModelRegistryRepository(
    BaseRepository
):

    # ========================================================
    # REGISTER MODEL
    # ========================================================

    def register_model(
        self,
        data: Dict
    ):

        model = ModelRegistry(
            **data
        )

        self.db.add(model)

        self.commit()

        self.refresh(model)

        return model

    # ========================================================
    # ACTIVE MODELS
    # ========================================================

    def active_models(self):

        return (

            self.db.query(ModelRegistry)

            .filter(
                ModelRegistry.is_active
                == True
            )

            .all()

        )

    # ========================================================
    # GET MODEL
    # ========================================================

    def get_model_by_name(
        self,
        name: str
    ):

        return (

            self.db.query(ModelRegistry)

            .filter(
                ModelRegistry.model_name
                == name
            )

            .first()

        )

# ============================================================
# EXPORT REPOSITORY
# ============================================================

class ExportRepository(
    BaseRepository
):

    def save_export(
        self,
        data: Dict
    ):

        export = ExportHistory(
            **data
        )

        self.db.add(export)

        self.commit()

        self.refresh(export)

        return export

    def user_exports(
        self,
        user_id: int
    ):

        return (

            self.db.query(ExportHistory)

            .filter(
                ExportHistory.user_id
                == user_id
            )

            .all()

        )

# ============================================================
# AUDIT LOG REPOSITORY
# ============================================================

class AuditRepository(
    BaseRepository
):

    def log(
        self,
        data: Dict
    ):

        audit = AuditLog(**data)

        self.db.add(audit)

        self.commit()

        return audit

    def recent_logs(
        self,
        limit: int = 100
    ):

        return (

            self.db.query(AuditLog)

            .order_by(
                desc(
                    AuditLog.created_at
                )
            )

            .limit(limit)

            .all()

        )

# ============================================================
# API KEY REPOSITORY
# ============================================================

class ApiKeyRepository(
    BaseRepository
):

    def create_api_key(
        self,
        data: Dict
    ):

        api_key = ApiKey(**data)

        self.db.add(api_key)

        self.commit()

        self.refresh(api_key)

        return api_key

    def validate_api_key(
        self,
        api_key: str
    ):

        return (

            self.db.query(ApiKey)

            .filter(
                ApiKey.api_key
                == api_key
            )

            .filter(
                ApiKey.is_active
                == True
            )

            .first()

        )

# ============================================================
# NOTIFICATION REPOSITORY
# ============================================================

class NotificationRepository(
    BaseRepository
):

    def create_notification(
        self,
        data: Dict
    ):

        notification = Notification(
            **data
        )

        self.db.add(notification)

        self.commit()

        self.refresh(notification)

        return notification

    def unread_notifications(
        self,
        user_id: int
    ):

        return (

            self.db.query(Notification)

            .filter(
                Notification.user_id
                == user_id
            )

            .filter(
                Notification.is_read
                == False
            )

            .all()

        )

# ============================================================
# ANALYTICS CACHE REPOSITORY
# ============================================================

class AnalyticsCacheRepository(
    BaseRepository
):

    def save_cache(
        self,
        key: str,
        value: Dict
    ):

        cache = AnalyticsCache(

            cache_key=key,

            cache_value=value,

            expires_at=datetime.utcnow()

        )

        self.db.add(cache)

        self.commit()

        return cache

    def get_cache(
        self,
        key: str
    ):

        return (

            self.db.query(
                AnalyticsCache
            )

            .filter(
                AnalyticsCache.cache_key
                == key
            )

            .first()

        )

# ============================================================
# SYSTEM METRICS REPOSITORY
# ============================================================

class MetricsRepository(
    BaseRepository
):

    def save_metric(
        self,
        data: Dict
    ):

        metric = SystemMetric(
            **data
        )

        self.db.add(metric)

        self.commit()

        return metric

    def latest_metrics(
        self,
        limit: int = 20
    ):

        return (

            self.db.query(SystemMetric)

            .order_by(
                desc(
                    SystemMetric.created_at
                )
            )

            .limit(limit)

            .all()

        )

# ============================================================
# SCHEDULER REPOSITORY
# ============================================================

class SchedulerRepository(
    BaseRepository
):

    def create_job(
        self,
        data: Dict
    ):

        job = SchedulerJob(
            **data
        )

        self.db.add(job)

        self.commit()

        return job

    def all_jobs(self):

        return (

            self.db.query(
                SchedulerJob
            )

            .all()

        )

# ============================================================
# ACTIVITY TRACKER REPOSITORY
# ============================================================

class ActivityRepository(
    BaseRepository
):

    def track_activity(
        self,
        data: Dict
    ):

        activity = ActivityTracker(
            **data
        )

        self.db.add(activity)

        self.commit()

        return activity

    def recent_activities(
        self,
        limit: int = 50
    ):

        return (

            self.db.query(
                ActivityTracker
            )

            .order_by(
                desc(
                    ActivityTracker.created_at
                )
            )

            .limit(limit)

            .all()

        )

# ============================================================
# ROLE PERMISSION REPOSITORY
# ============================================================

class PermissionRepository(
    BaseRepository
):

    def permissions_by_role(
        self,
        role: str
    ):

        return (

            self.db.query(
                RolePermission
            )

            .filter(
                RolePermission.role_name
                == role
            )

            .all()

        )

# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD REPOSITORY LAYER")
    print("=" * 60)

    print("\nRepositories Loaded Successfully\n")

    repos = [

        "UserRepository",
        "CustomerRepository",
        "PredictionRepository",
        "DatasetRepository",
        "ModelRegistryRepository",
        "ExportRepository",
        "AuditRepository",
        "ApiKeyRepository",
        "NotificationRepository",
        "AnalyticsCacheRepository",
        "MetricsRepository",
        "SchedulerRepository",
        "ActivityRepository",
        "PermissionRepository"

    ]

    for repo in repos:

        print(f"✔ {repo}")