"""
ChurnShield 2.0 — Database Seeder

File:
db/seed.py

Purpose:
Enterprise-grade database seeding engine
for ChurnShield AI platform.

Capabilities:
- seed demo users
- seed customers
- seed churn predictions
- seed analytics cache
- seed API keys
- seed notifications
- seed scheduler jobs
- seed audit logs
- seed model registry
- generate realistic churn datasets
- enterprise demo environment
- SaaS-ready test data
- benchmark-ready records
- fast synthetic generation
- industry-wise customer creation

Author:
ChurnShield AI
"""

import random
import secrets
from datetime import (
    datetime,
    timedelta
)

from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import (

    Base,

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
# CONFIG
# ============================================================

DATABASE_URL = "sqlite:///./churnshield.db"

fake = Faker()

# ============================================================
# DATABASE
# ============================================================

engine = create_engine(

    DATABASE_URL,

    connect_args={
        "check_same_thread": False
    }

)

SessionLocal = sessionmaker(

    autocommit=False,
    autoflush=False,
    bind=engine

)

# ============================================================
# CREATE TABLES
# ============================================================

Base.metadata.create_all(
    bind=engine
)

# ============================================================
# HELPERS
# ============================================================

INDUSTRIES = [

    "SaaS",
    "Telecom",
    "OTT",
    "Healthcare",
    "Banking",
    "Insurance",
    "Ecommerce",
    "Gaming"

]

REGIONS = [

    "North",
    "South",
    "East",
    "West"

]

PERSONAS = [

    "Loyal",
    "At Risk",
    "Premium",
    "Casual",
    "Enterprise"

]

SENTIMENTS = [

    "positive",
    "neutral",
    "negative"

]

RISK_LEVELS = [

    "low",
    "medium",
    "high",
    "critical"

]

# ============================================================
# SEED USERS
# ============================================================

def seed_users(db):

    users = []

    for i in range(10):

        user = User(

            full_name=fake.name(),

            email=f"user{i}@churnshield.ai",

            password_hash=secrets.token_hex(32),

            company=fake.company(),

            role="admin" if i == 0 else "user",

            is_admin=True if i == 0 else False,

            is_active=True,

            created_at=datetime.utcnow()

        )

        db.add(user)

        users.append(user)

    db.commit()

    return users

# ============================================================
# SEED DATASETS
# ============================================================

def seed_datasets(
    db,
    users
):

    datasets = []

    for user in users:

        for i in range(3):

            dataset = Dataset(

                user_id=user.id,

                file_name=f"dataset_{i}.csv",

                file_path=f"/user_data/{user.id}/dataset_{i}.csv",

                rows_count=random.randint(
                    1000,
                    50000
                ),

                columns_count=random.randint(
                    10,
                    100
                ),

                file_size_mb=round(

                    random.uniform(
                        1,
                        200
                    ),

                    2

                ),

                industry=random.choice(
                    INDUSTRIES
                ),

                upload_status="uploaded",

                validation_status="validated",

                schema_info={

                    "customer_id":
                        "string",

                    "monthly_spend":
                        "float"

                }

            )

            db.add(dataset)

            datasets.append(dataset)

    db.commit()

    return datasets

# ============================================================
# SEED CUSTOMERS
# ============================================================

def seed_customers(
    db,
    count=1000
):

    customers = []

    for i in range(count):

        churn_probability = round(

            random.uniform(
                0,
                1
            ),

            4

        )

        customer = Customer(

            customer_uid=f"CUST-{100000+i}",

            full_name=fake.name(),

            email=fake.email(),

            phone=fake.phone_number(),

            region=random.choice(
                REGIONS
            ),

            industry=random.choice(
                INDUSTRIES
            ),

            monthly_spend=round(

                random.uniform(
                    100,
                    10000
                ),

                2

            ),

            tenure=random.randint(
                1,
                60
            ),

            churn_probability=
                churn_probability,

            churn_prediction=
                churn_probability > 0.5,

            risk_level=random.choice(
                RISK_LEVELS
            ),

            persona=random.choice(
                PERSONAS
            ),

            sentiment=random.choice(
                SENTIMENTS
            ),

            metadata_json={

                "source":
                    "synthetic_seed",

                "engagement":
                    random.randint(
                        1,
                        100
                    )

            }

        )

        db.add(customer)

        customers.append(customer)

    db.commit()

    return customers

# ============================================================
# SEED PREDICTIONS
# ============================================================

def seed_predictions(
    db,
    users,
    customers
):

    predictions = []

    for i in range(500):

        customer = random.choice(
            customers
        )

        user = random.choice(
            users
        )

        probability = round(

            random.uniform(
                0,
                1
            ),

            4

        )

        prediction = Prediction(

            user_id=user.id,

            customer_id=customer.id,

            model_name="global_model.pkl",

            prediction_value=
                int(probability > 0.5),

            churn_probability=
                probability,

            confidence_score=
                round(

                    random.uniform(
                        0.70,
                        0.99
                    ),

                    4

                ),

            risk_level=random.choice(
                RISK_LEVELS
            ),

            explanation={

                "top_reason":
                    "Low engagement",

                "shap_score":
                    round(

                        random.uniform(
                            0,
                            1
                        ),

                        4

                    )

            },

            recommendations=[

                "Offer discount",

                "Send retention email"

            ]

        )

        db.add(prediction)

        predictions.append(prediction)

    db.commit()

    return predictions

# ============================================================
# SEED MODEL REGISTRY
# ============================================================

def seed_models(db):

    for industry in INDUSTRIES:

        model = ModelRegistry(

            model_name=
                f"{industry.lower()}_model.pkl",

            model_version="v1.0",

            model_path=
                f"/models/{industry.lower()}_model.pkl",

            industry=industry,

            accuracy=round(

                random.uniform(
                    0.82,
                    0.98
                ),

                4

            ),

            precision=round(

                random.uniform(
                    0.80,
                    0.97
                ),

                4

            ),

            recall=round(

                random.uniform(
                    0.78,
                    0.95
                ),

                4

            ),

            f1_score=round(

                random.uniform(
                    0.80,
                    0.96
                ),

                4

            ),

            roc_auc=round(

                random.uniform(
                    0.85,
                    0.99
                ),

                4

            ),

            training_rows=random.randint(
                5000,
                500000
            ),

            features_used=[

                "monthly_spend",

                "tenure",

                "support_calls"

            ]

        )

        db.add(model)

    db.commit()

# ============================================================
# SEED API KEYS
# ============================================================

def seed_api_keys(
    db,
    users
):

    for user in users:

        api_key = ApiKey(

            user_id=user.id,

            api_key=secrets.token_hex(32),

            is_active=True,

            requests_count=random.randint(
                100,
                50000
            ),

            last_used=datetime.utcnow()

        )

        db.add(api_key)

    db.commit()

# ============================================================
# SEED AUDIT LOGS
# ============================================================

def seed_audit_logs(db):

    actions = [

        "login",
        "upload",
        "prediction",
        "export",
        "dashboard_view"

    ]

    for i in range(300):

        log = AuditLog(

            user_id=random.randint(
                1,
                10
            ),

            action=random.choice(
                actions
            ),

            endpoint="/api/v1/test",

            method="POST",

            ip_address=fake.ipv4(),

            status="success",

            metadata_json={

                "latency_ms":
                    random.randint(
                        10,
                        300
                    )

            }

        )

        db.add(log)

    db.commit()

# ============================================================
# SEED NOTIFICATIONS
# ============================================================

def seed_notifications(db):

    for i in range(50):

        notification = Notification(

            user_id=random.randint(
                1,
                10
            ),

            title="High Churn Alert",

            message=
                "Critical churn spike detected.",

            notification_type="alert",

            is_read=random.choice(
                [True, False]
            )

        )

        db.add(notification)

    db.commit()

# ============================================================
# SEED ANALYTICS CACHE
# ============================================================

def seed_cache(db):

    for i in range(20):

        cache = AnalyticsCache(

            cache_key=f"dashboard_cache_{i}",

            cache_value={

                "value":
                    random.randint(
                        1,
                        1000
                    )

            },

            expires_at=

                datetime.utcnow()

                +

                timedelta(hours=1)

        )

        db.add(cache)

    db.commit()

# ============================================================
# SEED SYSTEM METRICS
# ============================================================

def seed_metrics(db):

    for i in range(100):

        metric = SystemMetric(

            cpu_usage=round(

                random.uniform(
                    10,
                    90
                ),

                2

            ),

            memory_usage=round(

                random.uniform(
                    20,
                    95
                ),

                2

            ),

            disk_usage=round(

                random.uniform(
                    10,
                    80
                ),

                2

            ),

            active_users=random.randint(
                10,
                1000
            ),

            active_models=random.randint(
                1,
                20
            ),

            api_requests=random.randint(
                1000,
                100000
            ),

            prediction_latency=round(

                random.uniform(
                    10,
                    300
                ),

                2

            )

        )

        db.add(metric)

    db.commit()

# ============================================================
# SEED ROLE PERMISSIONS
# ============================================================

def seed_permissions(db):

    roles = [

        "admin",
        "manager",
        "user"

    ]

    permissions = [

        "dashboard",
        "prediction",
        "upload",
        "export",
        "analytics"

    ]

    for role in roles:

        for permission in permissions:

            perm = RolePermission(

                role_name=role,

                permission_name=permission,

                can_create=True,

                can_read=True,

                can_update=role != "user",

                can_delete=role == "admin"

            )

            db.add(perm)

    db.commit()

# ============================================================
# SEED ACTIVITY TRACKER
# ============================================================

def seed_activity_tracker(db):

    for i in range(200):

        activity = ActivityTracker(

            user_id=random.randint(
                1,
                10
            ),

            activity_type=random.choice([

                "login",
                "prediction",
                "export",
                "upload"

            ]),

            activity_data={

                "device":
                    "Chrome",

                "duration":
                    random.randint(
                        10,
                        500
                    )

            },

            device="Desktop",

            browser="Chrome",

            location="India"

        )

        db.add(activity)

    db.commit()

# ============================================================
# MAIN SEED FUNCTION
# ============================================================

def run_seed():

    db = SessionLocal()

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD DATABASE SEEDER")
    print("=" * 60)

    print("\nSeeding users...")
    users = seed_users(db)

    print("✔ Users seeded")

    print("\nSeeding datasets...")
    seed_datasets(db, users)

    print("✔ Datasets seeded")

    print("\nSeeding customers...")
    customers = seed_customers(db)

    print("✔ Customers seeded")

    print("\nSeeding predictions...")
    seed_predictions(
        db,
        users,
        customers
    )

    print("✔ Predictions seeded")

    print("\nSeeding models...")
    seed_models(db)

    print("✔ Models seeded")

    print("\nSeeding API keys...")
    seed_api_keys(db, users)

    print("✔ API keys seeded")

    print("\nSeeding audit logs...")
    seed_audit_logs(db)

    print("✔ Audit logs seeded")

    print("\nSeeding notifications...")
    seed_notifications(db)

    print("✔ Notifications seeded")

    print("\nSeeding analytics cache...")
    seed_cache(db)

    print("✔ Analytics cache seeded")

    print("\nSeeding metrics...")
    seed_metrics(db)

    print("✔ Metrics seeded")

    print("\nSeeding permissions...")
    seed_permissions(db)

    print("✔ Permissions seeded")

    print("\nSeeding activity tracker...")
    seed_activity_tracker(db)

    print("✔ Activity tracker seeded")

    print("\n")
    print("=" * 60)
    print("DATABASE SEEDING COMPLETED")
    print("=" * 60)

    db.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_seed()