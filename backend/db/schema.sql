-- ============================================================
-- ChurnShield 2.0 — Enterprise Database Schema
-- File: db/schema.sql
-- Database: PostgreSQL / SQLite Compatible
-- ============================================================

-- ============================================================
-- USERS
-- ============================================================

CREATE TABLE IF NOT EXISTS users (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    full_name VARCHAR(255) NOT NULL,

    email VARCHAR(255) UNIQUE NOT NULL,

    password_hash VARCHAR(500) NOT NULL,

    company VARCHAR(255),

    role VARCHAR(100) DEFAULT 'user',

    is_active BOOLEAN DEFAULT 1,

    is_admin BOOLEAN DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    last_login TIMESTAMP

);

CREATE INDEX IF NOT EXISTS idx_users_email
ON users(email);

-- ============================================================
-- DATASETS
-- ============================================================

CREATE TABLE IF NOT EXISTS datasets (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER,

    file_name VARCHAR(255) NOT NULL,

    file_path VARCHAR(1000),

    rows_count INTEGER DEFAULT 0,

    columns_count INTEGER DEFAULT 0,

    file_size_mb FLOAT DEFAULT 0,

    industry VARCHAR(255),

    upload_status VARCHAR(100) DEFAULT 'uploaded',

    validation_status VARCHAR(100) DEFAULT 'pending',

    schema_info TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(user_id)
    REFERENCES users(id)

);

CREATE INDEX IF NOT EXISTS idx_dataset_user
ON datasets(user_id);

-- ============================================================
-- CUSTOMERS
-- ============================================================

CREATE TABLE IF NOT EXISTS customers (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    customer_uid VARCHAR(255) UNIQUE,

    full_name VARCHAR(255),

    email VARCHAR(255),

    phone VARCHAR(50),

    region VARCHAR(255),

    industry VARCHAR(255),

    monthly_spend FLOAT DEFAULT 0,

    tenure INTEGER DEFAULT 0,

    churn_probability FLOAT DEFAULT 0,

    churn_prediction BOOLEAN DEFAULT 0,

    risk_level VARCHAR(100) DEFAULT 'low',

    persona VARCHAR(255),

    sentiment VARCHAR(100),

    metadata_json TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

CREATE INDEX IF NOT EXISTS idx_customer_uid
ON customers(customer_uid);

CREATE INDEX IF NOT EXISTS idx_customer_risk
ON customers(risk_level);

-- ============================================================
-- PREDICTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER,

    customer_id INTEGER,

    model_name VARCHAR(255),

    prediction_value INTEGER,

    churn_probability FLOAT,

    confidence_score FLOAT,

    risk_level VARCHAR(100),

    explanation TEXT,

    recommendations TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(user_id)
    REFERENCES users(id),

    FOREIGN KEY(customer_id)
    REFERENCES customers(id)

);

CREATE INDEX IF NOT EXISTS idx_prediction_customer
ON predictions(customer_id);

-- ============================================================
-- MODEL REGISTRY
-- ============================================================

CREATE TABLE IF NOT EXISTS model_registry (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    model_name VARCHAR(255) UNIQUE,

    model_version VARCHAR(100),

    model_path VARCHAR(1000),

    industry VARCHAR(255),

    accuracy FLOAT,

    precision_score FLOAT,

    recall_score FLOAT,

    f1_score FLOAT,

    roc_auc FLOAT,

    training_rows INTEGER,

    features_used TEXT,

    is_active BOOLEAN DEFAULT 1,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

CREATE INDEX IF NOT EXISTS idx_model_name
ON model_registry(model_name);

-- ============================================================
-- EXPORT HISTORY
-- ============================================================

CREATE TABLE IF NOT EXISTS export_history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER,

    export_type VARCHAR(100),

    file_name VARCHAR(255),

    file_path VARCHAR(1000),

    download_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(user_id)
    REFERENCES users(id)

);

-- ============================================================
-- API KEYS
-- ============================================================

CREATE TABLE IF NOT EXISTS api_keys (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER,

    api_key VARCHAR(500) UNIQUE,

    is_active BOOLEAN DEFAULT 1,

    requests_count INTEGER DEFAULT 0,

    last_used TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(user_id)
    REFERENCES users(id)

);

CREATE INDEX IF NOT EXISTS idx_api_key
ON api_keys(api_key);

-- ============================================================
-- AUDIT LOGS
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_logs (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER,

    action VARCHAR(255),

    endpoint VARCHAR(500),

    method VARCHAR(50),

    ip_address VARCHAR(100),

    status VARCHAR(100),

    metadata_json TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

CREATE INDEX IF NOT EXISTS idx_audit_user
ON audit_logs(user_id);

-- ============================================================
-- SCHEDULER JOBS
-- ============================================================

CREATE TABLE IF NOT EXISTS scheduler_jobs (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    job_name VARCHAR(255),

    job_type VARCHAR(255),

    status VARCHAR(100) DEFAULT 'pending',

    last_run TIMESTAMP,

    next_run TIMESTAMP,

    execution_time FLOAT,

    logs TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

-- ============================================================
-- NOTIFICATIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS notifications (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER,

    title VARCHAR(255),

    message TEXT,

    notification_type VARCHAR(100),

    is_read BOOLEAN DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

CREATE INDEX IF NOT EXISTS idx_notification_user
ON notifications(user_id);

-- ============================================================
-- ANALYTICS CACHE
-- ============================================================

CREATE TABLE IF NOT EXISTS analytics_cache (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    cache_key VARCHAR(255) UNIQUE,

    cache_value TEXT,

    expires_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

CREATE INDEX IF NOT EXISTS idx_cache_key
ON analytics_cache(cache_key);

-- ============================================================
-- SYSTEM METRICS
-- ============================================================

CREATE TABLE IF NOT EXISTS system_metrics (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    cpu_usage FLOAT,

    memory_usage FLOAT,

    disk_usage FLOAT,

    active_users INTEGER,

    active_models INTEGER,

    api_requests INTEGER,

    prediction_latency FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

-- ============================================================
-- ROLE PERMISSIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS role_permissions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    role_name VARCHAR(255),

    permission_name VARCHAR(255),

    can_create BOOLEAN DEFAULT 0,

    can_read BOOLEAN DEFAULT 1,

    can_update BOOLEAN DEFAULT 0,

    can_delete BOOLEAN DEFAULT 0

);

CREATE INDEX IF NOT EXISTS idx_role_name
ON role_permissions(role_name);

-- ============================================================
-- ACTIVITY TRACKER
-- ============================================================

CREATE TABLE IF NOT EXISTS activity_tracker (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER,

    activity_type VARCHAR(255),

    activity_data TEXT,

    device VARCHAR(255),

    browser VARCHAR(255),

    location VARCHAR(255),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

CREATE INDEX IF NOT EXISTS idx_activity_user
ON activity_tracker(user_id);

-- ============================================================
-- OPTIONAL VIEW
-- HIGH RISK CUSTOMERS
-- ============================================================

CREATE VIEW IF NOT EXISTS high_risk_customers AS

SELECT

    id,
    customer_uid,
    full_name,
    email,
    churn_probability,
    risk_level,
    industry

FROM customers

WHERE churn_probability >= 0.80;

-- ============================================================
-- OPTIONAL VIEW
-- MODEL PERFORMANCE
-- ============================================================

CREATE VIEW IF NOT EXISTS model_performance_summary AS

SELECT

    model_name,
    industry,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc

FROM model_registry

WHERE is_active = 1;

-- ============================================================
-- OPTIONAL VIEW
-- ACTIVE USERS
-- ============================================================

CREATE VIEW IF NOT EXISTS active_users_view AS

SELECT

    id,
    full_name,
    email,
    company,
    role

FROM users

WHERE is_active = 1;

-- ============================================================
-- SAMPLE ADMIN USER
-- ============================================================

INSERT INTO users (

    full_name,
    email,
    password_hash,
    company,
    role,
    is_active,
    is_admin

)

VALUES (

    'Admin User',
    'admin@churnshield.ai',
    'hashed_password_here',
    'ChurnShield',
    'admin',
    1,
    1

);

-- ============================================================
-- SAMPLE ROLE PERMISSIONS
-- ============================================================

INSERT INTO role_permissions (

    role_name,
    permission_name,
    can_create,
    can_read,
    can_update,
    can_delete

)

VALUES

('admin', 'dashboard', 1, 1, 1, 1),
('admin', 'prediction', 1, 1, 1, 1),
('admin', 'upload', 1, 1, 1, 1),

('manager', 'dashboard', 1, 1, 1, 0),
('manager', 'prediction', 1, 1, 1, 0),

('user', 'dashboard', 0, 1, 0, 0),
('user', 'prediction', 1, 1, 0, 0);

-- ============================================================
-- COMPLETED
-- ============================================================

-- ChurnShield Enterprise Schema Loaded Successfully