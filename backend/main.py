"""
ChurnShield 2.0 — Main FastAPI Application
Entry point for the entire backend.
Starts server, loads models, registers all routes, starts scheduler.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from config import (
    MODELS_DIR, DB_PATH, ANTHROPIC_API_KEY,
    DAILY_REFRESH_HOUR, WEEKLY_REPORT_DAY
)
from db.database import init_db, seed_india_calendar
from ml.predictor import ModelRegistry
from scheduler.jobs import start_scheduler, shutdown_scheduler

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("churnshield.main")


# ─────────────────────────────────────────────
# APPLICATION LIFESPAN — Startup and Shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup: initialize DB, load models, start scheduler.
    Runs on shutdown: clean up resources.
    """
    logger.info("=" * 60)
    logger.info("ChurnShield 2.0 — Starting Up")
    logger.info("=" * 60)

    # 1. Initialize SQLite database and create all tables
    logger.info("Initializing database...")
    init_db()
    seed_india_calendar()
    logger.info("Database ready ✅")

    # 2. Load any pre-trained models into memory registry
    logger.info("Loading ML model registry...")
    ModelRegistry.load_all_from_disk(MODELS_DIR)
    logger.info(f"Models loaded: {ModelRegistry.count()} models in registry ✅")

    # 3. Validate API key is set
    if ANTHROPIC_API_KEY == "your-claude-api-key-here":
        logger.warning("⚠️  ANTHROPIC_API_KEY not set — LLM features will use fallback")
    else:
        logger.info("Claude API key found ✅")

    # 4. Start background scheduler
    logger.info("Starting background scheduler...")
    start_scheduler()
    logger.info("Scheduler running ✅")

    logger.info("=" * 60)
    logger.info("ChurnShield 2.0 — Ready to serve requests 🚀")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown sequence
    logger.info("ChurnShield shutting down...")
    shutdown_scheduler()
    logger.info("Shutdown complete.")


# ─────────────────────────────────────────────
# CREATE FASTAPI APPLICATION
# ─────────────────────────────────────────────
app = FastAPI(
    title="ChurnShield 2.0 API",
    description="Universal Churn Intelligence Platform — Predict, Explain, Retain",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────

# CORS — Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ─────────────────────────────────────────────
# REGISTER ALL ROUTE FILES
# ─────────────────────────────────────────────
from routes.search import router as search_router
from routes.upload import router as upload_router
from routes.customers import router as customers_router
from routes.analytics import router as analytics_router
from routes.export import router as export_router

app.include_router(search_router,    prefix="/search",    tags=["Universal Search"])
app.include_router(upload_router,    prefix="/upload",    tags=["Own Data Upload"])
app.include_router(customers_router, prefix="/customers", tags=["Customers"])
app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
app.include_router(export_router,    prefix="/export",    tags=["Export"])


# ─────────────────────────────────────────────
# SYSTEM ROUTES (defined directly in main.py)
# ─────────────────────────────────────────────
from fastapi.responses import JSONResponse
from db.database import get_db
from config import INDUSTRY_FIELDS, INDIA_CALENDAR
from datetime import datetime


@app.get("/", tags=["System"])
async def root():
    """API root — confirms the service is live."""
    return {
        "service": "ChurnShield 2.0",
        "status": "running",
        "version": "2.0.0",
        "message": "Universal Churn Intelligence Platform",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
async def health_check():
    """
    Deep health check — verifies DB connection, model registry,
    scheduler status, and API key presence.
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "unknown",
        "models_loaded": ModelRegistry.count(),
        "scheduler": "running",
        "claude_api": "configured" if ANTHROPIC_API_KEY != "your-claude-api-key-here" else "not_configured",
    }

    # Check DB
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        health["database"] = "connected"
    except Exception as e:
        health["database"] = f"error: {str(e)}"
        health["status"] = "degraded"

    return health


@app.get("/fields", tags=["System"])
async def get_supported_fields():
    """
    Returns all 100+ supported industry fields.
    Frontend uses this to show autocomplete suggestions.
    """
    fields = []
    for field_name, config in INDUSTRY_FIELDS.items():
        fields.append({
            "name": field_name,
            "category": config["category"],
            "display_name": field_name.title(),
        })

    # Group by category for better UI display
    categories = {}
    for f in fields:
        cat = f["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(f["name"])

    return {
        "total_fields": len(fields),
        "fields": fields,
        "by_category": categories,
    }


@app.get("/india/calendar", tags=["System"])
async def get_india_calendar():
    """
    Returns current month's India business calendar events
    and their effect on churn score adjustment.
    """
    current_month = datetime.now().month
    current_event = INDIA_CALENDAR.get(current_month, {})

    return {
        "current_month": current_month,
        "month_name": datetime.now().strftime("%B"),
        "event": current_event.get("event", "Normal Month"),
        "adjustment_multiplier": current_event.get("adjustment", 1.0),
        "note": current_event.get("note", "No special adjustments"),
        "effect_percent": round((1 - current_event.get("adjustment", 1.0)) * 100),
        "full_calendar": INDIA_CALENDAR,
    }


@app.get("/india/calendar/full", tags=["System"])
async def get_full_calendar():
    """Returns the complete India business calendar for all 12 months."""
    return {
        "calendar": INDIA_CALENDAR,
        "description": "Monthly adjustment multipliers based on Indian business events",
    }


# ─────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────
from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception across the entire app.
    Returns clean error JSON instead of crashing.
    """
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Something went wrong. Our team has been notified.",
            "detail": str(exc) if app.debug else "Contact support",
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "message": f"Endpoint {request.url.path} does not exist",
            "available_docs": "/docs",
        },
    )


# ─────────────────────────────────────────────
# RUN DIRECTLY (for development)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,               # Auto-reload on code changes
        log_level="info",
        workers=1,                 # Single worker for SQLite compatibility
    )