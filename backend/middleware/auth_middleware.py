"""
ChurnShield 2.0 — Authentication Middleware

File:
middleware/auth_middleware.py

Purpose:
Enterprise-grade JWT authentication
and authorization middleware.

Capabilities:
- JWT authentication
- access token validation
- role-based authorization
- API key authentication
- token expiry handling
- secure user extraction
- FastAPI dependency injection
- admin-only protection
- enterprise SaaS security
- audit logging hooks
- request-level auth validation

Author:
ChurnShield AI
"""

import os
import jwt
import logging

from datetime import (
    datetime,
    timedelta
)

from typing import (
    Optional,
    Dict,
    Any
)

from fastapi import (

    Request,
    HTTPException,
    Depends,
    status

)

from fastapi.security import (

    HTTPBearer,
    HTTPAuthorizationCredentials

)

from sqlalchemy.orm import Session

from db.database import get_db

from db.models import (

    User,
    ApiKey

)

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "auth_middleware"
)

logging.basicConfig(
    level=logging.INFO
)

# ============================================================
# CONFIG
# ============================================================

JWT_SECRET = os.getenv(
    "JWT_SECRET",
    "SUPER_SECRET_CHURNSHIELD_KEY"
)

JWT_ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

security = HTTPBearer()

# ============================================================
# TOKEN CREATION
# ============================================================

def create_access_token(

    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None

):

    """
    Create JWT access token
    """

    to_encode = data.copy()

    expire = (

        datetime.utcnow()

        +

        (
            expires_delta

            or

            timedelta(
                minutes=ACCESS_TOKEN_EXPIRE_MINUTES
            )
        )

    )

    to_encode.update({

        "exp": expire,

        "iat": datetime.utcnow()

    })

    encoded_jwt = jwt.encode(

        to_encode,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM

    )

    return encoded_jwt

# ============================================================
# VERIFY TOKEN
# ============================================================

def verify_token(

    token: str

):

    """
    Verify JWT token
    """

    try:

        payload = jwt.decode(

            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]

        )

        return payload

    except jwt.ExpiredSignatureError:

        raise HTTPException(

            status_code=status.HTTP_401_UNAUTHORIZED,

            detail="Token expired"

        )

    except jwt.InvalidTokenError:

        raise HTTPException(

            status_code=status.HTTP_401_UNAUTHORIZED,

            detail="Invalid token"

        )

# ============================================================
# GET CURRENT USER
# ============================================================

async def get_current_user(

    credentials: HTTPAuthorizationCredentials = Depends(security),

    db: Session = Depends(get_db)

):

    """
    Extract current user from JWT
    """

    token = credentials.credentials

    payload = verify_token(token)

    user_id = payload.get("user_id")

    if not user_id:

        raise HTTPException(

            status_code=401,

            detail="Invalid authentication"

        )

    user = (

        db.query(User)

        .filter(User.id == user_id)

        .first()

    )

    if not user:

        raise HTTPException(

            status_code=404,

            detail="User not found"

        )

    if not user.is_active:

        raise HTTPException(

            status_code=403,

            detail="User account disabled"

        )

    return user

# ============================================================
# OPTIONAL AUTH
# ============================================================

async def get_optional_user(

    request: Request,
    db: Session = Depends(get_db)

):

    """
    Optional auth middleware
    """

    auth_header = request.headers.get(
        "Authorization"
    )

    if not auth_header:

        return None

    try:

        token = auth_header.split(" ")[1]

        payload = verify_token(token)

        user_id = payload.get("user_id")

        if not user_id:

            return None

        user = (

            db.query(User)

            .filter(User.id == user_id)

            .first()

        )

        return user

    except Exception:

        return None

# ============================================================
# ADMIN CHECK
# ============================================================

async def admin_required(

    current_user: User = Depends(
        get_current_user
    )

):

    """
    Admin-only access
    """

    if not current_user.is_admin:

        raise HTTPException(

            status_code=403,

            detail="Admin access required"

        )

    return current_user

# ============================================================
# ROLE CHECK
# ============================================================

def role_required(
    allowed_roles
):

    """
    Role-based authorization
    """

    async def checker(

        current_user: User = Depends(
            get_current_user
        )

    ):

        if current_user.role not in allowed_roles:

            raise HTTPException(

                status_code=403,

                detail="Permission denied"

            )

        return current_user

    return checker

# ============================================================
# API KEY AUTH
# ============================================================

async def verify_api_key(

    request: Request,
    db: Session = Depends(get_db)

):

    """
    Validate API key
    """

    api_key = request.headers.get(
        "X-API-KEY"
    )

    if not api_key:

        raise HTTPException(

            status_code=401,

            detail="API key missing"

        )

    db_key = (

        db.query(ApiKey)

        .filter(ApiKey.api_key == api_key)

        .filter(ApiKey.is_active == True)

        .first()

    )

    if not db_key:

        raise HTTPException(

            status_code=401,

            detail="Invalid API key"

        )

    # update usage count
    db_key.requests_count += 1

    db_key.last_used = datetime.utcnow()

    db.commit()

    return db_key

# ============================================================
# REQUEST LOGGER
# ============================================================

async def auth_logger(

    request: Request,
    current_user: Optional[User] = Depends(
        get_optional_user
    )

):

    """
    Request auth logging
    """

    try:

        logger.info({

            "path": request.url.path,

            "method": request.method,

            "user":

                current_user.email

                if current_user

                else "anonymous",

            "timestamp":

                datetime.utcnow().isoformat()

        })

    except Exception as e:

        logger.error(

            f"Logging error: {str(e)}"

        )

# ============================================================
# TOKEN REFRESH
# ============================================================

def refresh_access_token(

    old_payload: Dict[str, Any]

):

    """
    Refresh JWT token
    """

    new_payload = {

        "user_id":
            old_payload.get("user_id"),

        "email":
            old_payload.get("email"),

        "role":
            old_payload.get("role")

    }

    return create_access_token(
        new_payload
    )

# ============================================================
# TOKEN BLACKLIST CHECK
# ============================================================

BLACKLISTED_TOKENS = set()

def blacklist_token(
    token: str
):

    """
    Blacklist token
    """

    BLACKLISTED_TOKENS.add(token)

def is_blacklisted(
    token: str
):

    """
    Check blacklist
    """

    return token in BLACKLISTED_TOKENS

# ============================================================
# SAFE TOKEN VERIFY
# ============================================================

def safe_verify_token(
    token: str
):

    """
    Verify token securely
    """

    if is_blacklisted(token):

        raise HTTPException(

            status_code=401,

            detail="Token revoked"

        )

    return verify_token(token)

# ============================================================
# USER SERIALIZER
# ============================================================

def serialize_user(
    user: User
):

    """
    Safe user serializer
    """

    return {

        "id": user.id,

        "name": user.full_name,

        "email": user.email,

        "company": user.company,

        "role": user.role,

        "is_active": user.is_active,

        "created_at":

            str(user.created_at)

    }

# ============================================================
# HEALTH CHECK
# ============================================================

def auth_health_check():

    """
    Auth middleware status
    """

    return {

        "status": "healthy",

        "jwt_algorithm": JWT_ALGORITHM,

        "token_expiry_minutes":
            ACCESS_TOKEN_EXPIRE_MINUTES,

        "blacklisted_tokens":
            len(BLACKLISTED_TOKENS)

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD AUTH MIDDLEWARE")
    print("=" * 60)

    sample_payload = {

        "user_id": 1,

        "email": "admin@churnshield.ai",

        "role": "admin"

    }

    token = create_access_token(
        sample_payload
    )

    print("\nGenerated JWT:\n")

    print(token)

    print("\nVerifying token...\n")

    decoded = verify_token(token)

    print(decoded)

    print("\n")
    print("=" * 60)
    print("AUTH SYSTEM READY")
    print("=" * 60)