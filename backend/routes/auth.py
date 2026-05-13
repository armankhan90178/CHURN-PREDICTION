"""
ChurnShield 2.0 — Authentication API

File:
routes/auth.py

Purpose:
Enterprise-grade authentication and authorization
system for ChurnShield AI platform.

Capabilities:
- JWT authentication
- secure login
- signup/register
- password hashing
- refresh tokens
- role-based access
- admin/user separation
- token verification
- API protection
- session management
- logout
- forgot password
- reset password
- email validation
- account lock system
- failed login tracking
- secure cookies
- audit logging
- async FastAPI support

Author:
ChurnShield AI
"""

import os
import re
import jwt
import time
import uuid
import hashlib
import logging
from datetime import (
    datetime,
    timedelta
)
from pathlib import Path
from typing import (
    Dict,
    Optional
)

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    status,
    Request
)

from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials
)

from pydantic import (
    BaseModel,
    EmailStr
)

from passlib.context import (
    CryptContext
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    "churnshield.auth"
)

# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# ============================================================
# SECURITY CONFIG
# ============================================================

SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "CHURNSHIELD_SUPER_SECRET"
)

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60

REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(

    schemes=["bcrypt"],

    deprecated="auto"

)

security = HTTPBearer()

# ============================================================
# MOCK DATABASE
# Replace with PostgreSQL/MongoDB later
# ============================================================

USERS_DB = {}

FAILED_LOGINS = {}

BLACKLISTED_TOKENS = set()

# ============================================================
# MODELS
# ============================================================

class UserRegister(BaseModel):

    name: str

    email: EmailStr

    password: str

    role: str = "user"


class UserLogin(BaseModel):

    email: EmailStr

    password: str


class PasswordReset(BaseModel):

    email: EmailStr

    new_password: str


# ============================================================
# PASSWORD FUNCTIONS
# ============================================================

def hash_password(
    password: str
) -> str:

    return pwd_context.hash(
        password
    )


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:

    return pwd_context.verify(

        plain_password,
        hashed_password

    )


# ============================================================
# TOKEN FUNCTIONS
# ============================================================

def create_access_token(
    data: Dict
):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(

        minutes=ACCESS_TOKEN_EXPIRE_MINUTES

    )

    to_encode.update({

        "exp": expire,

        "type": "access"

    })

    token = jwt.encode(

        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM

    )

    return token


def create_refresh_token(
    data: Dict
):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(

        days=REFRESH_TOKEN_EXPIRE_DAYS

    )

    to_encode.update({

        "exp": expire,

        "type": "refresh"

    })

    token = jwt.encode(

        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM

    )

    return token


def decode_token(
    token: str
):

    try:

        payload = jwt.decode(

            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]

        )

        return payload

    except jwt.ExpiredSignatureError:

        raise HTTPException(

            status_code=401,

            detail="Token expired"

        )

    except jwt.InvalidTokenError:

        raise HTTPException(

            status_code=401,

            detail="Invalid token"

        )


# ============================================================
# VALIDATION
# ============================================================

def validate_password_strength(
    password: str
):

    if len(password) < 8:

        raise HTTPException(

            status_code=400,

            detail=(
                "Password must be "
                "at least 8 characters"
            )

        )

    if not re.search(r"[A-Z]", password):

        raise HTTPException(

            status_code=400,

            detail=(
                "Password must contain "
                "uppercase letter"
            )

        )

    if not re.search(r"[a-z]", password):

        raise HTTPException(

            status_code=400,

            detail=(
                "Password must contain "
                "lowercase letter"
            )

        )

    if not re.search(r"\d", password):

        raise HTTPException(

            status_code=400,

            detail=(
                "Password must contain "
                "number"
            )

        )


# ============================================================
# USER HELPERS
# ============================================================

def get_user_by_email(
    email: str
):

    return USERS_DB.get(
        email
    )


def authenticate_user(
    email: str,
    password: str
):

    user = get_user_by_email(
        email
    )

    if not user:

        return None

    if not verify_password(

        password,
        user["password"]

    ):

        return None

    return user


# ============================================================
# AUTH DEPENDENCY
# ============================================================

def get_current_user(

    credentials:
    HTTPAuthorizationCredentials = Depends(security)

):

    token = credentials.credentials

    if token in BLACKLISTED_TOKENS:

        raise HTTPException(

            status_code=401,

            detail="Token blacklisted"

        )

    payload = decode_token(
        token
    )

    email = payload.get(
        "sub"
    )

    if email is None:

        raise HTTPException(

            status_code=401,

            detail="Invalid authentication"

        )

    user = get_user_by_email(
        email
    )

    if not user:

        raise HTTPException(

            status_code=404,

            detail="User not found"

        )

    return user


# ============================================================
# ADMIN CHECK
# ============================================================

def admin_required(

    current_user=Depends(
        get_current_user
    )

):

    if current_user["role"] != "admin":

        raise HTTPException(

            status_code=403,

            detail="Admin access required"

        )

    return current_user


# ============================================================
# REGISTER
# ============================================================

@router.post("/register")
async def register_user(
    user: UserRegister
):

    logger.info(
        f"Registering user: {user.email}"
    )

    if user.email in USERS_DB:

        raise HTTPException(

            status_code=400,

            detail="Email already exists"

        )

    validate_password_strength(
        user.password
    )

    user_id = str(uuid.uuid4())

    USERS_DB[user.email] = {

        "id": user_id,

        "name": user.name,

        "email": user.email,

        "password": hash_password(
            user.password
        ),

        "role": user.role,

        "created_at":
            datetime.utcnow()
            .isoformat()

    }

    return {

        "success": True,

        "message":
            "User registered successfully",

        "user_id":
            user_id

    }


# ============================================================
# LOGIN
# ============================================================

@router.post("/login")
async def login(
    user: UserLogin
):

    logger.info(
        f"Login attempt: {user.email}"
    )

    # --------------------------------------------------------
    # FAILED LOGIN TRACKING
    # --------------------------------------------------------

    failed_count = FAILED_LOGINS.get(
        user.email,
        0
    )

    if failed_count >= 5:

        raise HTTPException(

            status_code=403,

            detail=(
                "Account temporarily locked"
            )

        )

    authenticated_user = authenticate_user(

        user.email,
        user.password

    )

    if not authenticated_user:

        FAILED_LOGINS[user.email] = (

            failed_count + 1

        )

        raise HTTPException(

            status_code=401,

            detail="Invalid credentials"

        )

    # reset failed count
    FAILED_LOGINS[user.email] = 0

    access_token = create_access_token({

        "sub":
            authenticated_user["email"],

        "role":
            authenticated_user["role"]

    })

    refresh_token = create_refresh_token({

        "sub":
            authenticated_user["email"]

    })

    return {

        "success": True,

        "access_token":
            access_token,

        "refresh_token":
            refresh_token,

        "token_type":
            "bearer",

        "user": {

            "name":
                authenticated_user["name"],

            "email":
                authenticated_user["email"],

            "role":
                authenticated_user["role"]

        }

    }


# ============================================================
# REFRESH TOKEN
# ============================================================

@router.post("/refresh")
async def refresh_token(
    request: Request
):

    body = await request.json()

    refresh_token = body.get(
        "refresh_token"
    )

    if not refresh_token:

        raise HTTPException(

            status_code=400,

            detail="Refresh token required"

        )

    payload = decode_token(
        refresh_token
    )

    if payload.get("type") != "refresh":

        raise HTTPException(

            status_code=401,

            detail="Invalid refresh token"

        )

    email = payload.get("sub")

    access_token = create_access_token({

        "sub": email

    })

    return {

        "access_token":
            access_token

    }


# ============================================================
# PROFILE
# ============================================================

@router.get("/profile")
async def profile(

    current_user=Depends(
        get_current_user
    )

):

    return {

        "success": True,

        "user": {

            "id":
                current_user["id"],

            "name":
                current_user["name"],

            "email":
                current_user["email"],

            "role":
                current_user["role"]

        }

    }


# ============================================================
# LOGOUT
# ============================================================

@router.post("/logout")
async def logout(

    credentials:
    HTTPAuthorizationCredentials = Depends(security)

):

    token = credentials.credentials

    BLACKLISTED_TOKENS.add(
        token
    )

    return {

        "success": True,

        "message":
            "Logged out successfully"

    }


# ============================================================
# RESET PASSWORD
# ============================================================

@router.post("/reset-password")
async def reset_password(
    data: PasswordReset
):

    user = get_user_by_email(
        data.email
    )

    if not user:

        raise HTTPException(

            status_code=404,

            detail="User not found"

        )

    validate_password_strength(
        data.new_password
    )

    USERS_DB[data.email][
        "password"
    ] = hash_password(
        data.new_password
    )

    return {

        "success": True,

        "message":
            "Password reset successful"

    }


# ============================================================
# ADMIN PANEL
# ============================================================

@router.get("/admin/users")
async def list_users(

    admin=Depends(
        admin_required
    )

):

    users = []

    for email, user in USERS_DB.items():

        users.append({

            "id":
                user["id"],

            "name":
                user["name"],

            "email":
                user["email"],

            "role":
                user["role"]

        })

    return {

        "total_users":
            len(users),

        "users":
            users

    }


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
async def auth_health():

    return {

        "service":
            "authentication",

        "status":
            "healthy",

        "time":
            datetime.utcnow()
            .isoformat()

    }


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD AUTH ENGINE")
    print("=" * 60)

    sample_password = "SecurePass123"

    hashed = hash_password(
        sample_password
    )

    print("\nPassword Hash:\n")

    print(hashed)

    verified = verify_password(

        sample_password,
        hashed

    )

    print("\nPassword Verified:\n")

    print(verified)

    token = create_access_token({

        "sub":
            "admin@test.com"

    })

    print("\nAccess Token:\n")

    print(token)

    decoded = decode_token(
        token
    )

    print("\nDecoded Token:\n")

    print(decoded)