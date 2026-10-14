from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from config import get_config

_cfg = get_config()
JWT_SECRET = _cfg.jwt_secret
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = _cfg.jwt_expire_minutes
COOKIE_NAME = "ra_session"


def hash_password(password: str) -> str:
    # bcrypt works on bytes and has a 72-byte ceiling
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8")[:72], hashed_password.encode("utf-8"))


def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "email": email, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None
