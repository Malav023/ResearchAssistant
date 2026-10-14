from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from auth_utils import (
    COOKIE_NAME,
    JWT_EXPIRE_MINUTES,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from config import get_config
from db import users_collection
from models import LoginRequest, RegisterRequest, TopicsRequest, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])

COOKIE_KWARGS = dict(
    httponly=True,
    samesite="lax",
    secure=get_config().cookie_secure,  # COOKIE_SECURE=true once served over HTTPS
    max_age=JWT_EXPIRE_MINUTES * 60,
    path="/",
)


def _user_out(doc: dict) -> UserOut:
    return UserOut(
        id=str(doc["_id"]),
        name=doc.get("name", ""),
        email=doc["email"],
        topics=doc.get("topics", []),
    )


async def get_current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    try:
        oid = ObjectId(payload["sub"])
    except (InvalidId, KeyError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid session")
    doc = await users_collection().find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=401, detail="User not found")
    return doc


@router.post("/register", response_model=UserOut)
async def register(body: RegisterRequest, response: Response):
    existing = await users_collection().find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    doc = {
        "name": body.name,
        "email": body.email,
        "password_hash": hash_password(body.password),
        "topics": [],
    }
    result = await users_collection().insert_one(doc)
    doc["_id"] = result.inserted_id

    token = create_access_token(str(doc["_id"]), doc["email"])
    response.set_cookie(COOKIE_NAME, token, **COOKIE_KWARGS)
    return _user_out(doc)


@router.post("/login", response_model=UserOut)
async def login(body: LoginRequest, response: Response):
    doc = await users_collection().find_one({"email": body.email})
    if not doc or not verify_password(body.password, doc["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(str(doc["_id"]), doc["email"])
    response.set_cookie(COOKIE_NAME, token, **COOKIE_KWARGS)
    return _user_out(doc)


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me", response_model=UserOut)
async def me(user=Depends(get_current_user)):
    return _user_out(user)


@router.post("/topics", response_model=UserOut)
async def set_topics(body: TopicsRequest, user=Depends(get_current_user)):
    await users_collection().update_one({"_id": user["_id"]}, {"$set": {"topics": body.topics}})
    user["topics"] = body.topics
    return _user_out(user)
