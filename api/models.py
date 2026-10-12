from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TopicsRequest(BaseModel):
    topics: list[str]


class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    topics: list[str] = []


class PaperOut(BaseModel):
    id: str
    title: str
    year: Optional[int] = None
    citations: int = 0
    first_author: str = ""
    top_concept: str = ""
    doi: str = ""


class SaveSessionRequest(BaseModel):
    query: str
    paper_ids: list[str] = []
