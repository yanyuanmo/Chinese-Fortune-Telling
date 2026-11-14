from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: str = Field(default="kimi-k3.1-pro")

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int