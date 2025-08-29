from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import uuid  # For generating unique session IDs
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelName(str, Enum):
    DEEPSEEK_PROVER_V2 = "deepseek/deepseek-prover-v2"
    DEEPSEEK_CHAT = "deepseek-chat"

class QueryInput(BaseModel):
    question: str = Field(..., example="What is LangChain?")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), example="123e4567-e89b-12d3-a456-426614174000")
    model: ModelName = Field(default=ModelName.DEEPSEEK_PROVER_V2, example=ModelName.DEEPSEEK_CHAT)

    def __init__(self, **data):
        super().__init__(**data)
        logging.info(f"New QueryInput created with Session ID: {self.session_id}")

class QueryResponse(BaseModel):
    answer: str = Field(..., example="LangChain is a framework for building applications with LLMs.")
    session_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    model: ModelName = Field(..., example=ModelName.DEEPSEEK_CHAT)

class DocumentInfo(BaseModel):
    id: int = Field(..., example=1)
    filename: str = Field(..., example="example.pdf")
    upload_timestamp: datetime = Field(..., example="2025-08-29T12:34:56")

class DeleteFileRequest(BaseModel):
    file_id: int = Field(..., example=1)