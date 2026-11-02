import logging
import os
import time

from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from support_rag import answer_question


# Load environment configuration.
load_dotenv()

MAX_QUESTION_LENGTH = int(
    os.getenv("MAX_QUESTION_LENGTH", "1000")
)


# Create a logs directory.
Path("logs").mkdir(exist_ok=True)


# Configure logging.
file_handler = RotatingFileHandler(
    "logs/app.log",
    maxBytes=1_000_000,
    backupCount=3,
    encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        file_handler
    ]
)

logger = logging.getLogger(__name__)


# Request model with input validation.
class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="Customer support question"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value


# Response model.
class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    response_time_ms: float


# Application lifecycle.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Customer Support RAG API starting")
    yield
    logger.info("Customer Support RAG API shutting down")


app = FastAPI(
    title="Customer Support RAG API",
    description=(
        "A FastAPI application for customer support "
        "using Retrieval-Augmented Generation and Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", include_in_schema=False)
def root():
    """
    Redirect the base URL to the interactive API docs.
    """

    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    """
    Check whether the FastAPI application is running.
    """

    return {
        "status": "healthy",
        "service": "Customer Support RAG API"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Accept a customer question and return a RAG answer.
    """

    start_time = time.perf_counter()

    try:
        result = answer_question(request.question)

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        logger.info(
            "Request successful | response_time_ms=%.2f",
            elapsed_ms
        )

        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            response_time_ms=elapsed_ms
        )

    except Exception:
        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        # Keep internal exception details out of the response.
        logger.exception(
            "Request failed | response_time_ms=%.2f",
            elapsed_ms
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The answer service is temporarily unavailable. "
                "Please try again later."
            )
        )