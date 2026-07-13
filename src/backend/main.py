import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config import settings
from src.logging_setup import setup_logging
from src.backend.api.router import root_router

# Setup logger before starting the app
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle startup and shutdown routines.
    """
    logger.info(f"Starting {settings.PROJECT_NAME} in environment: {settings.ENV}")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Intelligent Multi-Agent Machine Learning Experimentation & Orchestration Platform Backend",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Set up CORS origins
# Allow all origins in dev mode, restrict or customize in production settings
origins = [
    "http://localhost:8501",  # Streamlit default port
    "http://127.0.0.1:8501",
    "*",                      # Broad access for simplicity of dev/containers
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main api router under settings prefix (default: /api)
app.include_router(root_router, prefix="/api")


@app.get("/health", tags=["system"])
def health_check():
    """
    Returns system status and basic environment settings.
    """
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "environment": settings.ENV,
        "api_prefix": f"/api{settings.API_V1_STR}"
    }


if __name__ == "__main__":
    import uvicorn
    # If file run directly, boot local dev server on configured port
    uvicorn.run(
        "src.backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True
    )
