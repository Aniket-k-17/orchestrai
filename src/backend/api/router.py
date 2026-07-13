from fastapi import APIRouter
from src.backend.api.v1.router import api_router as v1_router

root_router = APIRouter()

# Mount the v1 router under '/api/v1' (or dynamic prefix if preferred)
root_router.include_router(v1_router, prefix="/v1")
