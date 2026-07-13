from fastapi import APIRouter
from src.backend.api.v1.endpoints import experiments, agents, datasets

api_router = APIRouter()

# Include feature sub-routers
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])

