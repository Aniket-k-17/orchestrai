import logging
from fastapi import APIRouter, HTTPException, Depends
from src.backend.schemas.agent import AgentTaskRequest, AgentTaskResponse
from src.backend.services.langchain_service import LangChainOrchestratorService

logger = logging.getLogger(__name__)
router = APIRouter()

# Instantiate service instance
orchestrator_service = LangChainOrchestratorService()


@router.post("/execute", response_model=AgentTaskResponse)
async def execute_agent_task(task_request: AgentTaskRequest):
    """
    Submits a prompt instruction to the multi-agent orchestrator.
    Evaluates context, executes agent planner, and returns final plan/output.
    """
    logger.info(f"Received agent task execution request. Session: {task_request.session_id}")
    try:
        response = await orchestrator_service.execute_task(
            prompt=task_request.prompt,
            session_id=task_request.session_id,
            context_data=task_request.context_data
        )
        return response
    except Exception as e:
        logger.error(f"Error executing agent task in endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal agent server error: {str(e)}"
        )
