from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class AgentMessage(BaseModel):
    role: str = Field(..., example="assistant", description="Sender role (e.g. system, user, assistant, agent)")
    content: str = Field(..., description="Message string content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time sent")


class AgentTaskRequest(BaseModel):
    prompt: str = Field(..., example="Analyze dataset and propose a preprocessing plan", description="Instruction or task query for the LLM agent orchestrator")
    session_id: Optional[str] = Field(None, description="Optional ID to continue an ongoing session or chat thread")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Optional metadata or variables to inject into prompt context")


class AgentStepDetail(BaseModel):
    step_number: int = Field(..., description="Order of step in agent execution plan")
    agent_name: str = Field(..., description="Sub-agent that executed this step (e.g. DataAnalyzer, HyperparameterOptimizer)")
    action: str = Field(..., description="Description of the action taken")
    observation: str = Field(..., description="Result or observation from the action")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentTaskResponse(BaseModel):
    task_id: str = Field(..., description="Identifier for this orchestration run")
    status: str = Field(..., example="completed", description="Task execution status (running, completed, failed)")
    prompt: str = Field(..., description="Original task prompt")
    final_output: str = Field(..., description="Text summary of response from the agent")
    steps: List[AgentStepDetail] = Field(default_factory=list, description="Sub-agent trace steps execution history")
    chat_history: List[AgentMessage] = Field(default_factory=list, description="Associated message feed for user UI display")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
