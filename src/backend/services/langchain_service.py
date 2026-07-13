import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.config import settings
from src.utils.helpers import generate_id
from src.backend.schemas.agent import AgentTaskResponse, AgentStepDetail, AgentMessage

# Try importing LangChain & Groq components, log warning if missing
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


class LangChainOrchestratorService:
    """
    Service responsible for orchestrating multi-agent tasks and code planning
    using LangChain and the Groq LLM API.
    """
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model_name = "mixtral-8x7b-32768"  # Standard high-speed Groq model
        
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY is not set or using default placeholder. Service will run in mock mode.")
            self.mock_mode = True
        elif not HAS_LANGCHAIN:
            logger.warning("LangChain or LangChain-Groq not fully installed/available. Running in mock mode.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            try:
                self.llm = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=0.2
                )
                logger.info(f"LangChainOrchestratorService initialized with Groq model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize ChatGroq client: {e}. Falling back to mock mode.")
                self.mock_mode = True

    async def execute_task(
        self, 
        prompt: str, 
        session_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> AgentTaskResponse:
        """
        Executes a task prompt, calling the LangChain/Groq stack.
        """
        task_id = generate_id("task")
        start_time = datetime.utcnow()
        
        logger.info(f"Executing agent task {task_id} - Prompt: '{prompt[:50]}...'")
        
        if self.mock_mode:
            return self._execute_mock_task(task_id, prompt, start_time)
            
        try:
            # Simple LangChain prompt and completion chain
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are the Lead ML Architect inside the ML Experiment Orchestrator application. "
                           "Your goal is to guide the user in setting up, preprocessing, tuning, and evaluating models. "
                           "Respond with concrete, structured, and helpful answers."),
                ("user", "{input}")
            ])
            
            chain = chat_prompt | self.llm | StrOutputParser()
            final_output = await chain.ainvoke({"input": prompt})
            
            # Construct a dummy trace steps to simulate orchestrator agent pipeline
            steps = [
                AgentStepDetail(
                    step_number=1,
                    agent_name="PlanningAgent",
                    action="Parsed request prompt and determined execution steps",
                    observation="Understood query. Will formulate response.",
                    timestamp=datetime.utcnow()
                ),
                AgentStepDetail(
                    step_number=2,
                    agent_name="MLArchitectAgent",
                    action="Generated recommendations using LLM",
                    observation="Text completed successfully from ChatGroq.",
                    timestamp=datetime.utcnow()
                )
            ]
            
            chat_history = [
                AgentMessage(role="user", content=prompt, timestamp=start_time),
                AgentMessage(role="assistant", content=final_output, timestamp=datetime.utcnow())
            ]
            
            return AgentTaskResponse(
                task_id=task_id,
                status="completed",
                prompt=prompt,
                final_output=final_output,
                steps=steps,
                chat_history=chat_history,
                created_at=start_time,
                completed_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error during LangChain LLM execution: {e}")
            return AgentTaskResponse(
                task_id=task_id,
                status="failed",
                prompt=prompt,
                final_output=f"Error executing agent pipeline: {str(e)}",
                steps=[
                    AgentStepDetail(
                        step_number=1,
                        agent_name="System",
                        action="Attempted LLM generation",
                        observation=f"Failed with exception: {str(e)}",
                        timestamp=datetime.utcnow()
                    )
                ],
                chat_history=[
                    AgentMessage(role="user", content=prompt, timestamp=start_time),
                    AgentMessage(role="assistant", content=f"Failed to generate response: {str(e)}", timestamp=datetime.utcnow())
                ],
                created_at=start_time,
                completed_at=datetime.utcnow()
            )

    def _execute_mock_task(self, task_id: str, prompt: str, start_time: datetime) -> AgentTaskResponse:
        """
        Executes a mock task with helpful default messages when LLM is unavailable.
        """
        # Mock answers based on keyword triggers
        prompt_lower = prompt.lower()
        if "preprocess" in prompt_lower or "data" in prompt_lower:
            mock_output = (
                "### Mock Agent Response: Data Preprocessing Plan\n\n"
                "To preprocess your dataset, I recommend the following sequence:\n"
                "1. **Imputation**: Handle missing features. Use median values for numerical data, mode for categoricals.\n"
                "2. **Scaling**: Standardize numerical variables using `StandardScaler` from scikit-learn.\n"
                "3. **Categorical Encoding**: Convert fields like model types or categories using `OneHotEncoder`.\n"
                "4. **Feature Selection**: Apply Mutual Information or variance thresholding.\n\n"
                "*Note: Groq API key is not configured. This is a local mock response.*"
            )
        elif "hyperparameter" in prompt_lower or "tune" in prompt_lower:
            mock_output = (
                "### Mock Agent Response: Hyperparameter Optimization\n\n"
                "For XGBoost tuning, I recommend optimizing the following grid:\n"
                "- `max_depth`: [3, 5, 7, 9]\n"
                "- `learning_rate`: [0.01, 0.05, 0.1, 0.2]\n"
                "- `n_estimators`: [100, 200, 500]\n"
                "- `subsample`: [0.7, 0.9, 1.0]\n\n"
                "*Note: Groq API key is not configured. This is a local mock response.*"
            )
        else:
            mock_output = (
                f"### Mock Agent Response\n\n"
                f"You asked: '{prompt}'\n\n"
                "I am the ML Orchestrator Agent. Currently, I am running in **Mock Mode** because the "
                "`GROQ_API_KEY` is either missing or contains the default placeholder in `.env`.\n\n"
                "To enable real LLM generation:\n"
                "1. Create a `.env` file from `.env.example`.\n"
                "2. Set `GROQ_API_KEY=gsk_your_key_here`.\n"
                "3. Restart the backend application."
            )

        steps = [
            AgentStepDetail(
                step_number=1,
                agent_name="MockPlanningAgent",
                action="Intercepted request in Mock Mode",
                observation="Identified prompt key phrases.",
                timestamp=datetime.utcnow()
            ),
            AgentStepDetail(
                step_number=2,
                agent_name="MockResponseAgent",
                action="Built boilerplate ML orchestration output",
                observation="Completed planning simulation.",
                timestamp=datetime.utcnow()
            )
        ]

        chat_history = [
            AgentMessage(role="user", content=prompt, timestamp=start_time),
            AgentMessage(role="assistant", content=mock_output, timestamp=datetime.utcnow())
        ]

        return AgentTaskResponse(
            task_id=task_id,
            status="completed",
            prompt=prompt,
            final_output=mock_output,
            steps=steps,
            chat_history=chat_history,
            created_at=start_time,
            completed_at=datetime.utcnow()
        )
