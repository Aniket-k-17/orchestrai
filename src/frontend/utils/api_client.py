import logging
from typing import List, Dict, Any, Optional
import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class BackendAPIClient:
    """
    API client for communicating with the FastAPI backend.
    """
    def __init__(self, base_url: Optional[str] = None):
        # Fallback order: passed parameter -> settings.BACKEND_URL -> default localhost
        self.base_url = base_url or settings.BACKEND_URL or "http://localhost:8000"
        self.api_prefix = f"{self.base_url}/api/v1"
        self.timeout = httpx.Timeout(10.0, connect=5.0)

    def check_health(self) -> Dict[str, Any]:
        """
        Check health endpoint of the backend.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.warning(f"Backend health check connection failed: {e}")
        return {"status": "unreachable", "error": "Cannot connect to FastAPI backend server."}

    def get_experiments(self) -> List[Dict[str, Any]]:
        """
        Fetches all experiments from backend.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_prefix}/experiments/")
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error fetching experiments: {e}")
        return []

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches details for a single experiment run.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_prefix}/experiments/{experiment_id}")
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error fetching experiment {experiment_id}: {e}")
        return None

    def create_experiment(self, experiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Submits request to start a new experiment training run.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.api_prefix}/experiments/", json=experiment_data)
                if response.status_code == 201:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error creating experiment: {e}")
        return None

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Deletes a specific experiment run metadata.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.delete(f"{self.api_prefix}/experiments/{experiment_id}")
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Error deleting experiment {experiment_id}: {e}")
        return False

    def execute_agent_task(self, prompt: str, session_id: Optional[str] = None, context_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Sends custom LLM execution instructions to the LangChain orchestrator.
        """
        payload = {
            "prompt": prompt,
            "session_id": session_id,
            "context_data": context_data or {}
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{self.api_prefix}/agents/execute", json=payload)
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error executing agent task: {e}")
        return None

    def upload_dataset(self, file_name: str, file_content: bytes) -> Optional[Dict[str, Any]]:
        """
        Uploads a CSV dataset to the FastAPI backend.
        """
        try:
            files = {"file": (file_name, file_content, "text/csv")}
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{self.api_prefix}/datasets/upload", files=files)
                if response.status_code == 201:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error uploading dataset: {e}")
        return None

    def get_datasets(self) -> List[Dict[str, Any]]:
        """
        Retrieves metadata list for all uploaded datasets.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_prefix}/datasets/")
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error listing datasets: {e}")
        return []

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves detailed statistics of a single dataset.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_prefix}/datasets/{dataset_id}")
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error retrieving dataset {dataset_id}: {e}")
        return None

    def get_dataset_preview(self, dataset_id: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        """
        Retrieves row-level previews of a dataset.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_prefix}/datasets/{dataset_id}/preview", params={"limit": limit})
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error previewing dataset {dataset_id}: {e}")
        return None

    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Deletes a dataset metadata and file from backend.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.delete(f"{self.api_prefix}/datasets/{dataset_id}")
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Error deleting dataset {dataset_id}: {e}")
        return False

    def analyze_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Submits request to run the LangChain Dataset Analysis Agent.
        """
        try:
            # Increase timeout since agent call may take up to a minute
            timeout = httpx.Timeout(60.0, connect=5.0)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{self.api_prefix}/datasets/{dataset_id}/analyze")
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error running agent analysis on {dataset_id}: {e}")
        return None

    def clean_dataset(self, dataset_id: str, clean_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Submits data cleaning payload to perform preprocessing on a dataset.
        """
        try:
            timeout = httpx.Timeout(60.0, connect=5.0)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{self.api_prefix}/datasets/{dataset_id}/clean", json=clean_config)
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error executing data cleaning pipeline on {dataset_id}: {e}")
        return None



