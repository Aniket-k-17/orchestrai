import json
import uuid
import time
from typing import Any, Dict
from datetime import datetime
import numpy as np
import pandas as pd


class MLJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy and Pandas data types seamlessly.
    Useful when returning metrics or hyperparameters from backend services.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return super().default(obj)


def generate_id(prefix: str = "run") -> str:
    """
    Generates a unique identifier with a given prefix.
    """
    unique_suffix = uuid.uuid4().hex[:8]
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{unique_suffix}"


def format_duration(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable format.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.1f}s"
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


def clean_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively converts dictionary values to basic Python types using MLJSONEncoder.
    Useful for cleaning up model hyperparameters.
    """
    serialized = json.dumps(params, cls=MLJSONEncoder)
    return json.loads(serialized)
