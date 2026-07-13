import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.backend.schemas.experiment import ExperimentCreate, ExperimentResponse, ExperimentUpdate
from src.utils.helpers import generate_id, format_duration

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage for experiments simulation
experiments_db: Dict[str, Dict[str, Any]] = {
    "run_1718100000_abc12345": {
        "id": "run_1718100000_abc12345",
        "name": "XGBoost Hyperparameter Tuning",
        "model_type": "xgboost",
        "hyperparameters": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6},
        "dataset_name": "california_housing.csv",
        "target_column": "median_house_value",
        "status": "completed",
        "metrics": {"rmse": 0.482, "r2": 0.812},
        "artifacts": {
            "model_binary": "/artifacts/run_1718100000_abc12345/model.pkl",
            "learning_curve": "/artifacts/run_1718100000_abc12345/loss.png"
        },
        "created_at": datetime.utcnow() - timedelta(hours=2),
        "finished_at": datetime.utcnow() - timedelta(hours=2, minutes=58),
        "duration": "1m 42.0s",
        "logs": [
            "Initializing California Housing Dataset loading...",
            "Splitting data into train/test sets (80/20 split)...",
            "Starting XGBoost model training with GridSearch...",
            "Completed estimator training. Best estimator found: max_depth=6",
            "Evaluating model on test dataset...",
            "Plotting residuals and training history...",
            "Saving model binary to disk. Done."
        ]
    },
    "run_1718105000_def67890": {
        "id": "run_1718105000_def67890",
        "name": "Random Forest Baseline",
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
        "dataset_name": "california_housing.csv",
        "target_column": "median_house_value",
        "status": "completed",
        "metrics": {"rmse": 0.521, "r2": 0.774},
        "artifacts": {
            "model_binary": "/artifacts/run_1718105000_def67890/model.pkl"
        },
        "created_at": datetime.utcnow() - timedelta(hours=1),
        "finished_at": datetime.utcnow() - timedelta(hours=1, minutes=59),
        "duration": "48.20s",
        "logs": [
            "Initializing California Housing Dataset loading...",
            "Splitting data into train/test sets...",
            "Starting Random Forest training...",
            "Finished training 100 decision tree estimators.",
            "Evaluating model on test dataset...",
            "Model artifacts saved. Run complete."
        ]
    }
}


def simulate_experiment_run(experiment_id: str):
    """
    Background worker function that simulates the lifecycle of an ML experiment
    """
    logger.info(f"Background training simulator started for: {experiment_id}")
    try:
        run = experiments_db[experiment_id]
        
        # Step 1: Update status to 'running'
        run["status"] = "running"
        run["logs"].append("Experiment status updated to RUNNING.")
        run["logs"].append("Loading dataset and performing pre-processing checks...")
        
        # Simulate loading duration
        import time
        time.sleep(2)
        
        # Step 2: Simulate train epochs / progress
        run["logs"].append(f"Training started for algorithm: {run['model_type']}")
        for epoch in range(1, 4):
            time.sleep(2)
            loss_val = 0.8 / epoch + 0.1
            run["logs"].append(f"Epoch {epoch}/3 - loss: {loss_val:.4f}")
        
        # Step 3: Compute final metrics and finish
        time.sleep(1.5)
        run["status"] = "completed"
        
        # Specific metrics based on model type
        if run["model_type"] == "xgboost":
            run["metrics"] = {"rmse": 0.456, "r2": 0.831}
        elif run["model_type"] == "random_forest":
            run["metrics"] = {"rmse": 0.505, "r2": 0.791}
        else:
            run["metrics"] = {"rmse": 0.612, "r2": 0.695}
            
        run["artifacts"] = {
            "model_binary": f"/artifacts/{experiment_id}/model.pkl",
            "metrics_report": f"/artifacts/{experiment_id}/report.json"
        }
        
        run["finished_at"] = datetime.utcnow()
        duration_sec = (run["finished_at"] - run["created_at"]).total_seconds()
        run["duration"] = format_duration(duration_sec)
        run["logs"].append("Model evaluation complete. Saving artifacts.")
        run["logs"].append("Experiment COMPLETED successfully.")
        logger.info(f"Background training simulator finished for: {experiment_id}")
        
    except Exception as e:
        logger.error(f"Error during training simulation: {e}")
        if experiment_id in experiments_db:
            run = experiments_db[experiment_id]
            run["status"] = "failed"
            run["finished_at"] = datetime.utcnow()
            run["logs"].append(f"CRITICAL ERROR during training: {str(e)}")


@router.get("/", response_model=List[ExperimentResponse])
def get_experiments():
    """
    List all ML experiments in memory.
    """
    logger.info("Fetching all experiments")
    return list(experiments_db.values())


@router.post("/", response_model=ExperimentResponse, status_code=201)
def create_experiment(experiment_in: ExperimentCreate, background_tasks: BackgroundTasks):
    """
    Initiates an ML experiment run (launches simulation in the background).
    """
    experiment_id = generate_id("run")
    logger.info(f"Creating new experiment: {experiment_in.name} with ID: {experiment_id}")
    
    new_run = {
        "id": experiment_id,
        "name": experiment_in.name,
        "model_type": experiment_in.model_type,
        "hyperparameters": experiment_in.hyperparameters,
        "dataset_name": experiment_in.dataset_name,
        "target_column": experiment_in.target_column,
        "status": "pending",
        "metrics": {},
        "artifacts": {},
        "created_at": datetime.utcnow(),
        "finished_at": None,
        "duration": None,
        "logs": [
            f"Experiment generated with ID: {experiment_id}",
            "Queued for processing. Awaiting worker execution..."
        ]
    }
    
    experiments_db[experiment_id] = new_run
    
    # Enqueue background task simulator
    background_tasks.add_task(simulate_experiment_run, experiment_id)
    
    return new_run


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: str):
    """
    Retrieve details of a specific ML experiment run.
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment run not found")
    return experiments_db[experiment_id]


@router.delete("/{experiment_id}", status_code=200)
def delete_experiment(experiment_id: str):
    """
    Remove an experiment run from tracking.
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail="Experiment run not found")
    del experiments_db[experiment_id]
    return {"message": f"Experiment {experiment_id} successfully deleted"}
