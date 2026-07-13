import os
import shutil
import logging
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from src.backend.schemas.dataset import DatasetColumnInfo, DatasetSummaryResponse, DatasetPreviewResponse, DatasetAnalysisResult
from src.backend.schemas.clean import DatasetCleanRequest, DatasetCleanResponse, CleanComparison
from src.utils.helpers import generate_id
from src.backend.services.dataset_agent import DatasetAnalysisAgentService
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

agent_service = DatasetAnalysisAgentService()



logger = logging.getLogger(__name__)
router = APIRouter()

# Data upload folder setup
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
    "data", 
    "uploads"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory database of uploaded datasets metadata
datasets_db: Dict[str, Dict[str, Any]] = {}


def generate_mock_housing_dataset():
    """
    Generates and saves a default california housing dataset on startup for immediate user preview.
    """
    mock_id = "dataset_california_housing"
    mock_filename = "california_housing_sample.csv"
    mock_filepath = os.path.join(UPLOAD_DIR, mock_filename)
    
    if os.path.exists(mock_filepath) and mock_id in datasets_db:
        return
        
    try:
        # Create a mock dataframe with varied columns
        np.random.seed(42)
        rows = 150
        data = {
            "longitude": np.random.uniform(-124.3, -114.3, rows),
            "latitude": np.random.uniform(32.5, 41.9, rows),
            "housing_median_age": np.random.randint(1, 52, rows).astype(float),
            "total_rooms": np.random.randint(100, 5000, rows).astype(float),
            "total_bedrooms": np.random.randint(20, 1000, rows).astype(float),
            "population": np.random.randint(50, 3000, rows).astype(float),
            "households": np.random.randint(10, 1200, rows).astype(float),
            "median_income": np.random.uniform(1.5, 15.0, rows),
            "median_house_value": np.random.uniform(50000, 500000, rows),
            "ocean_proximity": np.random.choice(["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"], rows)
        }
        
        # Inject some nulls to show missing value detection
        for _ in range(5):
            data["total_bedrooms"][np.random.randint(0, rows)] = np.nan
            data["median_income"][np.random.randint(0, rows)] = np.nan
            
        df = pd.DataFrame(data)
        
        # Duplicate a few rows to show duplicate row detection
        df = pd.concat([df, df.iloc[:3]], ignore_index=True)
        
        df.to_csv(mock_filepath, index=False)
        profile_and_store_dataset(mock_id, mock_filename, mock_filepath)
        logger.info("Generated default California Housing dataset preview.")
    except Exception as e:
        logger.error(f"Error generating mock housing dataset: {e}")


def profile_and_store_dataset(dataset_id: str, filename: str, filepath: str):
    """
    Parses a CSV file using Pandas, extracts data profiles, and stores metadata in DB.
    """
    df = pd.read_csv(filepath)
    size_bytes = os.path.getsize(filepath)
    rows_count, cols_count = df.shape
    
    duplicate_rows = int(df.duplicated().sum())
    
    columns_info = []
    numerical_columns = []
    categorical_columns = []
    
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = float((null_count / rows_count) * 100)
        is_num = bool(pd.api.types.is_numeric_dtype(df[col]))
        
        columns_info.append(
            DatasetColumnInfo(
                name=str(col),
                data_type=str(df[col].dtype),
                null_count=null_count,
                null_percentage=null_pct,
                is_numerical=is_num,
                is_categorical=not is_num
            )
        )
        
        if is_num:
            numerical_columns.append(str(col))
        else:
            categorical_columns.append(str(col))
            
    # Compute descriptive statistics, replace NaN with None for JSON compliance
    desc_df = df.describe(include="all")
    desc_clean = desc_df.replace({np.nan: None}).to_dict()
    
    datasets_db[dataset_id] = {
        "id": dataset_id,
        "filename": filename,
        "filepath": filepath,
        "rows": rows_count,
        "columns_count": cols_count,
        "size_bytes": size_bytes,
        "duplicate_rows": duplicate_rows,
        "columns": columns_info,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "summary_stats": desc_clean,
        "created_at": datetime.utcnow()
    }


# Initialize default dataset on load
generate_mock_housing_dataset()


@router.post("/upload", response_model=DatasetSummaryResponse, status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accepts CSV uploads, saves files in workspace, profiles statistics, and stores metadata.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV datasets are currently supported.")
        
    dataset_id = generate_id("dataset")
    filepath = os.path.join(UPLOAD_DIR, f"{dataset_id}_{file.filename}")
    
    logger.info(f"Saving uploaded file {file.filename} as {dataset_id}")
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        profile_and_store_dataset(dataset_id, file.filename, filepath)
        return datasets_db[dataset_id]
        
    except Exception as e:
        logger.error(f"Error handling CSV upload: {e}")
        # Clean up files if failed
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Failed to process dataset file: {str(e)}")


@router.get("/", response_model=List[DatasetSummaryResponse])
def list_datasets():
    """
    List details of all uploaded datasets.
    """
    return list(datasets_db.values())


@router.get("/{dataset_id}", response_model=DatasetSummaryResponse)
def get_dataset(dataset_id: str):
    """
    Retrieves metadata summary for a specific dataset ID.
    """
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return datasets_db[dataset_id]


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_dataset(dataset_id: str, limit: int = Query(50, ge=1, le=10000)):
    """
    Returns the first N rows of a dataset as a list of dictionaries.
    """
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    db_entry = datasets_db[dataset_id]
    filepath = db_entry["filepath"]
    
    try:
        # Load preview slice
        df = pd.read_csv(filepath, nrows=limit)
        # Handle NaN values to avoid JSON decoding crashes
        df_clean = df.replace({np.nan: None})
        records = df_clean.to_dict(orient="records")
        
        return DatasetPreviewResponse(
            id=dataset_id,
            filename=db_entry["filename"],
            columns=list(df.columns),
            data=records,
            total_rows=db_entry["rows"]
        )
    except Exception as e:
        logger.error(f"Error previewing dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading dataset preview: {str(e)}")


@router.delete("/{dataset_id}", status_code=200)
def delete_dataset(dataset_id: str):
    """
    Deletes the dataset file from disk and deletes database record.
    """
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    db_entry = datasets_db[dataset_id]
    filepath = db_entry["filepath"]
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
        del datasets_db[dataset_id]
        logger.info(f"Deleted dataset {dataset_id} successfully.")
        return {"message": f"Dataset {dataset_id} deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.post("/{dataset_id}/analyze", response_model=DatasetAnalysisResult)
async def analyze_dataset(dataset_id: str):
    """
    Submits dataset statistics and sample records to the LLM dataset analyst agent.
    Returns structured analysis recommendations.
    """
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    db_entry = datasets_db[dataset_id]
    
    try:
        # Load sample for parsing (limit to 50 rows for prompt safety)
        preview_res = preview_dataset(dataset_id, limit=50)
        result = await agent_service.analyze_dataset(db_entry, preview_res.model_dump())
        return result
    except Exception as e:
        logger.error(f"Error running agent analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze dataset: {str(e)}")


@router.post("/{dataset_id}/clean", response_model=DatasetCleanResponse)
def clean_dataset(dataset_id: str, clean_in: DatasetCleanRequest):
    """
    Applies preprocessing operations (duplicate dropping, missing values imputation,
    categorical column encoding, and numerical attributes scaling) on the target dataset.
    Saves the cleaned dataset as a new dataset in the workspace.
    """
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    db_entry = datasets_db[dataset_id]
    filepath = db_entry["filepath"]
    filename = db_entry["filename"]
    
    logs = []
    logs.append(f"Starting Data Cleaning Pipeline for '{filename}'...")
    
    try:
        df = pd.read_csv(filepath)
        
        # Record before metrics
        orig_rows, orig_cols = df.shape
        orig_missing = int(df.isnull().sum().sum())
        orig_duplicates = int(df.duplicated().sum())
        
        # 1. Remove Duplicates
        if clean_in.remove_duplicates:
            dup_count = int(df.duplicated().sum())
            if dup_count > 0:
                df = df.drop_duplicates().reset_index(drop=True)
                logs.append(f"Removed {dup_count} duplicate rows.")
            else:
                logs.append("No duplicate rows detected to remove.")
                
        # 2. Missing Value Imputation
        for col, strategy in clean_in.imputation_strategies.items():
            if col not in df.columns:
                continue
            null_count = int(df[col].isnull().sum())
            if null_count == 0 or strategy == "none":
                continue
                
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
            elif strategy == "constant":
                fill_val = 0 if pd.api.types.is_numeric_dtype(df[col]) else "missing"
                df[col] = df[col].fillna(fill_val)
            elif strategy == "drop":
                df = df.dropna(subset=[col]).reset_index(drop=True)
                
            logs.append(f"Imputed {null_count} missing values in '{col}' using '{strategy}' strategy.")
            
        # 3. Scale Numerical Columns
        for col, strategy in clean_in.scaling_strategies.items():
            if col not in df.columns or strategy == "none":
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                logs.append(f"Warning: Cannot scale non-numerical column '{col}' using '{strategy}' scaling. Skipped.")
                continue
                
            # Handle NaNs in column if any remain before scaling
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                
            if strategy == "standard":
                df[col] = StandardScaler().fit_transform(df[[col]])
            elif strategy == "minmax":
                df[col] = MinMaxScaler().fit_transform(df[[col]])
            elif strategy == "robust":
                df[col] = RobustScaler().fit_transform(df[[col]])
                
            logs.append(f"Scaled column '{col}' using '{strategy}' scaler.")
            
        # 4. Encode Categorical Columns
        for col, strategy in list(clean_in.encoding_strategies.items()):
            if col not in df.columns or strategy == "none":
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                logs.append(f"Warning: Cannot encode numerical column '{col}' using '{strategy}' encoding. Skipped.")
                continue
                
            if strategy == "one_hot":
                # Fill NaNs first if any remain
                if df[col].isnull().any():
                    df[col] = df[col].fillna("missing")
                    
                # One-hot encode using pandas
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                logs.append(f"One-hot encoded categorical column '{col}'. Added {dummies.shape[1]} features.")
            elif strategy == "label":
                df[col] = df[col].astype("category").cat.codes
                logs.append(f"Label encoded categorical column '{col}' (converted labels to integer codes).")
            elif strategy == "drop":
                df = df.drop(columns=[col])
                logs.append(f"Dropped categorical column '{col}'.")
                
        # Record final metrics
        clean_rows, clean_cols = df.shape
        clean_missing = int(df.isnull().sum().sum())
        clean_duplicates = int(df.duplicated().sum())
        
        # Save new cleaned dataset
        cleaned_id = generate_id("dataset_cleaned")
        cleaned_filename = f"cleaned_{filename}"
        cleaned_filepath = os.path.join(UPLOAD_DIR, f"{cleaned_id}_{cleaned_filename}")
        
        df.to_csv(cleaned_filepath, index=False)
        
        # Profile and register new cleaned dataset in DB
        profile_and_store_dataset(cleaned_id, cleaned_filename, cleaned_filepath)
        
        logs.append(f"Cleaned dataset successfully saved as '{cleaned_filename}' with ID: {cleaned_id}")
        
        comparison = CleanComparison(
            original_rows=orig_rows,
            cleaned_rows=clean_rows,
            original_cols=orig_cols,
            cleaned_cols=clean_cols,
            original_missing=orig_missing,
            cleaned_missing=clean_missing,
            original_duplicates=orig_duplicates,
            cleaned_duplicates=clean_duplicates
        )
        
        return DatasetCleanResponse(
            success=True,
            message="Dataset cleaning pipeline executed successfully.",
            cleaned_dataset_id=cleaned_id,
            cleaned_filename=cleaned_filename,
            comparison=comparison,
            execution_logs=logs
        )
        
    except Exception as e:
        logger.error(f"Error in cleaning execution: {e}")
        raise HTTPException(status_code=500, detail=f"Data cleaning pipeline crashed: {str(e)}")


