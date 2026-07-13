from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class DatasetCleanRequest(BaseModel):
    remove_duplicates: bool = Field(True, description="Flag indicating whether to drop duplicate rows")
    imputation_strategies: Dict[str, str] = Field(
        default_factory=dict, 
        description="Imputation strategy for specific columns (mean, median, mode, constant, drop, none)"
    )
    encoding_strategies: Dict[str, str] = Field(
        default_factory=dict, 
        description="Encoding strategy for categorical columns (one_hot, label, drop, none)"
    )
    scaling_strategies: Dict[str, str] = Field(
        default_factory=dict, 
        description="Scaling strategy for numerical columns (standard, minmax, robust, none)"
    )


class CleanComparison(BaseModel):
    original_rows: int = Field(..., description="Original row count")
    cleaned_rows: int = Field(..., description="Cleaned row count")
    original_cols: int = Field(..., description="Original column count")
    cleaned_cols: int = Field(..., description="Cleaned column count")
    original_missing: int = Field(..., description="Original missing values sum count")
    cleaned_missing: int = Field(..., description="Cleaned missing values sum count")
    original_duplicates: int = Field(..., description="Original duplicate rows count")
    cleaned_duplicates: int = Field(..., description="Cleaned duplicate rows count")


class DatasetCleanResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the pipeline executed without critical error")
    message: str = Field(..., description="Status summary details message")
    cleaned_dataset_id: str = Field(..., description="Unique ID generated for the new cleaned dataset")
    cleaned_filename: str = Field(..., description="Name of the saved cleaned CSV file")
    comparison: CleanComparison = Field(..., description="Before vs After metrics profile comparisons")
    execution_logs: List[str] = Field(default_factory=list, description="Step-by-step pipeline logging metrics")
