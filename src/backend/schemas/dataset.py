from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class DatasetColumnInfo(BaseModel):
    name: str = Field(..., description="Column header name")
    data_type: str = Field(..., description="Pandas/NumPy data type (e.g. int64, float64, object)")
    null_count: int = Field(..., description="Number of missing values in this column")
    null_percentage: float = Field(..., description="Percentage of missing values in this column")
    is_numerical: bool = Field(..., description="Flag indicating if the column is numerical")
    is_categorical: bool = Field(..., description="Flag indicating if the column is categorical")


class DatasetSummaryResponse(BaseModel):
    id: str = Field(..., description="Unique generated dataset identifier")
    filename: str = Field(..., description="Original name of the uploaded CSV file")
    rows: int = Field(..., description="Total row count")
    columns_count: int = Field(..., description="Total column count")
    size_bytes: int = Field(..., description="File size on disk in bytes")
    duplicate_rows: int = Field(..., description="Count of exact duplicate rows in the dataset")
    columns: List[DatasetColumnInfo] = Field(..., description="List of dataset columns detailed profiling information")
    numerical_columns: List[str] = Field(..., description="List of detected numerical column names")
    categorical_columns: List[str] = Field(..., description="List of detected categorical column names")
    summary_stats: Dict[str, Dict[str, Any]] = Field(..., description="Descriptive statistics mapping (mean, std, min, max, quantiles)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Time uploaded")


class DatasetPreviewResponse(BaseModel):
    id: str = Field(..., description="Unique dataset identifier")
    filename: str = Field(..., description="Original name of the dataset")
    columns: List[str] = Field(..., description="List of columns headers in order")
    data: List[Dict[str, Any]] = Field(..., description="List of records previewing dataset rows")
    total_rows: int = Field(..., description="Total row count of the dataset")


class ColumnAnalysis(BaseModel):
    name: str = Field(..., description="Column header name")
    inferred_type: str = Field(..., description="Inferred ML feature type (e.g. continuous, discrete, nominal, ordinal)")
    description: str = Field(..., description="Brief description of the column semantics")
    anomalies_detected: List[str] = Field(..., description="Detected anomalies specific to this column (outliers, skewed, high nulls)")


class DatasetAnalysisResult(BaseModel):
    target_column_suggestion: str = Field(..., description="Suggested machine learning target column")
    target_rationale: str = Field(..., description="Reasoning for target selection")
    ml_problem_type: str = Field(..., description="Inferred ML task (e.g. Binary Classification, Multi-class Classification, Regression)")
    missing_values_assessment: str = Field(..., description="General evaluation of missing values across the dataset")
    duplicates_assessment: str = Field(..., description="Analysis of duplicated rows")
    outliers_assessment: str = Field(..., description="Analysis of outliers in numerical columns")
    class_imbalance_assessment: Optional[str] = Field(None, description="Analysis of class distribution if classification task")
    dataset_summary: str = Field(..., description="High level summary of the dataset")
    column_analyses: List[ColumnAnalysis] = Field(..., description="Details for each analyzed column")
    preprocessing_recommendations: List[str] = Field(..., description="List of suggested preprocessing tasks")

