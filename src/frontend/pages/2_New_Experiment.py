import streamlit as st
from src.frontend.utils.api_client import BackendAPIClient

api_client = BackendAPIClient()

st.set_page_config(page_title="New Experiment - ML Orchestrator", page_icon="🚀", layout="wide")

# Custom Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0c10;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Configure & Trigger New Experiment")
st.write("Set up your pipeline parameter configuration grid below to launch background model fitting simulation.")

# Form to hold experiment parameters
with st.form("experiment_config_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Dataset Configuration")
        experiment_name = st.text_input("Experiment Name", value="XGBoost Housing Run")
        
        dataset_name = st.selectbox(
            "Select Training Dataset",
            ["california_housing.csv", "diabetes_dataset.csv", "churn_data.csv", "custom_upload.csv"]
        )
        
        target_column = st.text_input("Target Target Label (Column Name)", value="median_house_value")
        
    with col2:
        st.subheader("🛠️ Model & Parameters")
        model_type = st.selectbox(
            "Model Algorithm",
            ["xgboost", "random_forest", "svm"],
            index=0
        )
        
        st.write("---")
        st.write("**Hyperparameters Tuning Grid:**")
        
        hyperparameters = {}
        
        # Dynamic inputs depending on model selection
        if model_type == "xgboost":
            learning_rate = st.slider("Learning Rate (eta)", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
            max_depth = st.slider("Max Tree Depth", min_value=3, max_value=15, value=6, step=1)
            n_estimators = st.number_input("Number of Estimators", min_value=50, max_value=2000, value=500, step=50)
            subsample = st.slider("Subsample Ratio", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
            
            hyperparameters = {
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "subsample": subsample
            }
            
        elif model_type == "random_forest":
            n_estimators = st.number_input("Number of Trees", min_value=10, max_value=1000, value=150, step=10)
            max_depth = st.slider("Max Tree Depth", min_value=2, max_value=32, value=12, step=1)
            min_samples_split = st.slider("Min Samples to Split Node", min_value=2, max_value=10, value=2, step=1)
            criterion = st.selectbox("Split Criterion", ["gini", "entropy", "log_loss"])
            
            hyperparameters = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "criterion": criterion
            }
            
        elif model_type == "svm":
            c_val = st.number_input("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0)
            kernel = st.selectbox("Kernel Function", ["rbf", "linear", "poly", "sigmoid"])
            gamma = st.selectbox("Gamma scale", ["scale", "auto"])
            
            hyperparameters = {
                "C": c_val,
                "kernel": kernel,
                "gamma": gamma
            }

    # Submit button
    submitted = st.form_submit_key = st.form_submit_button("Initiate ML Experiment")

if submitted:
    if not experiment_name:
        st.error("Please enter a valid experiment name.")
    elif not target_column:
        st.error("Please specify a target prediction column.")
    else:
        # Prepare post data payload
        payload = {
            "name": experiment_name,
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "dataset_name": dataset_name,
            "target_column": target_column
        }
        
        st.info("Sending configuration parameters to backend orchestrator...")
        response = api_client.create_experiment(payload)
        
        if response:
            st.success(f"🎉 Success! Experiment '{response.get('name')}' created successfully.")
            st.markdown(f"**Run ID generated:** `{response.get('id')}`")
            st.markdown("Navigate to the **Dashboard page** to monitor progress and read train output metrics.")
        else:
            st.error("❌ Failed to initiate experiment. Verify FastAPI service connectivity.")
