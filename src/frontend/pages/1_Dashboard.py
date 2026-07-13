import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from src.frontend.utils.api_client import BackendAPIClient

api_client = BackendAPIClient()

st.set_page_config(page_title="Dashboard - ML Orchestrator", page_icon="📊", layout="wide")

# Custom CSS for styling tables and metric blocks
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0c10;
    }
    .metric-card {
        background: rgba(31, 40, 56, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #66fcf1;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Experiment Dashboard")
st.write("Monitor existing model runs, hyperparameter grids, and comparative metric reports.")

# Fetch active runs from backend
runs = api_client.get_experiments()

if not runs:
    st.warning("⚠️ No active experiment runs found or backend is unreachable.")
    st.info("💡 Pro-tip: Run the backend and launch an experiment from the 'New Experiment' page.")
else:
    # Build dataframe for analysis
    data_list = []
    for run in runs:
        metrics = run.get("metrics") or {}
        created_at_dt = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) if isinstance(run["created_at"], str) else run["created_at"]
        formatted_date = created_at_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        data_list.append({
            "Run ID": run["id"],
            "Name": run["name"],
            "Model": run["model_type"].upper(),
            "Status": run["status"],
            "Dataset": run["dataset_name"],
            "Target": run["target_column"],
            "RMSE": metrics.get("rmse", None),
            "R2": metrics.get("r2", None),
            "Duration": run["duration"] or "N/A",
            "Created At": formatted_date,
            "raw_run": run
        })
        
    df = pd.DataFrame(data_list)
    
    # Render KPI Metric Cards
    total_runs = len(df)
    completed_runs = len(df[df["Status"] == "completed"])
    running_runs = len(df[df["Status"] == "running"])
    failed_runs = len(df[df["Status"] == "failed"])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_runs}</div><div class="metric-label">Total Runs</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00ff88">{completed_runs}</div><div class="metric-label">Completed</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#33bbee">{running_runs}</div><div class="metric-label">Running</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff3344">{failed_runs}</div><div class="metric-label">Failed</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # Display Runs Table
    st.subheader("Model Running Sessions Log")
    
    # Hide raw run from display table
    display_cols = ["Run ID", "Name", "Model", "Status", "Dataset", "RMSE", "R2", "Duration", "Created At"]
    st.dataframe(df[display_cols], use_container_width=True)
    
    # Detailed inspector section
    st.subheader("🕵️ Deep Inspect Run Details")
    selected_run_id = st.selectbox("Choose a run to inspect logs and parameters:", df["Run ID"].tolist())
    
    if selected_run_id:
        selected_row = df[df["Run ID"] == selected_run_id].iloc[0]
        raw_data = selected_row["raw_run"]
        
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown("### Configured Parameters")
            st.write(f"**Dataset Name:** `{raw_data.get('dataset_name')}`")
            st.write(f"**Target Column:** `{raw_data.get('target_column')}`")
            st.write("**Hyperparameters Grid:**")
            st.json(raw_data.get("hyperparameters", {}))
            
            if raw_data.get("artifacts"):
                st.write("**Generated Model Artifacts:**")
                st.json(raw_data.get("artifacts", {}))
                
        with detail_col2:
            st.markdown("### Console Execution Logs")
            log_text = "\n".join(raw_data.get("logs", []))
            st.text_area("Console Output Stream", log_text, height=220, disabled=True)
            
    # Chart Visualizations (Plotly)
    st.write("---")
    st.subheader("📈 Performance Analysis")
    
    completed_df = df[df["Status"] == "completed"].dropna(subset=["RMSE", "R2"])
    
    if completed_df.empty:
        st.info("No completed runs with valid evaluation metrics found to plot charts yet.")
    else:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_rmse = px.bar(
                completed_df, 
                x="Name", 
                y="RMSE", 
                color="Model",
                title="Root Mean Squared Error (RMSE) - Lower is Better",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_rmse.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rmse, use_container_width=True)
            
        with chart_col2:
            fig_r2 = px.scatter(
                completed_df,
                x="RMSE",
                y="R2",
                color="Model",
                size=[15] * len(completed_df),
                hover_name="Name",
                title="Model Fit Tradeoff: RMSE vs R2 Coefficient",
                template="plotly_dark"
            )
            fig_r2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_r2, use_container_width=True)
