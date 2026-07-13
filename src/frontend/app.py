import streamlit as st
import logging
from src.frontend.utils.api_client import BackendAPIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize backend api client
api_client = BackendAPIClient()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ML Experiment Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Injection for Glassmorphism & High-Quality Design
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@300;500&display=swap');
    
    /* Main Layout Styling */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    /* Elegant Title Header */
    .title-banner {
        background: linear-gradient(135deg, #1f4068 0%, #162447 50%, #0f1a36 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
    }
    .title-banner h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(45deg, #66fcf1 0%, #45a29e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    .title-banner p {
        font-size: 1.1rem;
        color: #8b9bb4;
        margin: 0;
    }
    
    /* Premium Styled Card Container */
    .feature-card {
        background: rgba(31, 40, 56, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        height: 100%;
        margin-bottom: 1rem;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(102, 252, 241, 0.08);
        border: 1px solid rgba(102, 252, 241, 0.2);
    }
    .feature-card h3 {
        color: #66fcf1;
        margin-top: 0;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .feature-card p {
        color: #9aa8bc;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Glowing Indicator Dots */
    .indicator-container {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        padding: 6px 12px;
        border-radius: 20px;
        width: fit-content;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-green {
        background-color: #00ff88;
        box-shadow: 0 0 10px #00ff88;
    }
    .dot-red {
        background-color: #ff3344;
        box-shadow: 0 0 10px #ff3344;
    }
    .status-text {
        font-size: 0.85rem;
        font-weight: 600;
        color: #c5c6c7;
    }
</style>
""", unsafe_allow_html=True)

# Main Banner Layout
st.markdown("""
<div class="title-banner">
    <h1>ML Experiment Orchestrator</h1>
    <p>A production-ready platform to run ML pipelines, manage artifacts, and orchestrate intelligent agents using FastAPI, Streamlit, and LangChain.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar System Health Status
st.sidebar.title("Configuration")
health_data = api_client.check_health()

if health_data.get("status") == "healthy":
    st.sidebar.markdown("""
    <div class="indicator-container">
        <span class="dot dot-green"></span>
        <span class="status-text">Backend: Connected</span>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.info(f"⚡ Environment: **{health_data.get('environment', 'unknown').upper()}**")
else:
    st.sidebar.markdown("""
    <div class="indicator-container">
        <span class="dot dot-red"></span>
        <span class="status-text">Backend: Disconnected</span>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.warning("⚠️ FastAPI Backend is unreachable. Please verify settings or boot server.")

# Platform Info Body
st.header("Platform Ecosystem Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Experiment Dashboard</h3>
        <p>Monitor ongoing model training runs in real time, inspect hyperparameter settings, view logs dynamically, and compare metrics (e.g. RMSE, R2) with interactive visual charts.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🚀 Experiment Runner</h3>
        <p>Trigger modular machine learning pipeline simulations. Test model selection algorithms like XGBoost, Random Forest, or SVM over grid search spaces on key datasets.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 Agent Orchestrator</h3>
        <p>Interact with our lead LangChain assistant. Generate complex pandas preprocessing operations or hyperparameter recommendations powered by the high-speed Groq API.</p>
    </div>
    """, unsafe_allow_html=True)

# System Architecture & Guidance
st.markdown("---")
st.subheader("Getting Started Quick Guide")

guide_col1, guide_col2 = st.columns(2)

with guide_col1:
    st.markdown("""
    **Folder Navigation**:
    Use the **sidebar page navigation links** to move between:
    - **1_Dashboard**: View metrics plots and list of experiments.
    - **2_New_Experiment**: Spawn new training trials in the background.
    - **3_Agent_Orchestrator**: Chat with the LangChain service.
    """)

with guide_col2:
    st.markdown("""
    **Running via Console**:
    To launch the platform locally:
    ```bash
    # 1. Start FastAPI backend (Port 8000)
    uvicorn src.backend.main:app --reload
    
    # 2. Start Streamlit frontend (Port 8501)
    streamlit run src/frontend/app.py
    ```
    """)
st.info("💡 **Tip**: If running the services via Docker Compose, simply run `docker-compose up --build` from the project root directory!")
