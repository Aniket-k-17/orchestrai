# ML Experiment Orchestrator

The **ML Experiment Orchestrator** is a production-ready, modular architecture designed to manage, execute, and inspect Machine Learning pipelines and trigger LLM-powered multi-agent orchestration tasks.

Built using FastAPI for high-performance API services, Streamlit for responsive analytics dashboards, and LangChain/Groq for intelligent execution workflows.

---

## 🏗️ Architecture & Project Structure

The project code is modularized cleanly into `src/backend` and `src/frontend` directories to isolate UI layers from state logic and external API integrations.

```text
orchestrai/
├── .env.example            # Template for environment settings
├── .gitignore              # Git ignore files definition
├── Dockerfile.backend      # Container building definition for FastAPI service
├── Dockerfile.frontend     # Container building definition for Streamlit dashboard
├── docker-compose.yml      # Multi-container service orchestrator
├── README.md               # Project documentation
├── requirements.txt        # Python dependency manifest
└── src/
    ├── __init__.py
    ├── config.py           # Configuration management via Pydantic Settings
    ├── logging_setup.py    # Standardized Rotating File & Console logger
    ├── backend/
    │   ├── __init__.py
    │   ├── main.py         # FastAPI application bootstrap
    │   ├── api/
    │   │   ├── __init__.py
    │   │   ├── router.py   # Main API router mounting all version routes
    │   │   └── v1/
    │   │       ├── __init__.py
    │   │       ├── router.py  # V1 routes aggregator
    │   │       └── endpoints/
    │   │           ├── __init__.py
    │   │           ├── agents.py       # LangChain agent interaction routes
    │   │           └── experiments.py  # Experiment execution & simulation routes
    │   ├── schemas/
    │   │   ├── __init__.py
    │   │   ├── agent.py        # Agent response and planning step schemas
    │   │   └── experiment.py   # Experiment status and metrics schemas
    │   └── services/
    │       ├── __init__.py
    │       └── langchain_service.py  # ChatGroq LLM planning agent orchestrator
    ├── frontend/
    │   ├── __init__.py
    │   ├── app.py          # Streamlit landing view and styling setup
    │   ├── pages/          # Streamlit multi-page dashboards (sidebar auto-discovered)
    │   │   ├── 1_Dashboard.py         # Run logs and Plotly comparative graphs
    │   │   ├── 2_New_Experiment.py    # Form grids for hyperparameters tuning
    │   │   └── 3_Agent_Orchestrator.py # Multi-agent console & chat logs
    │   └── utils/
    │       ├── __init__.py
    │       └── api_client.py  # HTTP REST API client connection to FastAPI
    └── utils/
        ├── __init__.py
        ├── helpers.py      # Custom JSON encoder for NumPy/Pandas and ID helpers
        └── metrics.py      # Future module for ML evaluations
```

---

## 🛠️ Tech Stack & Dependencies

- **FastAPI**: Asynchronous web framework for high-throughput, automatic Swagger documentation generation, and route-level validation.
- **Streamlit**: Fast UI dashboard creation for inspecting models and rendering logs.
- **LangChain**: Chains prompts, structures output parsing, and guides agent planning pipelines.
- **Groq API**: High-speed, low-latency LLM inference (e.g. Mixtral-8x7b) for agent decisions.
- **Pandas & NumPy**: Data processing and numeric structures validation.
- **Scikit-learn**: Future ML baseline training algorithms (e.g. Random Forest, SVM).
- **Plotly & Matplotlib**: Interactive HTML charts for visual analysis of experiment trials.
- **Docker & Docker Compose**: Container build orchestration for microservices setup.

---

## 🚀 Setting Up the Project

### 1. Local Development Setup (Virtual Environment)

Ensure Python 3.10+ is installed on your machine.

```bash
# Clone the repository
git clone <repository_url>
cd orchestrai

# Create and activate virtual environment
python -m venv venv
# On Windows (cmd/PowerShell):
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all project requirements
pip install -r requirements.txt
```

### 2. Configure Environment Settings

Copy the environment template and edit parameters if needed:
```bash
cp .env.example .env
```
Open `.env` and add your **`GROQ_API_KEY`** to unlock live agent planning. If no key is set, the system runs with predefined mock answers for testing.

---

## 🏃 Running the Application

You can execute the architecture components locally in the terminal or inside Docker containers.

### Option A: Local Run (Separate Consoles)

**Step 1: Start FastAPI backend service (default port: 8000)**
```bash
uvicorn src.backend.main:app --reload
```
You can view the Swagger UI documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

**Step 2: Start Streamlit frontend server (default port: 8501)**
```bash
streamlit run src/frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the dashboard.

### Option B: Docker Compose Setup (Recommended)

Boot the entire ecosystem in containers with a single command:
```bash
# Build and run containers in background
docker-compose up --build
```
This launches the FastAPI backend and Streamlit frontend together, routing internal requests through docker's local bridge.

---

## 💡 Code Setup Highlights

1. **Robust Serializers**: `src.utils.helpers.MLJSONEncoder` handles serialization of NumPy ints/floats and Pandas series/dataframes so that model metrics don't trigger serialization exceptions in production APIs.
2. **Mocking & Fallbacks**: The `LangChainOrchestratorService` dynamically detects key omissions or missing installations, continuing in an offline mock mode.
3. **Pydantic Validation**: Strong schemas protect input variables for experiments (`ExperimentCreate`) and chat plans (`AgentTaskRequest`).
4. **Rotating Logs**: A console & file logger writes rotating system outputs under `logs/app.log`, keeping local disk consumption bounded.
