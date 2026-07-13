import streamlit as st
from datetime import datetime

from src.frontend.utils.api_client import BackendAPIClient

api_client = BackendAPIClient()

st.set_page_config(page_title="Agent Orchestrator - ML Orchestrator", page_icon="🤖", layout="wide")

# Custom UI Theme Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@300;500&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0c10;
    }
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .user-bubble {
        background: rgba(102, 252, 241, 0.08);
        border-left: 4px solid #66fcf1;
    }
    .agent-bubble {
        background: rgba(31, 40, 56, 0.4);
        border-left: 4px solid #1f4068;
    }
    .bubble-role {
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #8b9bb4;
        margin-bottom: 0.5rem;
    }
    .bubble-time {
        font-size: 0.75rem;
        color: #556677;
        margin-top: 0.8rem;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-Agent Orchestrator Console")
st.write("Interact with the LangChain service to plan data preprocessing or suggest hyperparameters.")

# Session chat logs array initialized in Streamlit state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "session_id" not in st.session_state:
    # Set a unique session ID
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar settings
st.sidebar.subheader("Agent Engine Parameters")
st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
model_provider = st.sidebar.selectbox("LLM Provider", ["Groq API (Mixtral)"])
temperature_val = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

# Suggestion quick prompts buttons
st.write("**Quick Instructions Suggestions:**")
col_s1, col_s2 = st.columns(2)
with col_s1:
    if st.button("📊 'Suggest data preprocessing plan for California housing dataset'"):
        st.session_state.custom_prompt_input = "Suggest data preprocessing plan for California housing dataset"
with col_s2:
    if st.button("⚙️ 'Recommend hyperparameter search space for tuning XGBoost model'"):
        st.session_state.custom_prompt_input = "Recommend hyperparameter search space for tuning XGBoost model"

# User prompt textarea input box
prompt = st.text_input("Enter prompt message for ML Agent...", key="custom_prompt_input")

if st.button("Send Instruction", type="primary"):
    if not prompt:
        st.warning("Please type a message first.")
    else:
        # User message
        user_msg = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        st.session_state.chat_messages.append(user_msg)
        
        with st.spinner("🤖 Agent planner is thinking and processing execution steps..."):
            # Call API client
            context = {
                "temperature": temperature_val,
                "provider": model_provider
            }
            response = api_client.execute_agent_task(
                prompt=prompt,
                session_id=st.session_state.session_id,
                context_data=context
            )
            
            if response:
                # Add assistant reply to logs
                assistant_msg = {
                    "role": "assistant",
                    "content": response.get("final_output", ""),
                    "steps": response.get("steps", []),
                    "timestamp": datetime.utcnow().strftime("%H:%M:%S")
                }
                st.session_state.chat_messages.append(assistant_msg)
            else:
                st.error("Failed to receive output from backend LangChain service. Is the FastAPI backend running?")

# Display Chat History logs backwards (latest at bottom)
st.write("---")
st.subheader("Interactive Message Feed")

for msg in st.session_state.chat_messages:
    role_class = "user-bubble" if msg["role"] == "user" else "agent-bubble"
    role_title = "User Request" if msg["role"] == "user" else "Agent Orchestrator"
    
    st.markdown(f"""
    <div class="chat-bubble {role_class}">
        <div class="bubble-role">{role_title}</div>
        <div>{msg['content']}</div>
        <div class="bubble-time">{msg['timestamp']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # If the message contains sub-agent step details trace, render inside an expander
    if msg["role"] == "assistant" and msg.get("steps"):
        with st.expander("🔍 View Multi-Agent Planning Execution Trace Steps"):
            for step in msg["steps"]:
                st.markdown(f"**Step {step.get('step_number')}: {step.get('agent_name')}**")
                st.write(f"- *Action taken:* {step.get('action')}")
                st.write(f"- *Observation:* `{step.get('observation')}`")
                st.write("---")
