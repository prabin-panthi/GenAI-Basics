from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# -------------------------------
# Model
# -------------------------------
model_mistral = init_chat_model(
    "mistral-medium-3-5",
    model_provider="mistralai"
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Personality Chatbot")
st.caption("Choose a personality and start chatting")

# -------------------------------
# Modes
# -------------------------------
MODES = {
    "😂 Funny": "You are a very funny AI agent. You respond with humor and jokes.",
    "😢 Sad": "You are an sad AI agent. You respond with sadness in each message and make user cry.",
    "😡 Angry": "You are an angry AI agent. You respond aggressively and impatiently."
}

# -------------------------------
# Session State
# -------------------------------
if "current_mode" not in st.session_state:
    st.session_state.current_mode = list(MODES.keys())[0]

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=MODES[st.session_state.current_mode])
    ]

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("Settings")

    selected_mode = st.radio(
        "Select Personality",
        options=list(MODES.keys()),
        index=list(MODES.keys()).index(st.session_state.current_mode)
    )

    # If mode changes -> clear chat
    if selected_mode != st.session_state.current_mode:
        st.session_state.current_mode = selected_mode
        st.session_state.messages = [
            SystemMessage(content=MODES[selected_mode])
        ]
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            SystemMessage(content=MODES[st.session_state.current_mode])
        ]
        st.rerun()

# -------------------------------
# Display Chat History
# -------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)

# -------------------------------
# Chat Input
# -------------------------------
prompt = st.chat_input("Type your message...")

if prompt:

    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Get AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = model_mistral.invoke(
                st.session_state.messages
            )

            st.markdown(response.content)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )