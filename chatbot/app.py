import os

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="OpenRouter Chatbot", page_icon="💬", layout="centered")

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat",
]


def get_api_key() -> str:
    """Look for the key in Streamlit secrets, then env var, then the sidebar."""
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENROUTER_API_KEY", "")


# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("OpenRouter API key", type="password")

    model = st.selectbox("Model", MODELS, index=0)
    custom_model = st.text_input("Or custom model ID (optional)")
    if custom_model.strip():
        model = custom_model.strip()

    system_prompt = st.text_area(
        "System prompt", value="You are a helpful, concise assistant.", height=100
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max tokens", 64, 4096, 1024, 64)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 OpenRouter Chatbot")
st.caption(f"Model: `{model}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Input / response ----------
if prompt := st.chat_input("Type your message..."):
    if not api_key:
        st.error("Please provide your OpenRouter API key in the sidebar or secrets.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    api_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            def token_generator():
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            reply = st.write_stream(token_generator())
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Request failed: {e}")