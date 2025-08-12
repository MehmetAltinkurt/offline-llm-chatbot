import streamlit as st

# -----------------------------
# CPU LLM Chat – minimal skeleton
# -----------------------------
# This first commit is intentionally simple so the app runs out-of-the-box.
# We'll add llama.cpp (llama-cpp-python) integration and a real chat loop next.
#
# Next steps:
# 1) Add a sidebar text input for GGUF model path (e.g., models/your-model.gguf)
# 2) Initialize Llama(...) from llama_cpp when the file exists
# 3) Implement a streaming chat response and maintain session history
# 4) Add basic inference settings (temperature, top_p, context window)

st.set_page_config(page_title="CPU LLM Chat", page_icon="💬", layout="centered")

st.title("💬 CPU LLM Chat")
st.write(
    "Welcome! This repository will provide a fully local, CPU-friendly LLM chat UI using "
    "`llama.cpp` through `llama-cpp-python`. For now, this is a minimal starter app to "
    "bootstrap the project."
)

st.subheader("Status")
st.info(
    "First commit: minimal Streamlit app is up. "
    "Model loading & chat features will be added in upcoming commits."
)

st.subheader("What’s next?")
st.markdown(
    "- Sidebar for model path (GGUF)\n"
    "- llama.cpp model initialization\n"
    "- Chat input & assistant responses\n"
    "- Export conversation"
)

# Simple placeholder chat input to verify the UI flow
user_msg = st.chat_input("Type a message (UI placeholder)…")
if user_msg:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(
            "This is a placeholder reply. In the next commits, I will respond "
            "using a local CPU LLM loaded via `llama-cpp-python`."
        )
