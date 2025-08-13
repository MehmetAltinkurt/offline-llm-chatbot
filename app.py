import os
from pathlib import Path
import time
import streamlit as st

# Try to import llama-cpp-python early and show a friendly hint if missing
try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None
    _import_err = e
else:
    _import_err = None

# --------------------------------------
# Offline LLM Chat – Day 2 milestone
# --------------------------------------
# What this version adds:
# 1) Sidebar model path + inference settings
# 2) Safe model (re)loader with clear error messages
# 3) Minimal chat with token streaming
#
# Next Day (Day 3) ideas:
# - Persist chat history; export .md/.json
# - Model auto-discovery from /models folder
# - Abort/stop generation button

st.set_page_config(page_title="Offline LLM Chat", page_icon="💬", layout="wide")
st.title("💬 Offline LLM Chat (CPU via llama.cpp)")

# -----------------------------
# Sidebar: model & inference UI
# -----------------------------
st.sidebar.header("Model & Inference Settings")

default_models_dir = "models"
Path(default_models_dir).mkdir(exist_ok=True)

model_path = st.sidebar.text_input(
    "GGUF model path",
    value=str(Path(default_models_dir) / "your-model.Q4_K_M.gguf"),
    help="Place a GGUF file under ./models and paste its path here."
)

n_ctx = st.sidebar.number_input("Context size (n_ctx)", min_value=256, max_value=16384, value=4096, step=256)
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.95, 0.05)
repeat_penalty = st.sidebar.slider("Repeat penalty", 1.0, 2.0, 1.1, 0.05)
max_new_tokens = st.sidebar.number_input("Max new tokens", min_value=16, max_value=4096, value=512, step=16)

reload_clicked = st.sidebar.button("🔄 Load / Reload Model")

# -----------------------------
# Session state
# -----------------------------
if "llm" not in st.session_state:
    st.session_state.llm = None
if "loaded_model_path" not in st.session_state:
    st.session_state.loaded_model_path = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role": "user"/"assistant"/"system", "content": str}

# -----------------------------
# Helper: load model
# -----------------------------
def load_model(path: str):
    """Load a GGUF model safely. Returns (llm, error_message)."""
    if Llama is None:
        return None, (
            "Failed to import `llama_cpp`. Install it first:\n\n"
            "```bash\npip install --upgrade llama-cpp-python\n```\n"
            "If you use conda: `conda install -c conda-forge llama-cpp-python`.\n\n"
            f"Import error detail: `{_import_err}`"
        )

    p = Path(path)
    if not p.exists() or not p.is_file():
        return None, f"Model file not found at: `{p}`. Put a GGUF file under `./models` and update the path."

    try:
        # Keep verbose=False to reduce console noise. You can tweak n_threads if needed.
        started = time.time()
        llm = Llama(
            model_path=str(p),
            n_ctx=int(n_ctx),
            verbose=False,
        )
        load_ms = int((time.time() - started) * 1000)
        return llm, f"Model loaded in {load_ms} ms."
    except Exception as e:
        return None, f"Error while loading model: {e}"

# Load model on demand (button) or if path changed
should_load = False
if reload_clicked:
    should_load = True
elif st.session_state.llm is None and model_path.strip():
    # First run: try to load automatically if a path is provided
    should_load = True
elif st.session_state.loaded_model_path and st.session_state.loaded_model_path != model_path:
    # Path changed: prompt user to reload
    st.sidebar.warning("Model path changed. Click 'Load / Reload Model' to apply.")

if should_load:
    with st.status("Loading model…", expanded=False) as status:
        llm, msg = load_model(model_path)
        if llm is not None:
            st.session_state.llm = llm
            st.session_state.loaded_model_path = model_path
            status.update(label="Model ready ✅", state="complete")
            st.sidebar.success(msg)
        else:
            status.update(label="Model load failed ❌", state="error")
            st.sidebar.error(msg)

# -----------------------------
# Chat history display
# -----------------------------
def render_history():
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

render_history()

# -----------------------------
# Prompt building (very simple)
# -----------------------------
def build_prompt(history, user_message: str) -> str:
    """
    Build a minimal plain-text prompt compatible with many instruct/chat GGUFs.
    For Day 2 we keep it simple: short 'System/User/Assistant' style without templates.
    """
    system_text = "You are a helpful assistant. Keep answers concise."
    lines = [f"System: {system_text}"]
    # Include only the last few turns to keep latency lower on CPU
    recent = history[-6:] if len(history) > 6 else history
    for t in recent:
        if t["role"] == "user":
            lines.append(f"User: {t['content']}")
        elif t["role"] == "assistant":
            lines.append(f"Assistant: {t['content']}")
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)

# -----------------------------
# Chat input & generation
# -----------------------------
if st.session_state.llm is None:
    st.info(
        "Load a GGUF model from the sidebar to start chatting. "
        "Place a file under `./models` and paste its path (e.g., `models/your-model.Q4_K_M.gguf`)."
    )

user_msg = st.chat_input("Type your message…", disabled=(st.session_state.llm is None))
if user_msg:
    # show user message
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # build prompt and stream reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        if st.session_state.llm is None:
            placeholder.warning("Model is not loaded. Please load a model from the sidebar.")
        else:
            prompt = build_prompt(st.session_state.history, user_msg)
            try:
                start = time.time()
                for out in st.session_state.llm(
                    prompt=prompt,
                    max_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    repeat_penalty=float(repeat_penalty),
                    stream=True,
                ):
                    token = out.get("choices", [{}])[0].get("text", "")
                    if token:
                        full_text += token
                        placeholder.markdown(full_text)
                elapsed = time.time() - start
                # (Optional) tiny footer with speed info
                placeholder.markdown(full_text + f"\n\n---\n_Generated in {elapsed:.1f}s_")
            except Exception as e:
                placeholder.error(f"Generation error: {e}")
                full_text = ""

        st.session_state.history.append({"role": "assistant", "content": full_text})

# Utilities
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear chat"):
        st.session_state.history = []
        st.rerun()
with col2:
    if st.session_state.loaded_model_path:
        st.caption(f"Model: `{st.session_state.loaded_model_path}`")