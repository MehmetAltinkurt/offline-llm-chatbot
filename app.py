import json
import time
from pathlib import Path
import streamlit as st

# Try to import llama-cpp-python early and show a friendly hint if missing
try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None
    _import_err = e
else:
    _import_err = None

st.set_page_config(page_title="Offline LLM Chat", page_icon="💬", layout="wide")
st.title("💬 Offline LLM Chat (CPU via llama.cpp)")

# -----------------------------
# Sidebar: model & inference UI
# -----------------------------
st.sidebar.header("Model & Inference Settings")

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_MODELS_DIR.mkdir(exist_ok=True)

def list_gguf_models(models_dir: Path) -> list[str]:
    return sorted(str(p) for p in models_dir.glob("**/*.gguf"))

available_models = list_gguf_models(DEFAULT_MODELS_DIR)

model_select_mode = st.sidebar.radio(
    "Model path source",
    options=["Pick from ./models", "Enter custom path"],
    index=0,
    help="Put GGUF files under ./models to have them listed automatically."
)

if model_select_mode == "Pick from ./models":
    if not available_models:
        st.sidebar.warning("No GGUF found under ./models. Switch to 'Enter custom path' or add a model.")
        model_path = ""
    else:
        model_path = st.sidebar.selectbox("GGUF file", available_models, index=0)
else:
    model_path = st.sidebar.text_input(
        "GGUF model path",
        value=str(DEFAULT_MODELS_DIR / "your-model.Q4_K_M.gguf"),
        help="Paste a full path to a GGUF file."
    )

n_ctx = st.sidebar.number_input("Context size (n_ctx)", min_value=256, max_value=16384, value=4096, step=256)
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.95, 0.05)
repeat_penalty = st.sidebar.slider("Repeat penalty", 1.0, 2.0, 1.1, 0.05)
max_new_tokens = st.sidebar.number_input("Max new tokens", min_value=16, max_value=4096, value=512, step=16)

reload_clicked = st.sidebar.button("🔄 Load / Reload Model", use_container_width=True)

# -----------------------------
# Session state
# -----------------------------
ss = st.session_state
ss.setdefault("llm", None)
ss.setdefault("loaded_model_path", None)
ss.setdefault("history", [])          # list of dicts: {"role": "user"/"assistant"/"system", "content": str}
ss.setdefault("abort", False)         # abort flag for streaming

# -----------------------------
# Helper: load model
# -----------------------------
def load_model(path: str):
    """Load a GGUF model safely. Returns (llm, message, ok:bool)."""
    if Llama is None:
        msg = (
            "Failed to import `llama_cpp`. Install it first:\n\n"
            "```bash\npip install --upgrade llama-cpp-python\n```\n"
            "If you use conda: `conda install -c conda-forge llama-cpp-python`.\n\n"
            f"Import error detail: `{_import_err}`"
        )
        return None, msg, False

    p = Path(path)
    if not p.exists() or not p.is_file():
        return None, f"Model file not found at: `{p}`.", False

    try:
        start = time.time()
        llm = Llama(
            model_path=str(p),
            n_ctx=int(n_ctx),
            verbose=False,
        )
        ms = int((time.time() - start) * 1000)
        return llm, f"Model loaded in {ms} ms.", True
    except Exception as e:
        return None, f"Error while loading model: {e}", False

# Decide whether to load
should_load = False
if reload_clicked:
    should_load = True
elif ss.llm is None and model_path.strip():
    should_load = True
elif ss.loaded_model_path and ss.loaded_model_path != model_path:
    st.sidebar.warning("Model path changed. Click 'Load / Reload Model' to apply.")

if should_load:
    with st.spinner("Loading model…"):
        llm, msg, ok = load_model(model_path)
    if ok:
        ss.llm = llm
        ss.loaded_model_path = model_path
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

# -----------------------------
# Chat history display
# -----------------------------
def render_history():
    for turn in ss.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

render_history()

# -----------------------------
# Prompt building (minimal)
# -----------------------------
def build_prompt(history, user_message: str) -> str:
    """
    Minimal plain-text prompt. Some models may prefer specific templates;
    we'll add template selection in a future day.
    """
    system_text = "You are a helpful assistant. Keep answers concise."
    lines = [f"System: {system_text}"]
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
# Export helpers
# -----------------------------
def export_markdown(history) -> str:
    parts = []
    for t in history:
        role = t["role"].capitalize()
        parts.append(f"**{role}**:\n\n{t['content']}\n")
    return "\n---\n".join(parts)

def export_json(history) -> str:
    return json.dumps(history, ensure_ascii=False, indent=2)

# -----------------------------
# Abort controls
# -----------------------------
def request_abort():
    ss.abort = True

# Top action row
left, mid, right = st.columns([1, 1, 2])
with left:
    if st.button("🧹 Clear chat", use_container_width=True):
        ss.history = []
        st.rerun()
with mid:
    st.download_button(
        "⬇️ Export .md",
        data=export_markdown(ss.history),
        file_name="conversation.md",
        mime="text/markdown",
        use_container_width=True,
    )
with right:
    st.download_button(
        "⬇️ Export .json",
        data=export_json(ss.history),
        file_name="conversation.json",
        mime="application/json",
        use_container_width=True,
    )

# -----------------------------
# Chat input & generation
# -----------------------------
if ss.llm is None:
    st.info(
        "Load a GGUF model from the sidebar to start chatting. "
        "Place a file under `./models` or paste a custom path."
    )

# Place the Stop button where the assistant message will render
stop_col, _ = st.columns([1, 3])
with stop_col:
    stop_pressed = st.button("⏹ Stop", disabled=(ss.llm is None), on_click=request_abort)

user_msg = st.chat_input("Type your message…", disabled=(ss.llm is None))

if user_msg:
    ss.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        ss.abort = False  # reset abort flag for this generation

        if ss.llm is None:
            placeholder.warning("Model is not loaded. Please load a model from the sidebar.")
        else:
            prompt = build_prompt(ss.history, user_msg)
            try:
                start = time.time()
                for out in ss.llm(
                    prompt=prompt,
                    max_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    repeat_penalty=float(repeat_penalty),
                    stream=True,
                ):
                    # Abort check
                    if ss.abort:
                        full_text += "\n\n_⏹ Generation stopped by user._"
                        break

                    token = out.get("choices", [{}])[0].get("text", "")
                    if token:
                        full_text += token
                        placeholder.markdown(full_text)

                elapsed = time.time() - start
                placeholder.markdown(full_text + f"\n\n---\n_Generated in {elapsed:.1f}s_")
            except Exception as e:
                placeholder.error(f"Generation error: {e}")
                full_text = ""

        ss.history.append({"role": "assistant", "content": full_text})

# Footer: show loaded model path
if ss.loaded_model_path:
    st.caption(f"Model: `{ss.loaded_model_path}`")
