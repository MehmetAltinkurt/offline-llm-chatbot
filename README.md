# Offline LLM Chatbot

A Streamlit-based, fully local (offline) chat UI for running LLMs on **CPU**—no GPU required.  
The project will use [llama.cpp](https://github.com/ggerganov/llama.cpp) via `llama-cpp-python` and GGUF models.

## Goals
- Simple local chat interface
- CPU-friendly defaults (works on typical laptops/desktops)
- Clear, incremental roadmap with daily small improvements

## Requirements
- Python 3.10+ (recommended)
- OS: Windows, macOS, or Linux

## Quick Start
```bash
git clone https://github.com/MehmetAltinkurt/offline-llm-chatbot.git
cd offline-llm-chatbot
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
