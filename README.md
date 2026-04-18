# Local Multimodal Character AI Clone

A fully local, multimodal AI chat interface that orchestrates text generation and image synthesis on local silicon. 

## 🏗️ Architecture
This project operates as a microservice architecture entirely on localhost:
1. **The Brain (Ollama):** Handles character persona logic, chat history, and image prompting using local LLMs (e.g., Qwen).
2. **The Orchestrator (FastAPI):** A Python backend that manages global state, intercepts specific `[IMAGE:]` triggers from the LLM, and routes API payloads.
3. **The Engine (Stable Diffusion Forge):** A locally hosted image generation API running hardware-accelerated generation via PyTorch.

## 🚀 Features
* **Zero External APIs:** Fully local generation, meaning zero rate limits and total privacy.
* **Multi-Pass Rendering:** The backend dynamically parses LLM text, requests an image generation from the SD API, saves it locally, and injects the formatted HTML back into the chat flow.
* **Dynamic Character Injection:** Easily add new characters with custom system prompts on the fly.

## 🛠️ Setup
1. Start Ollama locally on port `11434`.
2. Start Stable Diffusion WebUI (Forge) with the `--api` flag on port `7860`. 
   *(Note: For RTX 50-series Blackwell GPUs, ensure PyTorch is upgraded to the `cu130` nightly build for native hardware support).*
3. Install dependencies: `pip install -r requirements.txt`
4. Launch the FastAPI server: `python main.py`