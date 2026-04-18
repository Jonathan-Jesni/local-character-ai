"""
Character AI Clone — FastAPI Backend
=====================================
Local-only, multimodal Character AI system that interfaces with a local
Ollama instance (defaulting to the `qwen` model).

Architecture:
  1. Global state dict holds shared conversation history + character registry.
  2. Orchestration engine swaps system prompts per-character before each call.
  3. Multi-pass output pipeline: Parse → Structure → Render.
"""

import os
import uuid
import base64
import requests
import re
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ollama

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Character AI Clone", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Global State — shared conversation memory + character registry
# ---------------------------------------------------------------------------
chat_session: dict = {
    # Shared history visible to every character
    "history": [],
    # Maps character name → system prompt string
    "characters": {},
}

# Default Ollama config
OLLAMA_MODEL = "qwen3.5:9b"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CharacterPayload(BaseModel):
    """Payload for adding a new character."""
    name: str
    description: str


class ChatPayload(BaseModel):
    """Payload for sending a message and requesting a character response."""
    character_name: str
    user_message: str


class CharacterResponse(BaseModel):
    """Structured response returned to the frontend."""
    character: str
    text: str
    image_prompts: list[str]

# ---------------------------------------------------------------------------
# Multi-Pass Output Pipeline
# ---------------------------------------------------------------------------
# The pipeline runs in three stages on every raw LLM output:
#
#   PASS 1 — Parse
#     Scan the raw text for all occurrences of [IMAGE: <description>].
#     Return a list of matched description strings.
#
#   PASS 2 — Structure
#     Hand the extracted prompts off for backend processing.  In the MVP we
#     simply log them; a real implementation would queue them for an image
#     model such as Stable Diffusion.
#
#   PASS 3 — Render
#     Scrub every [IMAGE: …] trigger from the text and replace it with a
#     user-friendly action marker (*shares an image*).  The cleaned text is
#     what the frontend receives.
# ---------------------------------------------------------------------------

_IMAGE_PATTERN = re.compile(r"\[IMAGE:\s*(.+?)\]", re.DOTALL)

def generate_image(prompt: str) -> Optional[str]:
    """Generate image via SD WebUI (Forge) and return static file path."""
    url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    payload = {
        "prompt": f"{prompt.strip()}, masterpiece, high resolution, highly detailed",
        "negative_prompt": "lowres, bad quality, worst quality, ugly, messy",
        "steps": 20,
        "width": 512,
        "height": 512,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "images" in data and len(data["images"]) > 0:
            b64_img = data["images"][0]
            img_data = base64.b64decode(b64_img)
            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join("static", filename)
            with open(filepath, "wb") as f:
                f.write(img_data)
            logger.info("🖼️  Image generated: %s", filename)
            return f"/static/{filename}"
    except Exception as exc:
        logger.error("Image generation failed: %s", exc)
    return None


def intercept_output(raw_text: str) -> tuple[str, list[str]]:
    """
    Run the full multi-pass pipeline on a raw LLM response.
    Triggers local image generation and swaps trigger syntax.

    Returns
    -------
    (cleaned_text, image_prompts)
    """
    prompts = _IMAGE_PATTERN.findall(raw_text)
    cleaned_text = raw_text
    
    for prompt in prompts:
        img_path = generate_image(prompt)
        
        if img_path:
            # Replace exactly the matched pattern
            replace_pattern = re.compile(r"\[IMAGE:\s*" + re.escape(prompt) + r"\]")
            cleaned_text = replace_pattern.sub(f"[LOCAL_IMG:{img_path}]", cleaned_text, count=1)
        else:
            replace_pattern = re.compile(r"\[IMAGE:\s*" + re.escape(prompt) + r"\]")
            cleaned_text = replace_pattern.sub("*shares an image (failed to load)*", cleaned_text, count=1)

    return cleaned_text, prompts

# ---------------------------------------------------------------------------
# Character Initialization
# ---------------------------------------------------------------------------

def _build_system_prompt(name: str, description: str) -> str:
    """
    Format a character description into a strict system prompt that instructs
    the model on persona and available tool syntax.
    """
    return (
        f"You are {name}. {description}\n\n"
        "RULES:\n"
        "1. Always stay in character.\n"
        "2. Never break the fourth wall or mention that you are an AI.\n"
        "3. If you want to share an image, output EXACTLY the syntax "
        "[IMAGE: <description of what the image shows>]. "
        "The system will handle generation.\n"
        "4. Keep responses concise and engaging.\n"
    )

# ---------------------------------------------------------------------------
# Orchestration Engine
# ---------------------------------------------------------------------------

def generate_response(character_name: str) -> CharacterResponse:
    """
    Core generation function.
      1. Retrieve the character's system prompt.
      2. Build the payload: [system_prompt] + shared_history.
      3. Call ollama.chat().
      4. Run the multi-pass pipeline.
      5. Append the cleaned response to shared history.
    """
    if character_name not in chat_session["characters"]:
        raise ValueError(f"Character '{character_name}' not found.")

    system_prompt = chat_session["characters"][character_name]
    messages = [{"role": "system", "content": system_prompt}] + chat_session["history"]

    try:
        result = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    except Exception as exc:
        logger.error("Ollama call failed: %s", exc)
        raise RuntimeError(
            "Failed to reach Ollama. Is it running on localhost:11434?"
        ) from exc

    raw_text: str = result["message"]["content"]

    # Multi-pass interception
    cleaned_text, image_prompts = intercept_output(raw_text)

    # Append to shared history with speaker label
    chat_session["history"].append({
        "role": "assistant",
        "content": f"{character_name}: {cleaned_text}",
    })

    return CharacterResponse(
        character=character_name,
        text=cleaned_text,
        image_prompts=image_prompts,
    )

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_path, media_type="text/html")


@app.post("/add_character")
async def add_character(payload: CharacterPayload):
    """Dynamically inject a new character into the session."""
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Character name is required.")

    prompt = _build_system_prompt(name, payload.description)
    chat_session["characters"][name] = prompt

    logger.info("✅ Character added: %s", name)
    return {"status": "ok", "character": name}


@app.get("/characters")
async def list_characters():
    """Return list of active character names."""
    return {"characters": list(chat_session["characters"].keys())}


@app.post("/chat", response_model=CharacterResponse)
async def chat(payload: ChatPayload):
    """
    Accept a user message and generate a response from the specified character.
    """
    character = payload.character_name.strip()
    message = payload.user_message.strip()

    if not character:
        raise HTTPException(status_code=400, detail="character_name is required.")
    if character not in chat_session["characters"]:
        raise HTTPException(status_code=404, detail=f"Character '{character}' not found.")

    # Append user message to shared history
    chat_session["history"].append({"role": "user", "content": message})

    try:
        response = generate_response(character)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return response


@app.post("/reset")
async def reset_session():
    """Clear all history and characters."""
    chat_session["history"].clear()
    chat_session["characters"].clear()
    logger.info("🔄 Session reset.")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
