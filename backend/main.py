from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import requests
import json
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "convoiq"   # created from the Modelfile
# If you do not create the custom model, change this to "gemma:2b"

app = FastAPI(title="ConversationIQ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    conversation: str = Field(min_length=1, max_length=20000)

SCHEMA = {
    "type": "object",
    "properties": {
        "overall_tone": {"type": "string"},
        "emotion": {"type": "string"},
        "confidence": {"type": "number"},
        "stress_level": {"type": "number"},
        "communication_style": {"type": "string"},
        "communication_patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "quote": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["type", "quote", "reason"]
            }
        },
        "summary": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "overall_tone",
        "emotion",
        "confidence",
        "stress_level",
        "communication_style",
        "communication_patterns",
        "summary",
        "evidence"
    ]
}

def build_prompt(conversation: str):
    return f"""
You are an expert conversation analyst.

Analyze ONLY the conversation below.

Rules:

- Never invent dialogue.
- Never invent quotes.
- Never infer facts not present.
- Every communication pattern MUST reference an exact quote.
- If there is insufficient evidence, return an empty list.

Allowed communication pattern types:

Emotional Support
Validation
Encouragement
Curiosity
Active Listening
Generalization
Blame
Defensiveness
Social Proof
Guilt Appeal
Passive Aggression
Boundary Setting
Threat
Criticism

Conversation:

{conversation}
""".strip()

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    payload = {
        "model": MODEL_NAME,
        "prompt": build_prompt(req.conversation),
        "format": SCHEMA,
        "stream": False,
        "options": {
    "temperature": 0,
    "top_p": 0.1,
    "repeat_penalty": 1.2,
    "num_predict": 400
}
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {e}")

    raw = data.get("response", "").strip()
    print("\n========== RAW MODEL OUTPUT ==========\n")
    print(raw)
    print("\n======================================\n")

    try:
        parsed = json.loads(raw)
        conversation = req.conversation
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model did not return valid JSON. Raw output: {raw[:500]}"
    )

    conversation = req.conversation

    patterns = []

    for item in parsed.get("communication_patterns", []):
        if not isinstance(item, dict):
          continue

        quote = item.get("quote", "").strip()

        if quote and quote in conversation:
           patterns.append(item)

    parsed["communication_patterns"] = patterns

    return parsed