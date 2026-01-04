from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

# ✅ APP MUST BE DEFINED FIRST
app = FastAPI(
    title="AgentRoot – AI Student Mentor",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file")

# Request model
class ProblemRequest(BaseModel):
    problem: str

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.post("/analyze")
def analyze_problem(data: ProblemRequest):
    problem = data.problem.strip()

    if not problem:
        raise HTTPException(status_code=400, detail="Problem cannot be empty")

    prompt = f"""
You are an AI Student Mentor.

Respond STRICTLY in this format:

📌 Problem Summary:
🧠 Mentor Advice:
✅ Action Plan:
⚠️ Risk Level:
🔥 Motivation:

Student Problem:
{problem}
"""

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ]
            },
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        # SAFE RESPONSE HANDLING
        if "candidates" not in result:
            logging.error(result)
            return {"result": "❌ Gemini API error. Check API key or billing."}

        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return {"result": text}

    except Exception as e:
        logging.error(e)
        return {"result": "❌ AI service failed"}
