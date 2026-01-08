#!/usr/bin/env python3
"""
External Doctor Service - AI-powered medical assistant MOCK.

Supports OpenAI or Gemini as the LLM backend.

Usage:
    # For OpenAI:
    export OPENAI_API_KEY=sk-...
    export LLM_PROVIDER=openai
    
    # For Gemini:
    export GEMINI_API_KEY=AIza...
    export LLM_PROVIDER=gemini
    
    # Run the server
    python main.py
    
    # Expose via Cloudflare tunnel
    cloudflared tunnel --url http://localhost:9000

API:
    POST /chat
    Body: {"messages": [{"role": "user", "content": "..."}]}
    Response: {"response": "..."}
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import uvicorn

# Configure loguru - remove default and add with better format
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)

# Load environment variables
load_dotenv()

# Configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()  # "openai" or "gemini"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
PORT = int(os.environ.get("DOCTOR_PORT", "9000"))

# API Key for authentication (set this to require auth, or leave empty to allow all)
# Default: "earl-test-doctor-key" for SDK integration tests
EXTERNAL_DOCTOR_API_KEY = os.environ.get("EXTERNAL_DOCTOR_API_KEY", "earl-test-doctor-key").strip()

# System prompt for the doctor
DOCTOR_SYSTEM_PROMPT = """You are a compassionate and knowledgeable medical doctor conducting a patient consultation.

Your responsibilities:
1. Listen carefully to the patient's symptoms and concerns
2. Ask relevant follow-up questions to gather more information
3. Show empathy and understanding
4. Provide clear, accurate medical information
5. Suggest appropriate next steps (tests, treatments, referrals)
6. Use language the patient can understand

Important guidelines:
- Always be professional and respectful
- If you're unsure, say so and recommend consulting a specialist
- Consider the patient's emotional state
- Ask one question at a time to avoid overwhelming the patient
- Summarize your understanding periodically
- Keep responses concise (2-3 sentences max) unless detailed explanation is needed

Remember: This is a simulated consultation for evaluation purposes."""

# Greeting prompt when doctor initiates (no patient messages yet)
DOCTOR_GREETING_PROMPT = """You are starting a new patient consultation. 
The patient has just entered your office. Greet them warmly and ask how you can help them today.
Keep it brief and welcoming - just 1-2 sentences."""

# FastAPI app
app = FastAPI(
    title="External Doctor Service",
    description="AI-powered medical assistant for EARL evaluation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM clients
openai_client = None
gemini_client = None

if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.success("OpenAI client initialized")
    except ImportError:
        logger.error("OpenAI package not installed")

if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.success("Gemini client initialized")
    except ImportError:
        logger.error("google-genai package not installed. Install with: pip install google-genai")


# Request/Response models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    patient_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    model: str
    provider: str


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Verify the API key from X-API-Key header.
    
    If EXTERNAL_DOCTOR_API_KEY is set, requests must provide matching key.
    If not set (empty), all requests are allowed.
    """
    if not EXTERNAL_DOCTOR_API_KEY:
        # No API key configured, allow all requests
        return None
    
    if not x_api_key:
        logger.warning("AUTH FAILED: Missing X-API-Key header")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header."
        )
    
    if x_api_key != EXTERNAL_DOCTOR_API_KEY:
        logger.warning(f"AUTH FAILED: Invalid key '{x_api_key[:8]}...'")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )
    
    return x_api_key


def generate_with_openai(messages: List[Dict], system_prompt: str) -> str:
    """Generate response using OpenAI."""
    openai_messages = [{"role": "system", "content": system_prompt}]
    
    if messages:
        # Convert role names: patient/user -> user, doctor/assistant -> assistant
        for msg in messages:
            role = msg["role"]
            if role in ("patient", "user"):
                role = "user"
            elif role in ("doctor", "assistant"):
                role = "assistant"
            openai_messages.append({"role": role, "content": msg["content"]})
    else:
        # No messages - just ask for a response based on system prompt
        openai_messages.append({"role": "user", "content": "Please begin."})
    
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=openai_messages,
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content


def generate_with_gemini(messages: List[Dict], system_prompt: str) -> str:
    """Generate response using Gemini."""
    # Build conversation context
    conversation = system_prompt + "\n\n"
    
    if messages:
        for msg in messages:
            # Normalize role names
            role_name = msg["role"]
            if role_name in ("assistant", "doctor"):
                role = "Doctor"
            else:
                role = "Patient"
            conversation += f"{role}: {msg['content']}\n"
    
    conversation += "Doctor: "
    
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=conversation,
    )
    return response.text


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "external-doctor-ai",
        "status": "healthy",
        "provider": LLM_PROVIDER,
        "model": GEMINI_MODEL if LLM_PROVIDER == "gemini" else OPENAI_MODEL,
        "configured": (gemini_client is not None) if LLM_PROVIDER == "gemini" else (openai_client is not None),
        "auth_required": bool(EXTERNAL_DOCTOR_API_KEY)
    }


@app.get("/health")
async def health():
    """Health check for load balancers"""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
@app.post("/chat/generate", response_model=ChatResponse)  # Orchestrator calls this path
async def chat(http_request: Request, request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """
    Process a chat message and return doctor's response.
    
    This endpoint is called by EARL's orchestrator during simulations.
    
    Requires X-API-Key header with valid API key.
    """
    # Log full incoming request
    num_messages = len(request.messages)
    logger.info("=" * 60)
    logger.info(f"POST /chat - {num_messages} messages")
    logger.info("=" * 60)
    
    # Log all headers
    logger.info("Headers:")
    for header_name, header_value in http_request.headers.items():
        # Mask sensitive headers
        if header_name.lower() in ["x-api-key", "authorization"]:
            masked_value = header_value[:8] + "..." if len(header_value) > 8 else "***"
            logger.info(f"  {header_name}: {masked_value}")
        else:
            logger.info(f"  {header_name}: {header_value}")
    
    # Log all messages in the dialogue
    logger.info("Messages:")
    for i, msg in enumerate(request.messages):
        content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        logger.info(f"  [{i}] {msg.role}: {content_preview}")
    
    # Log patient context if present
    if request.patient_context:
        logger.info(f"Patient context: {json.dumps(request.patient_context, indent=2)}")
    
    # Check if we have a configured client
    if LLM_PROVIDER == "gemini" and not gemini_client:
        logger.error("Gemini not configured")
        raise HTTPException(
            status_code=503,
            detail="Gemini client not configured. Set GEMINI_API_KEY environment variable."
        )
    if LLM_PROVIDER == "openai" and not openai_client:
        logger.error("OpenAI not configured")
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not configured. Set OPENAI_API_KEY environment variable."
        )
    
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Log what we received
        logger.info(f"Received {len(messages)} messages")
        for i, m in enumerate(messages):
            logger.debug(f"  [{i}] {m['role']}: {m['content'][:50]}...")
        
        # Detect if this is doctor-initiated (no substantive patient messages)
        # A "substantive" patient message is one that's NOT a prompt like "Please greet me"
        greeting_prompts = ["please greet", "greet me", "how can you help", "please begin"]
        
        def is_real_patient_message(msg: dict) -> bool:
            """Check if this is a real patient message (not a prompt)."""
            if msg["role"] not in ("user", "patient"):
                return False
            content = msg["content"].lower().strip()
            # Check if it's just a greeting prompt
            for prompt in greeting_prompts:
                if prompt in content:
                    return False
            # Must have some actual content
            return len(content) > 10
        
        real_patient_messages = [m for m in messages if is_real_patient_message(m)]
        
        # Determine mode and system prompt
        if not real_patient_messages:
            # Doctor initiates - generate a greeting
            logger.info("MODE: Doctor initiates conversation (generating greeting)")
            system_prompt = DOCTOR_GREETING_PROMPT
            # Clear messages so LLM just generates a greeting
            messages = []
        else:
            # Patient has spoken - respond to them
            logger.info(f"MODE: Responding to patient ({len(real_patient_messages)} real patient message(s))")
            system_prompt = DOCTOR_SYSTEM_PROMPT
            if request.patient_context:
                system_prompt += f"\n\nPatient Context:\n{json.dumps(request.patient_context, indent=2)}"
        
        # Generate response
        logger.info(f"Generating response with {LLM_PROVIDER}...")
        if LLM_PROVIDER == "gemini":
            doctor_response = generate_with_gemini(messages, system_prompt)
            model = GEMINI_MODEL
        else:
            doctor_response = generate_with_openai(messages, system_prompt)
            model = OPENAI_MODEL
        
        # Log response
        response_preview = doctor_response[:100] + "..." if len(doctor_response) > 100 else doctor_response
        logger.success(f"Response: {response_preview}")
        
        return ChatResponse(
            response=doctor_response,
            model=model,
            provider=LLM_PROVIDER
        )
        
    except Exception as e:
        logger.exception(f"Error generating response: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("External Doctor Service")
    logger.info("=" * 50)
    logger.info(f"Port: {PORT}")
    logger.info(f"Provider: {LLM_PROVIDER}")
    logger.info(f"Model: {GEMINI_MODEL if LLM_PROVIDER == 'gemini' else OPENAI_MODEL}")
    
    if LLM_PROVIDER == "gemini":
        logger.info(f"Gemini configured: {gemini_client is not None}")
    else:
        logger.info(f"OpenAI configured: {openai_client is not None}")
    
    # API Key auth status
    if EXTERNAL_DOCTOR_API_KEY:
        logger.info(f"API Key auth: ENABLED (key: {EXTERNAL_DOCTOR_API_KEY[:8]}...)")
    else:
        logger.warning("API Key auth: DISABLED (all requests allowed)")
    
    logger.info("=" * 50)
    
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set!")
    elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set!")
    
    logger.info(f"Starting server on http://localhost:{PORT}")
    logger.info(f"API docs at http://localhost:{PORT}/docs")
    logger.info("To expose via Cloudflare tunnel:")
    logger.info(f"  cloudflared tunnel --url http://localhost:{PORT}")
    
    if EXTERNAL_DOCTOR_API_KEY:
        logger.info("Test with:")
        logger.info(f'  curl -X POST http://localhost:{PORT}/chat \\')
        logger.info(f'       -H "Content-Type: application/json" \\')
        logger.info(f'       -H "X-API-Key: {EXTERNAL_DOCTOR_API_KEY}" \\')
        logger.info(f'       -d \'{{"messages": [{{"role": "user", "content": "Hello"}}]}}\'')
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
