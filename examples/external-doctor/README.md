# External Doctor Service

A sample external doctor service that demonstrates how to integrate your own AI doctor with the EARL evaluation platform.

## Overview

This service provides an AI-powered medical consultation endpoint that can be used by EARL to evaluate doctor-patient conversations. It supports both OpenAI and Google Gemini as the LLM backend.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `env.example` to `.env` and set your API key:

```bash
cp env.example .env
```

For Gemini (default):
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...
```

For OpenAI:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### 3. Run the service

```bash
python main.py
```

The service will start on http://localhost:9000.

### 4. Expose via Cloudflare Tunnel

For EARL to access your local service, expose it via Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://localhost:9000
```

This gives you a public URL like `https://xxx-xxx.trycloudflare.com`.

For a permanent subdomain (e.g., `aidoctor.onlyevals.com`), configure a named tunnel:

```bash
cloudflared tunnel login
cloudflared tunnel create aidoctor
cloudflared tunnel route dns aidoctor aidoctor.onlyevals.com
cloudflared tunnel run aidoctor
```

## API Reference

### Health Check

```
GET /
GET /health
```

Returns service status and configuration.

### Chat Endpoint

```
POST /chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Hello, I have a headache."}
  ],
  "patient_context": {
    "name": "John Doe",
    "age": 45
  }
}
```

Response:

```json
{
  "response": "I'm sorry to hear about your headache. Can you tell me more about when it started?",
  "model": "gemini-2.0-flash-exp",
  "provider": "gemini"
}
```

## Using with EARL SDK

```python
from earl_sdk import EarlClient, Environment
from earl_sdk.models import DoctorApiConfig

client = EarlClient(
    client_id="your-client-id",
    client_secret="your-secret",
    environment=Environment.TEST,
)

# Create a pipeline with your external doctor
pipeline = client.pipelines.create(
    name="my-doctor-eval",
    dimension_ids=["factuality", "empathy", "clarity"],
    patient_ids=["patient-123"],  # Get from default_pipeline
    doctor_config=DoctorApiConfig.external(
        api_url="https://xxx-xxx.trycloudflare.com/chat",
        api_key="optional-key",
    ),
)

# Start simulation
simulation = client.simulations.create(
    pipeline_name=pipeline.name,
    num_episodes=1,
)

# Wait for results
result = client.simulations.wait_for_completion(simulation.id)
print(f"Score: {result.summary['average_score']}/4")
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM provider: `openai` or `gemini` |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash-exp` | Gemini model to use |
| `DOCTOR_PORT` | `9000` | Server port |

## Conversation Modes

The doctor service automatically handles both conversation scenarios:

### Doctor Initiates
When called with no patient messages (or just a prompt like "please greet me"), the doctor generates a warm greeting:

```json
{"messages": []}
// or
{"messages": [{"role": "user", "content": "Please greet me"}]}
```

Response: "Hello! Welcome. I'm Dr. AI, and I'm here to help. How can I assist you today?"

### Patient Initiates/Responds
When called with actual patient messages, the doctor responds appropriately:

```json
{"messages": [
  {"role": "user", "content": "I've been having headaches for the past week."}
]}
```

Response: "I'm sorry to hear about your headaches. Can you tell me more about their location and intensity?"

## Doctor System Prompt

The default system prompt instructs the AI to act as a compassionate medical doctor:
- Listen to patient symptoms and concerns
- Ask relevant follow-up questions
- Show empathy and understanding
- Provide clear, accurate medical information
- Suggest appropriate next steps

You can customize the `DOCTOR_SYSTEM_PROMPT` variable in `main.py` to match your specific use case.

## Client-Driven Mode

For VPN/firewall scenarios where EARL can't reach your doctor API, use client-driven mode:

```python
from earl_sdk.models import DoctorApiConfig

# Create client-driven pipeline
pipeline = client.pipelines.create(
    name="vpn-doctor-eval",
    dimension_ids=["factuality", "empathy"],
    patient_ids=["Adrian_Cruickshank"],
    doctor_config=DoctorApiConfig.client_driven(),  # Your code orchestrates
    conversation_initiator="doctor",  # or "patient"
)

# Start simulation
simulation = client.simulations.create(pipeline_name=pipeline.name)

# Poll for episodes and orchestrate conversation
while True:
    episodes = client.simulations.get_episodes(simulation.id)
    for ep in episodes:
        if ep["status"] == "awaiting_doctor":
            # Call YOUR local doctor API
            doctor_msg = my_doctor_api.chat(ep["dialogue_history"])
            
            # Submit response to EARL
            client.simulations.submit_response(simulation.id, ep["episode_id"], doctor_msg)
```

