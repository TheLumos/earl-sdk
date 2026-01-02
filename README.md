# Earl SDK for Python

Python SDK for the Earl Medical Evaluation Platform. Evaluate your medical AI/doctor chatbots against realistic patient simulations.

## Installation

```bash
pip install earl-sdk
```

Or install from source:
```bash
cd sdk/python
pip install -e .
```

## Quick Start

```python
from earl_sdk import EarlClient, DoctorApiConfig

# Initialize with your Auth0 M2M credentials
client = EarlClient(
    client_id="your-m2m-client-id",
    client_secret="your-m2m-client-secret",
    organization="org_xxx",  # Your Auth0 organization ID
    environment="test",      # "test" or "prod" (default)
)

# Test the connection
client.test_connection()
print(f"Connected to {client.environment}!")

# List available dimensions
dimensions = client.dimensions.list()
for dim in dimensions:
    print(f"  {dim.id}: {dim.name}")

# List patients
patients = client.patients.list()

# Create a pipeline with your doctor API
pipeline = client.pipelines.create(
    name="my-evaluation",
    dimension_ids=["accuracy", "empathy", "safety"],
    patient_ids=[p.id for p in patients[:5]],
    doctor_config=DoctorApiConfig.external(
        api_url="https://your-doctor-api.com/chat",
        api_key="your-api-key",
    ),
)

# Run a simulation
simulation = client.simulations.create(
    pipeline_name=pipeline.name,
    num_episodes=5,
)

# Wait for completion with progress callback
def show_progress(sim):
    pct = int(sim.progress * 100)
    print(f"Progress: {sim.completed_episodes}/{sim.total_episodes} ({pct}%)")

completed = client.simulations.wait_for_completion(
    simulation.id,
    on_progress=show_progress,
)

# Get results
results = client.simulations.get_results(simulation.id)
print(f"Overall Score: {results.overall_score:.2f}/4")
```

## Environments

Earl provides two main environments:

| Environment | Description | API URL |
|-------------|-------------|---------|
| `test` | Testing/staging | https://test-api.thelumos.xyz |
| `prod` | Production (default) | https://api.earl.thelumos.ai |

```python
from earl_sdk import EarlClient, Environment

# Test environment
test_client = EarlClient(
    client_id="test-client-id",
    client_secret="test-secret",
    organization="org_xxx",
    environment="test",
)

# Production environment (default)
prod_client = EarlClient(
    client_id="prod-client-id",
    client_secret="prod-secret",
    organization="org_xxx",
)

# Check which environment you're connected to
print(f"Environment: {client.environment}")
print(f"API URL: {client.api_url}")
```

## Doctor API Configuration

### Using EARL's Internal Doctor (Default)

If you don't specify a `doctor_config`, EARL uses its built-in AI doctor:

```python
pipeline = client.pipelines.create(
    name="internal-doctor-test",
    dimension_ids=["accuracy", "empathy"],
    patient_ids=patient_ids,
    # No doctor_config = uses internal doctor
)
```

### Using Your External Doctor API

Test your own doctor API:

```python
from earl_sdk import DoctorApiConfig

# Create external doctor config
doctor_config = DoctorApiConfig.external(
    api_url="https://your-doctor.com/chat",
    api_key="your-secret-key",
)

pipeline = client.pipelines.create(
    name="my-doctor-test",
    dimension_ids=["accuracy", "empathy", "safety"],
    patient_ids=patient_ids,
    doctor_config=doctor_config,
)
```

### Validate Your Doctor API First

Before creating a pipeline, you can validate your doctor API is reachable:

```python
try:
    result = client.pipelines.validate_doctor_api(
        api_url="https://your-doctor.com/chat",
        api_key="your-key",
    )
    print(f"✓ {result['message']}")
except ValidationError as e:
    print(f"✗ {e}")
```

### Doctor API Contract

Your doctor API must accept POST requests with this format:

```json
{
  "messages": [
    {"role": "user", "content": "Patient message..."},
    {"role": "assistant", "content": "Previous doctor response..."}
  ],
  "patient_context": {"patient_id": "..."}
}
```

And return:

```json
{
  "response": "Doctor's response text..."
}
```

Authentication is via `X-API-Key` header.

## Conversation Flow Configuration

You can configure who initiates the conversation:

### Patient-Initiated (Default)

The patient sends the first message describing their symptoms. This is the typical telemedicine flow:

```python
pipeline = client.pipelines.create(
    name="telemedicine-eval",
    dimension_ids=["accuracy", "empathy"],
    patient_ids=patient_ids,
    conversation_initiator="patient",  # Default
)
# Patient: "I've been having headaches for a week..."
# Doctor: "I'm sorry to hear that. Can you describe the pain?"
```

### Doctor-Initiated

The doctor sends the first message (greeting/opening). Useful for proactive care or follow-up scenarios:

```python
pipeline = client.pipelines.create(
    name="proactive-care-eval",
    dimension_ids=["empathy", "thoroughness"],
    patient_ids=patient_ids,
    conversation_initiator="doctor",
)
# Doctor: "Hello, I'm Dr. Smith. What brings you in today?"
# Patient: "I've been feeling dizzy lately..."
```

### Check Pipeline's Initiator

```python
pipeline = client.pipelines.get("my-pipeline")
print(f"Initiator: {pipeline.conversation_initiator}")  # "patient" or "doctor"
```

## Working with Simulations

### Start a Simulation

```python
simulation = client.simulations.create(
    pipeline_name="my-pipeline",
    num_episodes=5,      # Number of patient conversations
    parallel_count=2,    # Parallel episodes (1-10)
)

print(f"Simulation ID: {simulation.id}")
print(f"Status: {simulation.status}")
```

### Track Progress

```python
# Get current status
sim = client.simulations.get(simulation_id)
print(f"Progress: {sim.completed_episodes}/{sim.total_episodes}")
print(f"Status: {sim.status}")

# Wait with progress callback
def on_progress(sim):
    print(f"  {sim.completed_episodes}/{sim.total_episodes} completed")

completed = client.simulations.wait_for_completion(
    simulation_id,
    poll_interval=5.0,   # Check every 5 seconds
    timeout=600.0,       # 10 minute timeout
    on_progress=on_progress,
)
```

### Get Episode Details

```python
# Get all episodes
episodes = client.simulations.get_episodes(simulation_id)
for ep in episodes:
    print(f"Episode {ep['episode_number']}: {ep['status']}")
    if ep['status'] == 'completed':
        print(f"  Score: {ep['total_score']:.2f}/4")

# Get single episode with full dialogue
episode = client.simulations.get_episode(simulation_id, episode_id)
for turn in episode.get('dialogue_history', []):
    print(f"  {turn['role']}: {turn['content'][:50]}...")
```

### Get Results

```python
results = client.simulations.get_results(simulation_id)

print(f"Overall Score: {results.overall_score:.2f}/4")
print(f"Patients: {results.successful_patients}/{results.total_patients}")

print("\nDimension Averages:")
for dim_id, avg in results.dimension_averages.items():
    print(f"  {dim_id}: {avg:.2f}/4")
```

### Get Complete Report

For a complete report with all episode data, dialogue history, and judge feedback in one call:

```python
report = client.simulations.get_report(simulation_id)

# Summary statistics
summary = report['summary']
print(f"Completed: {summary['completed']}/{summary['total_episodes']}")
print(f"Average Score: {summary['average_score']:.2f}/4")

# Per-dimension breakdown
print("\nDimension Scores:")
for dim_id, stats in report.get('dimension_scores', {}).items():
    print(f"  {dim_id}: avg={stats['average']:.2f}, min={stats['min']}, max={stats['max']}")

# All episodes with full details
print("\nEpisodes:")
for ep in report['episodes']:
    print(f"\n  Episode {ep['episode_number']}: {ep['patient_name']}")
    print(f"    Score: {ep['total_score']}")
    print(f"    Dialogue ({ep['dialogue_turns']} turns):")
    for turn in ep.get('dialogue_history', [])[:3]:  # First 3 turns
        role = turn['role'].upper()
        content = turn['content'][:60] + "..." if len(turn['content']) > 60 else turn['content']
        print(f"      {role}: {content}")
```

## Rate Limits

API calls are rate-limited per organization. You can check your current limits programmatically:

```python
# Get your organization's rate limits
limits = client.rate_limits.get()

print(f"Organization: {limits['organization_id']}")
print(f"Per minute: {limits['limits']['per_minute']}")
print(f"Per hour: {limits['limits']['per_hour']}")
print(f"Per day: {limits['limits']['per_day']}")

# Category-specific limits
print("\nEffective limits by category:")
for category, limit in limits['effective_limits'].items():
    print(f"  {category}: {limit}/min")

# Quick check for a specific category
sim_limit = client.rate_limits.get_effective_limit("simulations")
print(f"\nSimulations limit: {sim_limit}/min")
```

### Rate Limit Headers

Every API response includes rate limit headers:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed in current window |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Unix timestamp when the window resets |

When you exceed the limit, you'll receive `HTTP 429 Too Many Requests`.

## Error Handling

```python
from earl_sdk import EarlClient
from earl_sdk.exceptions import (
    EarlError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
    SimulationError,
)

try:
    results = client.simulations.get_results("invalid-id")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
except AuthorizationError as e:
    print(f"Access denied: {e.message}")
except NotFoundError as e:
    print(f"Not found: {e}")
except ValidationError as e:
    print(f"Invalid request: {e.message}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except SimulationError as e:
    print(f"Simulation {e.simulation_id} failed: {e.message}")
except ServerError as e:
    print(f"Server error: {e.message}")
except EarlError as e:
    print(f"Unexpected error: {e}")
```

## API Reference

### EarlClient

Main entry point for the SDK.

```python
client = EarlClient(
    client_id="...",
    client_secret="...",
    organization="org_xxx",
    environment="test",  # or "prod"
)

# Properties
client.environment   # Current environment
client.api_url       # API URL
client.organization  # Organization ID

# Test connection
client.test_connection()  # Returns True or raises
```

### DimensionsAPI

```python
# List all dimensions
dimensions = client.dimensions.list(include_custom=True)

# Get a specific dimension
dimension = client.dimensions.get("accuracy")

# Create custom dimension
dim = client.dimensions.create(
    name="Medical Accuracy",
    description="How accurate is the medical information",
    category="quality",
    weight=1.0,
)
```

### PatientsAPI

```python
# List patients
patients = client.patients.list(
    difficulty="medium",  # easy, medium, hard
    limit=100,
    offset=0,
)

# Get a specific patient
patient = client.patients.get("patient_id")
```

### PipelinesAPI

```python
# List pipelines (summary view)
pipelines = client.pipelines.list(active_only=True)

# Get pipeline with FULL details
pipeline = client.pipelines.get("pipeline_name")
print(f"Doctor: {pipeline.doctor_api.type}")  # 'internal' or 'external'
print(f"Patients: {pipeline.patient_ids}")
print(f"Dimensions: {pipeline.dimension_ids}")
print(f"Initiator: {pipeline.conversation.initiator}")  # 'patient' or 'doctor'

# Create pipeline
pipeline = client.pipelines.create(
    name="my-pipeline",
    dimension_ids=["accuracy", "empathy"],
    patient_ids=["patient1", "patient2"],
    doctor_config=DoctorApiConfig.external(...),
    description="My evaluation pipeline",
    validate_doctor=True,  # Validate API before creating
)

# Validate external doctor API
result = client.pipelines.validate_doctor_api(
    api_url="https://...",
    api_key="...",
)

# Update pipeline
client.pipelines.update(
    "pipeline_name",
    description="Updated description",
)

# Delete pipeline
client.pipelines.delete("pipeline_name")
```

### SimulationsAPI

```python
# List simulations
simulations = client.simulations.list(
    pipeline_id="my-pipeline",
    status=SimulationStatus.COMPLETED,
    limit=50,
)

# Get simulation
sim = client.simulations.get("simulation_id")

# Create simulation
sim = client.simulations.create(
    pipeline_name="my-pipeline",
    num_episodes=5,
    parallel_count=2,
)

# Wait for completion
completed = client.simulations.wait_for_completion(
    simulation_id,
    poll_interval=5.0,
    timeout=600.0,
    on_progress=lambda s: print(f"{s.progress:.0%}"),
)

# Get episodes
episodes = client.simulations.get_episodes(
    simulation_id,
    include_dialogue=True,
)

# Get single episode
episode = client.simulations.get_episode(simulation_id, episode_id)

# Get results
results = client.simulations.get_results(simulation_id)

# Get complete report (all data in one call)
report = client.simulations.get_report(simulation_id)

# Cancel simulation
client.simulations.cancel(simulation_id)
```

### RateLimitsAPI

```python
# Get all rate limit info
limits = client.rate_limits.get()
# Returns: {
#   "organization_id": "org_xxx",
#   "limits": {"per_minute": 60, "per_hour": 1000, "per_day": 10000},
#   "category_limits": {"evaluations": 10, "pipelines": 60, ...},
#   "effective_limits": {"evaluations": 10, "pipelines": 60, ...},
#   "headers_info": {...},
# }

# Get effective limit for a specific category
limit = client.rate_limits.get_effective_limit("simulations")  # Returns int
```

## Models

### Simulation

```python
sim.id                  # Simulation ID
sim.pipeline_name        # Pipeline name
sim.status              # SimulationStatus enum
sim.total_episodes      # Total episodes
sim.completed_episodes  # Completed episodes
sim.progress            # Progress ratio (0.0-1.0)
sim.error_message       # Error message if failed
sim.summary             # Summary dict if completed
```

### SimulationStatus

```python
from earl_sdk import SimulationStatus

SimulationStatus.PENDING
SimulationStatus.RUNNING
SimulationStatus.COMPLETED
SimulationStatus.FAILED
SimulationStatus.CANCELLED
```

### DoctorApiConfig

```python
from earl_sdk import DoctorApiConfig

# Internal doctor (EARL's built-in AI)
config = DoctorApiConfig.internal()
config = DoctorApiConfig.internal(prompt="Custom system prompt")

# External doctor (your API)
config = DoctorApiConfig.external(
    api_url="https://your-api.com/chat",
    api_key="your-key",
)
```

## Score Scale

Evaluation scores are on a 1-4 scale:

| Score | Meaning |
|-------|---------|
| 1 | Poor |
| 2 | Fair |
| 3 | Good |
| 4 | Excellent |

## Support

- Email: support@thelumos.ai

## License

MIT License - see LICENSE file for details.
