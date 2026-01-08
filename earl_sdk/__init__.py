"""
Earl SDK - Python client for Earl Medical Evaluation Platform.

A simple SDK for evaluating AI medical assistants against simulated patients.

Quick Start:
    ```python
    from earl_sdk import EarlClient, DoctorApiConfig
    
    # Initialize client with Auth0 M2M credentials
    client = EarlClient(
        client_id="your-m2m-client-id",
        client_secret="your-m2m-client-secret",
        organization="org_xxx",  # Your Auth0 organization ID
        environment="test",      # "test" or "prod" (default)
    )
    
    # List available evaluation dimensions
    dimensions = client.dimensions.list()
    for dim in dimensions:
        print(f"  {dim.id}: {dim.name}")
    
    # List available patients
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
    
    # Wait for completion with progress updates
    def show_progress(sim):
        print(f"Progress: {sim.completed_episodes}/{sim.total_episodes}")
    
    completed = client.simulations.wait_for_completion(
        simulation.id,
        on_progress=show_progress,
    )
    
    # Get complete report with all details
    report = client.simulations.get_report(simulation.id)
    print(f"Overall score: {report['summary']['average_score']:.2f}/4")
    ```

For internal doctor (EARL's built-in AI doctor):
    ```python
    pipeline = client.pipelines.create(
        name="internal-test",
        dimension_ids=["accuracy", "empathy"],
        patient_ids=patient_ids,
        # No doctor_config = uses internal doctor
    )
    ```

Environment Variables (optional):
    - EARL_CLIENT_ID: Default client ID
    - EARL_CLIENT_SECRET: Default client secret
    - EARL_ORGANIZATION: Default organization ID
    - EARL_ENVIRONMENT: Default environment ("test" or "prod")
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"  # Fallback for editable installs without build

from .client import EarlClient, Environment
from .models import (
    Dimension,
    Patient,
    Pipeline,
    Simulation,
    SimulationStatus,
    DoctorApiConfig,
    ConversationConfig,
)
from .exceptions import (
    EarlError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
    SimulationError,
)

__all__ = [
    # Main client
    "EarlClient",
    "Environment",
    # Models
    "Dimension",
    "Patient", 
    "Pipeline",
    "Simulation",
    "SimulationStatus",
    "DoctorApiConfig",
    "ConversationConfig",
    # Exceptions
    "EarlError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    "SimulationError",
]

