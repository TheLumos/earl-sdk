"""
Earl SDK — Python client for the Earl medical evaluation platform.

Authentication model
--------------------
The SDK talks to the orchestrator with Auth0-issued JWTs. Tokens are
identically shaped regardless of how they were obtained — backend authorizes
off the ``org_id`` claim in all cases.

* **Humans** should authenticate interactively with the ``earl`` CLI::

      earl login              # browser + PKCE (default)
      earl login --headless   # device flow (no browser)

  The CLI caches tokens in the OS keychain; the Python client picks them up
  automatically when you don't pass explicit credentials.

* **Automation / CI** should use service-account credentials provisioned by
  an org admin (``earl service-account create``). Pass them to
  :class:`EarlClient` directly or via env vars::

      EARL_CLIENT_ID=...        # Auth0 client_id of the service account
      EARL_CLIENT_SECRET=...    # shown exactly once on creation
      EARL_ORG_ID=org_xxx       # organization id to scope the token to
      EARL_ENVIRONMENT=staging  # optional: staging | prod | dev

Quick start (automation / CI)::

    from earl_sdk import EarlClient, DoctorApiConfig

    client = EarlClient(
        client_id="your-sa-client-id",
        client_secret="your-sa-client-secret",
        organization="org_xxx",
        environment="staging",
    )

    cases = client.cases.list()
    pipeline = client.pipelines.create(
        name="my-evaluation",
        case_id="carla-hypertension-yasmin",
        doctor_config=DoctorApiConfig.external(
            api_url="https://your-doctor-api.com/chat",
            api_key="your-api-key",
        ),
    )
    simulation = client.simulations.create(pipeline_name=pipeline.name, num_episodes=5)
    completed = client.simulations.wait_for_completion(simulation.id)
    report = client.simulations.get_report(simulation.id)

Quick start (interactive, using cached CLI login)::

    # First run `earl login` once in a terminal, then:
    from earl_sdk import EarlClient
    client = EarlClient(environment="staging")   # picks up OS-keychain token
    cases = client.cases.list()

Environment variables (all optional)
------------------------------------
Authentication (service accounts):

* ``EARL_CLIENT_ID``       — service-account client id
* ``EARL_CLIENT_SECRET``   — service-account client secret
* ``EARL_ORG_ID``          — organization id (canonical; preferred)
* ``EARL_ORGANIZATION``    — deprecated alias for ``EARL_ORG_ID``

Endpoint selection:

* ``EARL_ENVIRONMENT``     — ``dev`` | ``staging`` | ``prod`` (default: ``prod``)
* ``EARL_API_URL``, ``EARL_AUTH0_DOMAIN``, ``EARL_AUTH0_AUDIENCE`` — overrides
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

