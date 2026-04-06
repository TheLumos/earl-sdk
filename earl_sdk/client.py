"""Main Earl SDK client."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional, Union

from .api import (
    CasesAPI,
    DimensionsAPI,
    PatientsAPI,
    PipelinesAPI,
    SimulationsAPI,
    VerifiersAPI,
)
from .auth import Auth0Client


class Environment(str, Enum):
    """Earl platform environments."""

    LOCAL = "local"
    DEV = "dev"
    TEST = "test"
    PROD = "prod"
    PRODUCTION = "prod"  # Alias for PROD

    def __str__(self) -> str:
        return self.value


class EnvironmentConfig:
    """
    Pre-configured environment URLs.

    These are the official Earl platform endpoints.
    Users select which environment to connect to.
    """

    # API endpoints for each environment
    API_URLS = {
        "local": os.getenv("EARL_LOCAL_API_URL", "http://localhost:8006/api/v1"),
        "dev": "https://earl-api.thelumos.dev/api/v1",
        "test": "https://earl-api.thelumos.xyz/api/v1",
        "prod": "https://earl-api.thelumos.ai/api/v1",
    }

    # Auth0 configuration for each environment
    AUTH0_DOMAINS = {
        "local": os.getenv("EARL_LOCAL_AUTH0_DOMAIN", "dev-f4675lf8h3k0i3me.us.auth0.com"),
        "dev": "dev-f4675lf8h3k0i3me.us.auth0.com",
        "test": "dev-f4675lf8h3k0i3me.us.auth0.com",
        "prod": "dev-f4675lf8h3k0i3me.us.auth0.com",
    }

    AUTH0_AUDIENCES = {
        "local": os.getenv("EARL_LOCAL_AUTH0_AUDIENCE", "https://earl-api.thelumos.dev"),
        "dev": "https://earl-api.thelumos.dev",
        "test": "https://earl-api.thelumos.xyz",
        "prod": "https://earl-api.thelumos.ai",
    }

    @classmethod
    def get_api_url(cls, environment: str | Environment) -> str:
        """Get the API URL for an environment."""
        env_key = str(environment).lower()
        if env_key == "production":
            env_key = "prod"
        if env_key not in cls.API_URLS:
            raise ValueError(
                f"Unknown environment: {environment}. Use 'local', 'dev', 'test', or 'prod'"
            )
        return cls.API_URLS[env_key]

    @classmethod
    def get_auth0_domain(cls, environment: str | Environment) -> str:
        """Get the Auth0 domain for an environment."""
        env_key = str(environment).lower()
        if env_key == "production":
            env_key = "prod"
        if env_key not in cls.AUTH0_DOMAINS:
            raise ValueError(
                f"Unknown environment: {environment}. Use 'local', 'dev', 'test', or 'prod'"
            )
        return cls.AUTH0_DOMAINS[env_key]

    @classmethod
    def get_auth0_audience(cls, environment: str | Environment) -> str:
        """Get the Auth0 audience for an environment."""
        env_key = str(environment).lower()
        if env_key == "production":
            env_key = "prod"
        if env_key not in cls.AUTH0_AUDIENCES:
            raise ValueError(
                f"Unknown environment: {environment}. Use 'local', 'dev', 'test', or 'prod'"
            )
        return cls.AUTH0_AUDIENCES[env_key]


class EarlClient:
    """
    Earl SDK Client - Main entry point for the Earl Medical Evaluation Platform.

    This client provides access to all Earl API resources:
    - dimensions: Evaluation criteria for doctor responses
    - patients: Simulated patients for testing
    - pipelines: Evaluation configurations with doctor APIs
    - simulations: Run evaluations and get results

    Example:
        ```python
        from earl_sdk import EarlClient, DoctorApiConfig

        # Initialize with Auth0 M2M credentials
        client = EarlClient(
            client_id="your-client-id",
            client_secret="your-client-secret",
            organization="org_xxx",
            environment="test",  # or "prod"
        )

        # List evaluation dimensions
        dimensions = client.dimensions.list()

        # List patients
        patients = client.patients.list()

        # Create pipeline with external doctor API
        pipeline = client.pipelines.create(
            name="my-evaluation",
            dimension_ids=["factuality", "empathy"],
            patient_ids=[p.id for p in patients[:5]],
            doctor_config=DoctorApiConfig.external(
                api_url="https://my-doctor.com/chat",
                api_key="my-key",
            ),
        )

        # Run simulation
        simulation = client.simulations.create(
            pipeline_name=pipeline.name,
            num_episodes=5,
        )

        # Wait for completion with progress callback
        result = client.simulations.wait_for_completion(
            simulation.id,
            on_progress=lambda s: print(f"{s.completed_episodes}/{s.total_episodes}"),
        )

        # Get complete report with all details
        report = client.simulations.get_report(simulation.id)
        print(f"Score: {report['summary']['average_score']:.2f}/4")
        ```

    Using internal doctor (EARL's built-in AI):
        ```python
        pipeline = client.pipelines.create(
            name="internal-test",
            dimension_ids=["factuality"],
            patient_ids=patient_ids,
            # Omit doctor_config to use internal doctor
        )
        ```
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        organization: str = "",
        environment: str | Environment = Environment.PROD,
        api_url: Optional[str] = None,
        service_api_urls: Optional[dict[str, str]] = None,
        auth0_domain: Optional[str] = None,
        auth0_audience: Optional[str] = None,
        request_timeout: int = 120,
    ):
        """
        Initialize the Earl client.

        Args:
            client_id: Auth0 M2M application client ID
            client_secret: Auth0 M2M application client secret
            organization: Auth0 organization ID (org_xxx format). Optional for
                         M2M clients that have organization configured in Auth0.
            environment: Environment to connect to - "test" or "prod" (default)
            api_url: Optional global API base URL override for all services
                     (e.g. "http://localhost:8006/api/v1").
            service_api_urls: Optional per-service API URL overrides. Supported keys:
                             "cases", "dimensions", "patients", "pipelines", "simulations",
                             "verifiers" (generic Lumos catalog at ``GET /additional-verifiers``).
                             Values should be full base URLs ending with "/api/v1".
            auth0_domain: Optional Auth0 domain override (uses environment default if unset).
            auth0_audience: Optional Auth0 audience override (uses environment default if unset).
            request_timeout: HTTP request timeout in seconds (default 120).
                             Use a higher value when polling long-running simulations.

        Example:
            ```python
            # Test environment
            client = EarlClient(
                client_id="test-client-id",
                client_secret="test-secret",
                organization="org_abc123",
                environment="test",
            )

            # Longer timeout for simulation polling
            client = EarlClient(..., request_timeout=180)
            ```

        Note:
            Each environment requires its own set of credentials.
            Contact support@thelumos.ai to get credentials for each environment.
        """
        # Resolve environment
        if isinstance(environment, str):
            env_str = environment.lower()
            if env_str == "production":
                env_str = "prod"
            self._environment = env_str
        else:
            self._environment = str(environment)

        # Global API URL (fallback for all service clients)
        env_default_api_url = EnvironmentConfig.get_api_url(self._environment)
        self._api_url = (api_url or os.getenv("EARL_API_URL") or env_default_api_url).rstrip("/")

        # Per-service URL overrides (constructor args take precedence over env vars)
        service_api_urls = service_api_urls or {}
        self._service_api_urls = {
            "cases": (
                service_api_urls.get("cases") or os.getenv("EARL_CASES_API_URL") or self._api_url
            ).rstrip("/"),
            "dimensions": (
                service_api_urls.get("dimensions")
                or os.getenv("EARL_DIMENSIONS_API_URL")
                or self._api_url
            ).rstrip("/"),
            "patients": (
                service_api_urls.get("patients")
                or os.getenv("EARL_PATIENTS_API_URL")
                or self._api_url
            ).rstrip("/"),
            "pipelines": (
                service_api_urls.get("pipelines")
                or os.getenv("EARL_PIPELINES_API_URL")
                or self._api_url
            ).rstrip("/"),
            "simulations": (
                service_api_urls.get("simulations")
                or os.getenv("EARL_SIMULATIONS_API_URL")
                or self._api_url
            ).rstrip("/"),
            "verifiers": (
                service_api_urls.get("verifiers")
                or os.getenv("EARL_VERIFIERS_API_URL")
                or self._api_url
            ).rstrip("/"),
        }

        if os.getenv("EARL_CASES_API_URL"):
            import warnings

            warnings.warn(
                "EARL_CASES_API_URL is deprecated. Cases are now served through the main orchestrator API. "
                "Remove this env var; the SDK will route cases through the orchestrator automatically.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Initialize authentication with environment-specific Auth0 config
        self._auth = Auth0Client(
            client_id=client_id,
            client_secret=client_secret,
            organization=organization,
            domain=auth0_domain
            or os.getenv("EARL_AUTH0_DOMAIN")
            or EnvironmentConfig.get_auth0_domain(self._environment),
            audience=auth0_audience
            or os.getenv("EARL_AUTH0_AUDIENCE")
            or EnvironmentConfig.get_auth0_audience(self._environment),
        )

        self._request_timeout = request_timeout

        # Initialize API clients
        self._cases = CasesAPI(self._auth, self._service_api_urls["cases"], self._request_timeout)
        self._dimensions = DimensionsAPI(
            self._auth, self._service_api_urls["dimensions"], self._request_timeout
        )
        self._patients = PatientsAPI(
            self._auth, self._service_api_urls["patients"], self._request_timeout
        )
        self._pipelines = PipelinesAPI(
            self._auth, self._service_api_urls["pipelines"], self._request_timeout
        )
        self._simulations = SimulationsAPI(
            self._auth, self._service_api_urls["simulations"], self._request_timeout
        )
        self._verifiers = VerifiersAPI(
            self._auth, self._service_api_urls["verifiers"], self._request_timeout
        )

    @property
    def cases(self) -> CasesAPI:
        """Access the Cases API for pre-defined evaluation scenarios."""
        return self._cases

    @property
    def dimensions(self) -> DimensionsAPI:
        """Access the Dimensions API for evaluation criteria."""
        return self._dimensions

    @property
    def patients(self) -> PatientsAPI:
        """Access the Patients API for simulated patients."""
        return self._patients

    @property
    def pipelines(self) -> PipelinesAPI:
        """Access the Pipelines API for evaluation configurations."""
        return self._pipelines

    @property
    def simulations(self) -> SimulationsAPI:
        """Access the Simulations API for running evaluations."""
        return self._simulations

    @property
    def verifiers(self) -> VerifiersAPI:
        """Access the verifiers catalog API (generic Lumos gates + scoring dimensions)."""
        return self._verifiers

    @property
    def organization(self) -> str:
        """Get the current organization ID."""
        return self._auth.organization

    @property
    def environment(self) -> str:
        """Get the current environment (test or production)."""
        return self._environment

    @property
    def api_url(self) -> str:
        """Get the global API URL fallback."""
        return self._api_url

    @property
    def service_api_urls(self) -> dict[str, str]:
        """Get effective per-service API base URLs."""
        return dict(self._service_api_urls)

    def test_connection(self) -> bool:
        """
        Test the connection and authentication.

        Returns:
            True if connection is successful

        Raises:
            AuthenticationError: If authentication fails
        """
        # Try to fetch dimensions as a simple connectivity test
        try:
            self._dimensions.list()
            return True
        except Exception:
            raise

    def __repr__(self) -> str:
        return f"EarlClient(environment={self._environment!r}, organization={self._auth.organization!r})"
