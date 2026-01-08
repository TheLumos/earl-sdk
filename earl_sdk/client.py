"""Main Earl SDK client."""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from .auth import Auth0Client
from .api import DimensionsAPI, PatientsAPI, PipelinesAPI, SimulationsAPI, RateLimitsAPI


class Environment(str, Enum):
    """Earl platform environments."""
    
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
        "dev": "https://dev-api.onlyevals.com",
        "test": "https://test-api.thelumos.xyz",
        "prod": "https://api.earl.thelumos.ai",
    }
    
    # Auth0 configuration for each environment
    AUTH0_DOMAINS = {
        "dev": "dev-f4675lf8h3k0i3me.us.auth0.com",
        "test": "dev-f4675lf8h3k0i3me.us.auth0.com",
        "prod": "thelumos.us.auth0.com",
    }
    
    AUTH0_AUDIENCES = {
        "dev": "https://api.onlyevals.com",
        "test": "https://api.earl.thelumos.xyz",
        "prod": "https://api.earl.thelumos.xyz",
    }
    
    @classmethod
    def get_api_url(cls, environment: str | Environment) -> str:
        """Get the API URL for an environment."""
        env_key = str(environment).lower()
        if env_key == "production":
            env_key = "prod"
        if env_key not in cls.API_URLS:
            raise ValueError(f"Unknown environment: {environment}. Use 'dev', 'test', or 'prod'")
        return cls.API_URLS[env_key]
    
    @classmethod
    def get_auth0_domain(cls, environment: str | Environment) -> str:
        """Get the Auth0 domain for an environment."""
        env_key = str(environment).lower()
        if env_key == "production":
            env_key = "prod"
        if env_key not in cls.AUTH0_DOMAINS:
            raise ValueError(f"Unknown environment: {environment}. Use 'dev', 'test', or 'prod'")
        return cls.AUTH0_DOMAINS[env_key]
    
    @classmethod
    def get_auth0_audience(cls, environment: str | Environment) -> str:
        """Get the Auth0 audience for an environment."""
        env_key = str(environment).lower()
        if env_key == "production":
            env_key = "prod"
        if env_key not in cls.AUTH0_AUDIENCES:
            raise ValueError(f"Unknown environment: {environment}. Use 'dev', 'test', or 'prod'")
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
            dimension_ids=["accuracy", "empathy"],
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
            dimension_ids=["accuracy"],
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
    ):
        """
        Initialize the Earl client.
        
        Args:
            client_id: Auth0 M2M application client ID
            client_secret: Auth0 M2M application client secret
            organization: Auth0 organization ID (org_xxx format). Optional for
                         M2M clients that have organization configured in Auth0.
            environment: Environment to connect to - "test" or "prod" (default)
        
        Example:
            ```python
            # Test environment
            client = EarlClient(
                client_id="test-client-id",
                client_secret="test-secret",
                organization="org_abc123",
                environment="test",
            )
            
            # Production (default environment)
            client = EarlClient(
                client_id="prod-client-id",
                client_secret="prod-secret",
                organization="org_abc123",
            )
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
        
        # Get environment-specific URLs
        self._api_url = EnvironmentConfig.get_api_url(self._environment)
        
        # Initialize authentication with environment-specific Auth0 config
        self._auth = Auth0Client(
            client_id=client_id,
            client_secret=client_secret,
            organization=organization,
            domain=EnvironmentConfig.get_auth0_domain(self._environment),
            audience=EnvironmentConfig.get_auth0_audience(self._environment),
        )
        
        # Initialize API clients
        self._dimensions = DimensionsAPI(self._auth, self._api_url)
        self._patients = PatientsAPI(self._auth, self._api_url)
        self._pipelines = PipelinesAPI(self._auth, self._api_url)
        self._simulations = SimulationsAPI(self._auth, self._api_url)
        self._rate_limits = RateLimitsAPI(self._auth, self._api_url)
    
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
    def rate_limits(self) -> RateLimitsAPI:
        """Access the Rate Limits API to check your usage limits."""
        return self._rate_limits
    
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
        """Get the current API URL."""
        return self._api_url
    
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

