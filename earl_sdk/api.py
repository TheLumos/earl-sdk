"""API clients for Earl SDK resources."""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, Any, List, Callable
from abc import ABC

from . import __version__
from .auth import Auth0Client
from .models import (
    Dimension,
    Patient,
    Pipeline,
    Simulation,
    SimulationStatus,
    DoctorApiConfig,
)
from .exceptions import (
    EarlError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
)


class BaseAPI(ABC):
    """Base class for API clients."""
    
    def __init__(self, auth: Auth0Client, base_url: str):
        self.auth = auth
        self.base_url = base_url.rstrip("/")
    
    def _request(
        self,
        method: str,
        path: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Make an authenticated API request."""
        url = f"{self.base_url}{path}"
        
        # Add query parameters
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        
        # Prepare request
        headers = self.auth.get_headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        headers["User-Agent"] = f"EarlSDK/{__version__}"
        
        body = json.dumps(data).encode("utf-8") if data else None
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body) if response_body else {}
                
        except urllib.error.HTTPError as e:
            self._handle_error(e)
        except urllib.error.URLError as e:
            raise EarlError(f"Failed to connect to API: {str(e)}")
    
    def _handle_error(self, error: urllib.error.HTTPError) -> None:
        """Handle HTTP errors and raise appropriate exceptions."""
        try:
            error_body = error.read().decode("utf-8")
            error_data = json.loads(error_body) if error_body else {}
        except (json.JSONDecodeError, Exception):
            error_data = {"message": str(error)}
        
        message = error_data.get("message", error_data.get("error", str(error)))
        
        if error.code == 401:
            self.auth.invalidate_token()
            raise AuthenticationError(message, details=error_data)
        elif error.code == 403:
            raise AuthorizationError(message, details=error_data)
        elif error.code == 404:
            resource_type = error_data.get("resource_type", "Resource")
            resource_id = error_data.get("resource_id", "unknown")
            raise NotFoundError(resource_type, resource_id)
        elif error.code == 400:
            raise ValidationError(message, details=error_data)
        elif error.code == 429:
            retry_after = error_data.get("retry_after")
            raise RateLimitError(retry_after)
        elif error.code >= 500:
            raise ServerError(message, details=error_data)
        else:
            raise EarlError(message, status_code=error.code, details=error_data)


class DimensionsAPI(BaseAPI):
    """API client for managing evaluation dimensions."""
    
    def list(self, include_custom: bool = True) -> list[Dimension]:
        """
        List all available dimensions.
        
        Args:
            include_custom: Include custom dimensions created by the organization
            
        Returns:
            List of Dimension objects
        """
        response = self._request("GET", "/dimensions", params={"include_custom": include_custom})
        return [Dimension.from_dict(d) for d in response.get("dimensions", [])]
    
    def get(self, dimension_id: str) -> Dimension:
        """
        Get a specific dimension by ID.
        
        Args:
            dimension_id: The dimension ID
            
        Returns:
            Dimension object
        """
        response = self._request("GET", f"/dimensions/{dimension_id}")
        return Dimension.from_dict(response)
    
    def create(
        self,
        name: str,
        description: str,
        category: str = "custom",
        weight: float = 1.0,
    ) -> Dimension:
        """
        Create a custom dimension.
        
        Args:
            name: Human-readable name
            description: What this dimension evaluates
            category: Category for grouping
            weight: Default weight (0.0 to 1.0)
            
        Returns:
            Created Dimension object
        """
        data = {
            "name": name,
            "description": description,
            "category": category,
            "weight": weight,
        }
        response = self._request("POST", "/dimensions", data=data)
        return Dimension.from_dict(response)


class PatientsAPI(BaseAPI):
    """API client for accessing simulated patients."""
    
    def list(
        self,
        difficulty: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Patient]:
        """
        List available patients.
        
        Patients have rich emotional and cognitive state, session-based
        conversation management, termination signals, and internal thoughts.
        
        Args:
            difficulty: Filter by difficulty (easy, medium, hard)
            tags: Filter by tags
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of Patient objects
            
        Example:
            >>> patients = client.patients.list()
            >>> for p in patients:
            ...     print(f"{p.id}: {p.name} - {p.description}")
        """
        params = {"limit": limit, "offset": offset}
        if difficulty:
            params["difficulty"] = difficulty
        if tags:
            params["tags"] = ",".join(tags)
        
        response = self._request("GET", "/patients", params=params)
        return [Patient.from_dict(p) for p in response.get("patients", [])]
    
    def get(self, patient_id: str) -> Patient:
        """
        Get a specific patient by ID.
        
        Args:
            patient_id: The patient ID
            
        Returns:
            Patient object
        """
        response = self._request("GET", f"/patients/{patient_id}")
        return Patient.from_dict(response)


class PipelinesAPI(BaseAPI):
    """API client for managing evaluation pipelines."""
    
    def _validate_external_doctor(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        """
        Validate that an external doctor API is reachable and can handle POST requests.
        
        Sends a test POST request to the exact URL provided (no path appending).
        The URL should be an OpenAI-compatible completions API endpoint.
        
        For serverless APIs (Modal, AWS Lambda, etc.) that may have cold starts,
        we use a warming strategy with retries and increasing timeouts.
        
        Validation passes if:
        - Any 2xx response is received
        - Any 4xx response except 401/403/404 (means endpoint exists, just different format)
        
        Validation fails if:
        - 401/403: Authentication/authorization error
        - 404: Endpoint not found
        - 5xx: Server error
        - Connection error: Cannot reach the URL after retries
        
        Args:
            api_url: The doctor API URL (used as-is, no path appending)
            api_key: Optional API key (sent as X-API-Key header)
            timeout: Base request timeout in seconds (will increase on retries)
            
        Raises:
            ValidationError: If the API is not reachable or returns an auth/server error
        """
        import ssl
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"Earl-SDK-Validator/{__version__}",
        }
        if api_key:
            # Support both header formats for broader API compatibility
            headers["X-API-Key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Create SSL context
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
                
        # Test POST to the actual endpoint (OpenAI-compatible completions API)
        # For external doctors, the URL provided IS the final endpoint - no path appending
        # The orchestrator uses the URL as-is for external doctor APIs
        # 
        # We don't rely on health checks - only the actual POST matters.
        # If we get any response back (2xx, 4xx except auth errors), it works.
        endpoint_url = api_url.rstrip("/")
        test_payload = json.dumps({
            "model": "default",  # OpenAI-compatible APIs require a model field
            "messages": [{"role": "user", "content": "Hello, I am testing the connection."}],
            "max_tokens": 50,
        }).encode("utf-8")
        
        # Warming strategy for cold-start APIs (Modal, Lambda, etc.)
        # First attempt: quick check with short timeout
        # Second attempt: warming call with longer timeout (cold start)
        # Third attempt: final check with extended timeout
        attempts = [
            {"timeout": timeout, "desc": "initial check"},
            {"timeout": 60.0, "desc": "warming (cold start)"},
            {"timeout": 30.0, "desc": "retry after warm"},
        ]
        
        last_error = None
        
        for attempt_num, attempt in enumerate(attempts, 1):
            attempt_timeout = attempt["timeout"]
            attempt_desc = attempt["desc"]
            
            try:
                req = urllib.request.Request(
                    endpoint_url,
                    data=test_payload,
                    headers=headers,
                    method="POST",
                )
                
                with urllib.request.urlopen(req, timeout=attempt_timeout, context=ctx) as response:
                    # Any 2xx response means the endpoint works
                    if 200 <= response.status < 300:
                        return  # Success!
                        
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    raise ValidationError(
                        f"External doctor API authentication failed (401 Unauthorized).\n"
                        f"URL: {endpoint_url}\n"
                        f"API key provided: {'Yes' if api_key else 'No'}\n"
                        f"Please verify your API key is correct."
                    )
                elif e.code == 403:
                    raise ValidationError(
                        f"External doctor API access forbidden (403 Forbidden).\n"
                        f"URL: {endpoint_url}\n"
                        f"Please verify your API key has the correct permissions."
                    )
                elif e.code == 404:
                    raise ValidationError(
                        f"External doctor API endpoint not found (404).\n"
                        f"URL: {endpoint_url}\n"
                        f"The orchestrator will POST to this exact URL.\n"
                        f"Please verify the URL is correct and accepts POST requests."
                    )
                elif e.code >= 500:
                    # Server error - might be transient, continue to retry
                    last_error = f"Server error ({e.code})"
                    continue
                else:
                    # Any other response (including 400, 422, etc.) means API is reachable
                    # Payload format might differ but that's OK - endpoint exists and responds
                    return  # Success - endpoint is reachable
                    
            except urllib.error.URLError as e:
                last_error = f"Cannot connect: {e.reason}"
                # Continue to next attempt (might be cold start)
                continue
                    
            except Exception as e:
                error_str = str(e)
                # Check for timeout-related errors
                if "timed out" in error_str.lower() or "timeout" in error_str.lower():
                    last_error = f"Request timed out after {attempt_timeout}s ({attempt_desc})"
                    # Continue to next attempt with longer timeout
                    continue
                else:
                    last_error = error_str
                    continue
        
        # If we get here, we couldn't connect after all retries
        raise ValidationError(
            f"Cannot reach external doctor API.\n"
            f"URL: {endpoint_url}\n"
            f"Error: {last_error}\n\n"
            f"Tried {len(attempts)} times with increasing timeouts (up to 60s for cold start).\n\n"
            f"The orchestrator will POST to this URL during simulations.\n"
            f"Please verify:\n"
            f"  1. The URL is correct and accessible\n"
            f"  2. The service is running and not paused\n"
            f"  3. Any firewalls or VPNs allow the connection\n"
            f"  4. For serverless APIs (Modal, Lambda): the service may need to be warmed up"
        )
    
    def validate_doctor_api(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> dict:
        """
        Validate an external doctor API before creating a pipeline.
        
        Use this to test your doctor API configuration before creating a pipeline.
        The method will check:
        1. The URL is reachable
        2. The API key is valid (if provided)
        3. The service responds correctly
        
        Args:
            api_url: Your doctor API URL (e.g., "https://my-doctor.com/chat")
            api_key: Your API key (if required)
            timeout: Request timeout in seconds (default: 10)
            
        Returns:
            Dict with validation result:
            {
                "valid": True,
                "url": "https://...",
                "message": "Doctor API is reachable and responding"
            }
            
        Raises:
            ValidationError: If the API is not reachable or authentication fails
            
        Example:
            >>> result = client.pipelines.validate_doctor_api(
            ...     api_url="https://my-doctor.com/chat",
            ...     api_key="my-secret-key"
            ... )
            >>> print(result)
            {'valid': True, 'url': 'https://my-doctor.com/chat', 'message': '...'}
        """
        self._validate_external_doctor(api_url, api_key, timeout)
        return {
            "valid": True,
            "url": api_url,
            "message": "Doctor API is reachable and responding correctly",
        }
    
    def list(self, active_only: bool = True) -> list[Pipeline]:
        """
        List all pipelines for the organization.
        
        Args:
            active_only: Only return active pipelines
            
        Returns:
            List of Pipeline objects
        """
        response = self._request("GET", "/pipelines", params={"active_only": active_only})
        return [Pipeline.from_dict(p) for p in response.get("pipelines", [])]
    
    def get(self, pipeline_name: str) -> Pipeline:
        """
        Get full details for a specific pipeline.
        
        Returns the complete pipeline configuration including doctor settings,
        patient IDs, dimension IDs, conversation config, and more.
        
        Args:
            pipeline_name: The unique name of the pipeline
            
        Returns:
            Pipeline object with full configuration:
            - name: Pipeline name
            - description: Pipeline description
            - patient_ids: List of patient IDs
            - dimension_ids: List of dimension IDs for evaluation
            - doctor_api: Doctor API configuration (DoctorApiConfig)
            - conversation: Conversation settings (who initiates)
            - created_at: Creation timestamp
            
        Example:
            ```python
            pipeline = client.pipelines.get("my-pipeline")
            print(f"Doctor type: {pipeline.doctor_api.type}")
            print(f"Patients: {len(pipeline.patient_ids)}")
            print(f"Dimensions: {pipeline.dimension_ids}")
            print(f"Initiator: {pipeline.conversation.initiator}")
            ```
        """
        response = self._request("GET", f"/pipelines/{pipeline_name}")
        return Pipeline.from_dict(response)
    
    def create(
        self,
        name: str,
        dimension_ids: list[str],
        doctor_config: Optional[DoctorApiConfig | dict] = None,
        patient_ids: Optional[list[str]] = None,
        description: Optional[str] = None,
        use_internal_doctor: bool = True,
        validate_doctor: bool = True,
        conversation_initiator: str = "patient",
        max_turns: int = 10,
    ) -> Pipeline:
        """
        Create a new evaluation pipeline.
        
        Args:
            name: Pipeline name
            dimension_ids: List of dimension IDs to evaluate
            doctor_config: Configuration for the doctor API (internal or external)
                          If None and use_internal_doctor=True, uses internal doctor
            patient_ids: Optional list of patient IDs to include in pipeline.
                         Use `client.patients.list()` to see available patients.
            description: Optional description
            use_internal_doctor: If True and doctor_config is None, use internal doctor
            validate_doctor: If True (default), validates external doctor API is reachable
                            and API key works before creating the pipeline. Set to False
                            to skip validation.
            conversation_initiator: Who sends the first message - "patient" or "doctor".
                - "patient": Patient sends first message (typical telemedicine flow).
                             Patient describes symptoms, doctor responds.
                - "doctor": Doctor sends first message (proactive care).
                            Doctor greets patient, patient responds.
                Default is "patient".
            max_turns: Maximum number of conversation turns (1-50, default 10).
                The conversation ends after this many turns. The patient will
                indicate they need to leave as the limit approaches.
            
        Returns:
            Created Pipeline object
            
        Raises:
            ValidationError: If external doctor API validation fails
            
        Examples:
            # Create pipeline with internal doctor (default)
            pipeline = client.pipelines.create(
                name="my-eval",
                dimension_ids=["factuality", "empathy"],
            )
            
            # Create pipeline with external doctor API
            pipeline = client.pipelines.create(
                name="my-eval",
                dimension_ids=["factuality", "empathy"],
                doctor_config=DoctorApiConfig.external(
                    api_url="https://my-doctor-api.com/chat",
                    api_key="my-api-key"
                ),
            )
            
            # Doctor-initiated conversation (doctor starts)
            pipeline = client.pipelines.create(
                name="proactive-care-eval",
                dimension_ids=["empathy", "thoroughness"],
                conversation_initiator="doctor",
            )
            
            # With specific patients
            patients = client.patients.list()
            pipeline = client.pipelines.create(
                name="my-eval",
                dimension_ids=["empathy", "communication"],
                patient_ids=[p.id for p in patients[:3]],
            )
            
            # Custom max turns (longer conversations)
            pipeline = client.pipelines.create(
                name="detailed-eval",
                dimension_ids=["thoroughness", "accuracy"],
                max_turns=30,  # Allow up to 30 turns
            )
            
            # Skip validation (not recommended)
            pipeline = client.pipelines.create(
                name="my-eval",
                dimension_ids=["factuality", "empathy"],
                doctor_config=DoctorApiConfig.external(...),
                validate_doctor=False,  # Skip API validation
            )
        """
        # Build doctor configuration
        if doctor_config is None:
            if use_internal_doctor:
                doctor = {"type": "internal"}
            else:
                raise ValueError("doctor_config is required when use_internal_doctor=False")
        elif isinstance(doctor_config, dict):
            doctor = doctor_config
        else:
            doctor = doctor_config.to_dict()
        
        # Validate external doctor API before creating pipeline
        if doctor.get("type") == "external":
            api_url = doctor.get("api_url")
            api_key = doctor.get("api_key")
            
            if not api_url:
                raise ValidationError(
                    "External doctor API requires 'api_url' to be set.\n"
                    "Use DoctorApiConfig.external(api_url='...', api_key='...')"
                )
            
            # Validate the external doctor API is reachable and key works
            if validate_doctor:
                self._validate_external_doctor(api_url, api_key)
        
        # Validate conversation_initiator
        if conversation_initiator not in ("patient", "doctor"):
            raise ValidationError(
                f"Invalid conversation_initiator: '{conversation_initiator}'. "
                "Must be 'patient' or 'doctor'."
            )
        
        # Validate max_turns (1-50 range, system cap is 250)
        if not isinstance(max_turns, int) or max_turns < 1 or max_turns > 50:
            raise ValidationError(
                f"Invalid max_turns: {max_turns}. "
                "Must be an integer between 1 and 50."
            )
        
        # Build pipeline config in v2.0 format
        config = {
            "description": description or "",
            "doctor": doctor,
            "patients": {
                "patient_ids": patient_ids or [],
            },
            "conversation": {
                "initiator": conversation_initiator,
                "max_turns": max_turns,
            },
            "judge": {
                "enabled": True,
                "dimensions": dimension_ids
            }
        }
        
        data = {
            "name": name,
            "config": config
        }
        
        response = self._request("POST", "/pipelines", data=data)
        return Pipeline.from_dict(response)
    
    def update(
        self,
        pipeline_id: str,
        name: Optional[str] = None,
        dimension_ids: Optional[list[str]] = None,
        doctor_api: Optional[DoctorApiConfig | dict] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Pipeline:
        """
        Update an existing pipeline.
        
        Args:
            pipeline_id: The pipeline ID to update
            name: New name (optional)
            dimension_ids: New dimension IDs (optional)
            doctor_api: New doctor API config (optional)
            description: New description (optional)
            is_active: Active status (optional)
            
        Returns:
            Updated Pipeline object
        """
        data = {}
        if name is not None:
            data["name"] = name
        if dimension_ids is not None:
            data["dimension_ids"] = dimension_ids
        if doctor_api is not None:
            if isinstance(doctor_api, dict):
                doctor_api = DoctorApiConfig.from_dict(doctor_api)
            data["doctor_api"] = doctor_api.to_dict()
        if description is not None:
            data["description"] = description
        if is_active is not None:
            data["is_active"] = is_active
        
        response = self._request("PATCH", f"/pipelines/{pipeline_id}", data=data)
        return Pipeline.from_dict(response)
    
    def delete(self, pipeline_id: str) -> None:
        """
        Delete a pipeline (soft delete - sets is_active=False).
        
        Args:
            pipeline_id: The pipeline ID to delete
        """
        self._request("DELETE", f"/pipelines/{pipeline_id}")


class SimulationsAPI(BaseAPI):
    """API client for running and managing simulations."""
    
    def list(
        self,
        pipeline_id: Optional[str] = None,
        status: Optional[SimulationStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Simulation]:
        """
        List simulations for the organization.
        
        Args:
            pipeline_id: Filter by pipeline ID
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of Simulation objects
        """
        params = {"limit": limit, "offset": offset}
        if pipeline_id:
            params["pipeline_id"] = pipeline_id
        if status:
            params["status"] = status.value
        
        response = self._request("GET", "/simulations", params=params)
        return [Simulation.from_dict(s) for s in response.get("simulations", [])]
    
    def get(self, simulation_id: str) -> Simulation:
        """
        Get a specific simulation by ID.
        
        Args:
            simulation_id: The simulation ID
            
        Returns:
            Simulation object
        """
        response = self._request("GET", f"/simulations/{simulation_id}")
        return Simulation.from_dict(response)
    
    def create(
        self,
        pipeline_name: str,
        num_episodes: Optional[int] = None,
        parallel_count: int = 1,
    ) -> Simulation:
        """
        Create and start a new simulation.
        
        Args:
            pipeline_name: Name of the pipeline to use for evaluation
            num_episodes: Number of episodes to run (if None, uses patients in pipeline)
            parallel_count: Number of parallel episodes (1-10)
            
        Returns:
            Created Simulation object (status will be PENDING or RUNNING)
        """
        data = {
            "pipeline_name": pipeline_name,
            "parallel_count": min(max(parallel_count, 1), 10),
        }
        if num_episodes is not None:
            data["num_episodes"] = num_episodes
        
        response = self._request("POST", "/simulations/start", data=data)
        return Simulation.from_dict(response)
    
    def get_episodes(
        self,
        simulation_id: str,
        include_dialogue: bool = False,
    ) -> list[dict]:
        """
        Get all episodes for a simulation.
        
        Use this to get detailed per-episode status while a simulation is running,
        or to review individual episode results after completion.
        
        Args:
            simulation_id: The simulation ID
            include_dialogue: Whether to include full dialogue history (default: False)
            
        Returns:
            List of episode dictionaries, each containing:
            - episode_id: Unique episode identifier
            - episode_number: Episode index (0-based)
            - status: 'pending', 'running', 'completed', 'failed'
            - patient_id: Patient identifier
            - patient_name: Patient name (if available)
            - dialogue_turns: Number of conversation turns
            - total_score: Final score (1-4 scale, if completed)
            - judge_scores: Per-dimension scores (if completed)
            - error: Error message (if failed)
            - dialogue_history: Full conversation (if include_dialogue=True)
            
        Example:
            >>> episodes = client.simulations.get_episodes(sim_id)
            >>> for ep in episodes:
            ...     print(f"Episode {ep['episode_number']}: {ep['status']}")
        """
        params = {}
        if include_dialogue:
            params["include_dialogue"] = "true"
        
        response = self._request("GET", f"/simulations/{simulation_id}/episodes", params=params)
        return response.get("episodes", [])
    
    def get_episode(
        self,
        simulation_id: str,
        episode_id: str,
    ) -> dict:
        """
        Get a single episode with full details including dialogue history.
        
        Args:
            simulation_id: The simulation ID
            episode_id: The episode ID
            
        Returns:
            Episode dictionary with full details
        """
        response = self._request("GET", f"/simulations/{simulation_id}/episodes/{episode_id}")
        return response
    
    def get_report(self, simulation_id: str) -> dict:
        """
        Get a complete simulation report with all details in one call.
        
        Returns everything needed for a final report: all episodes with full
        dialogue history, judge scores, and detailed feedback.
        
        **Use this for:** Final reports, detailed analysis, exporting results.
        **For progress polling:** Use `get()` instead (lightweight).
        
        Args:
            simulation_id: The simulation ID
            
        Returns:
            Complete report dictionary containing:
            - simulation_id, pipeline_name, status, timing info
            - summary: total_episodes, completed, failed, average_score, etc.
            - dimension_scores: average/min/max per evaluation dimension
            - episodes: list of all episodes with:
                - dialogue_history: full conversation
                - judge_scores: per-dimension scores (1-4 scale)
                - judge_feedback: detailed rationale from the judge
                - patient_id, patient_name
                - status, error (if failed)
        
        Example:
            >>> report = client.simulations.get_report(sim_id)
            >>> print(f"Average score: {report['summary']['average_score']}")
            >>> for dim, stats in report['dimension_scores'].items():
            ...     print(f"{dim}: {stats['average']:.2f}")
            >>> for ep in report['episodes']:
            ...     print(f"Episode {ep['episode_number']}: {ep['total_score']}")
            ...     for turn in ep['dialogue_history']:
            ...         print(f"  {turn['role']}: {turn['content'][:50]}...")
        """
        response = self._request("GET", f"/simulations/{simulation_id}/report")
        return response
    
    def wait_for_completion(
        self,
        simulation_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[Simulation], None]] = None,
    ) -> Simulation:
        """
        Wait for a simulation to complete with optional progress updates.
        
        Args:
            simulation_id: The simulation ID
            poll_interval: Seconds between status checks (default: 5.0)
            timeout: Maximum seconds to wait (None = no timeout)
            on_progress: Optional callback function called with Simulation object
                         on each poll. Use to display progress updates.
            
        Returns:
            Completed Simulation object
            
        Raises:
            TimeoutError: If timeout is reached
            SimulationError: If simulation fails
            
        Example:
            >>> def show_progress(sim):
            ...     pct = int(sim.progress * 100)
            ...     print(f"Progress: {sim.completed_episodes}/{sim.total_episodes} ({pct}%)")
            >>> 
            >>> result = client.simulations.wait_for_completion(
            ...     simulation.id,
            ...     on_progress=show_progress
            ... )
        """
        start_time = time.time()
        
        while True:
            simulation = self.get(simulation_id)
            
            # Call progress callback if provided
            if on_progress:
                try:
                    on_progress(simulation)
                except Exception:
                    pass  # Don't let callback errors break the wait loop
            
            if simulation.status == SimulationStatus.COMPLETED:
                return simulation
            elif simulation.status == SimulationStatus.FAILED:
                from .exceptions import SimulationError
                # Get failed episode count for better error message
                error_message = simulation.error_message or "Simulation failed"
                try:
                    episodes = self.get_episodes(simulation_id)
                    failed = sum(1 for e in episodes if e.get("status") == "failed")
                    total = len(episodes)
                    if failed > 0:
                        error_message = f"{failed}/{total} episodes failed"
                except Exception:
                    pass  # Keep the original error_message if we can't get episodes
                raise SimulationError(simulation_id, error_message)
            elif simulation.status == SimulationStatus.CANCELLED:
                from .exceptions import SimulationError
                raise SimulationError(simulation_id, "Simulation was cancelled")
            
            if timeout and (time.time() - start_time) >= timeout:
                raise TimeoutError(f"Simulation {simulation_id} did not complete within {timeout}s")
            
            time.sleep(poll_interval)
    
    def cancel(self, simulation_id: str) -> Simulation:
        """
        Cancel a running simulation.
        
        Args:
            simulation_id: The simulation ID
            
        Returns:
            Updated Simulation object
        """
        response = self._request("POST", f"/simulations/{simulation_id}/cancel")
        return Simulation.from_dict(response)

    # =========================================================================
    # Client-Driven Simulation Methods
    # =========================================================================
    # These methods are for client-driven simulations where the customer
    # pushes doctor responses instead of the orchestrator calling their API.
    # =========================================================================
    
    def get_episode(self, simulation_id: str, episode_id: str) -> dict:
        """
        Get a single episode with full details.
        
        Use this to check episode state in client-driven simulations.
        The dialogue_history shows all messages exchanged so far.
        
        Args:
            simulation_id: The simulation ID
            episode_id: The episode ID
            
        Returns:
            Episode dictionary containing:
            - episode_id, simulation_id, episode_number
            - patient_id, patient_name
            - status: "pending", "awaiting_doctor", "conversation", "judging", "completed", "failed"
            - dialogue_history: list of {role: "patient"|"doctor", content: str}
            - judge_scores, total_score (if judged)
            - error (if failed)
            
        Example:
            ```python
            ep = client.simulations.get_episode(sim_id, episode_id)
            if ep["status"] == "awaiting_doctor":
                last_msg = ep["dialogue_history"][-1]["content"]
                print(f"Patient said: {last_msg}")
            ```
        """
        response = self._request("GET", f"/simulations/{simulation_id}/episodes/{episode_id}")
        return response
    
    def submit_response(
        self,
        simulation_id: str,
        episode_id: str,
        message: str,
    ) -> dict:
        """
        Submit a doctor response for a client-driven simulation.
        
        In client-driven mode, the orchestrator does NOT call your doctor API.
        Instead, YOU:
        1. Poll episodes to check for pending patient messages
        2. Call your own doctor (locally, behind VPN, etc.)
        3. Submit the response using this method
        
        The orchestrator will:
        1. Store the doctor message
        2. Call the Patient API with the updated conversation
        3. Store the patient's response
        4. If conversation is complete, trigger the judge
        
        Args:
            simulation_id: The simulation ID
            episode_id: The episode ID to respond to
            message: The doctor's response message
            
        Returns:
            Updated episode dictionary with new dialogue_history
            
        Raises:
            ValidationError: If episode is not awaiting a doctor response
            NotFoundError: If simulation or episode not found
            
        Example:
            ```python
            # Get episode state
            ep = client.simulations.get_episode(sim_id, episode_id)
            
            # Check if waiting for doctor
            if ep["status"] == "awaiting_doctor":
                # Get patient's message
                patient_msg = ep["dialogue_history"][-1]["content"]
                
                # Call YOUR doctor API (behind VPN, locally, etc.)
                doctor_response = my_doctor_api.chat(patient_msg)
                
                # Submit to EARL
                updated_ep = client.simulations.submit_response(
                    sim_id, 
                    episode_id, 
                    doctor_response
                )
                print(f"Dialogue now has {len(updated_ep['dialogue_history'])} turns")
            ```
        """
        response = self._request(
            "POST",
            f"/simulations/{simulation_id}/episodes/{episode_id}/respond",
            data={"message": message}
        )
        return response
    
    def get_pending_episodes(self, simulation_id: str) -> list[dict]:
        """
        Get all episodes awaiting a doctor response.
        
        Convenience method for client-driven simulations to find
        which episodes need attention.
        
        Args:
            simulation_id: The simulation ID
            
        Returns:
            List of episode dictionaries with status="awaiting_doctor"
            
        Example:
            ```python
            # Process all pending episodes
            pending = client.simulations.get_pending_episodes(sim_id)
            for ep in pending:
                patient_msg = ep["dialogue_history"][-1]["content"]
                response = my_doctor(patient_msg)
                client.simulations.submit_response(sim_id, ep["episode_id"], response)
            ```
        """
        episodes = self.get_episodes(simulation_id)
        return [ep for ep in episodes if ep.get("status") == "awaiting_doctor"]


class RateLimitsAPI(BaseAPI):
    """API client for checking rate limits."""
    
    def get(self) -> dict:
        """
        Get current rate limits for your organization.
        
        Returns the rate limits that apply to your API calls. Limits are enforced
        per organization and reset every minute.
        
        Returns:
            Dictionary containing:
            - organization_id: Your organization ID
            - limits: Your organization's configured limits
              - per_minute: Requests allowed per minute
              - per_hour: Requests allowed per hour  
              - per_day: Requests allowed per day
            - category_limits: Limits by endpoint category
            - effective_limits: Actual limits applied (min of category and org limits)
            - headers_info: Explanation of X-RateLimit-* headers
            
        Example:
            ```python
            limits = client.rate_limits.get()
            print(f"Per minute: {limits['limits']['per_minute']}")
            print(f"Simulations: {limits['effective_limits']['simulations']}")
            ```
        """
        return self._request("GET", "/rate-limits")
    
    def get_effective_limit(self, category: str = "default") -> int:
        """
        Get the effective rate limit for a specific category.
        
        Args:
            category: One of "evaluations", "pipelines", "simulations", or "default"
            
        Returns:
            Maximum requests per minute for this category
        """
        limits = self.get()
        return limits.get("effective_limits", {}).get(category, 60)
