"""API clients for Earl SDK resources.

Transport is handled by :mod:`earl_sdk._http`, which provides a shared
``httpx.Client`` with HTTP/2 (when ``h2`` is installed), connection pooling,
gzip compression, structured timeouts, and automatic retries on idempotent
verbs. This module just composes endpoint paths and decodes response shapes.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from abc import ABC
from collections.abc import Callable
from typing import Any

from . import _http
from .auth import Auth0Client
from .exceptions import AuthenticationError, ValidationError
from .models import Dimension, DoctorApiConfig, Patient, Pipeline, Simulation, SimulationStatus

logger = logging.getLogger("earl_sdk.api")


class BaseAPI(ABC):
    """Base class for API clients."""

    DEFAULT_REQUEST_TIMEOUT = 60

    def __init__(self, auth: Auth0Client, base_url: str, request_timeout: int | None = None):
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self._request_timeout = (
            request_timeout if request_timeout is not None else self.DEFAULT_REQUEST_TIMEOUT
        )

    def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Make an authenticated API request via the shared httpx client."""
        # URL-encode each path segment so IDs with spaces / slashes work.
        encoded_path = "/".join(
            urllib.parse.quote(segment, safe="") if segment else "" for segment in path.split("/")
        )
        url = f"{self.base_url}{encoded_path}"

        headers = self.auth.get_headers()
        headers["Content-Type"] = "application/json"

        try:
            body, _response = _http.request_json(
                method,
                url,
                headers=headers,
                params=params,
                json_body=data,
                timeout=self._request_timeout,
            )
        except AuthenticationError:
            # Invalidate the cached access token so the next call forces a
            # fresh auth exchange (or refresh-token swap for device profiles).
            self.auth.invalidate_token()
            raise

        return body if isinstance(body, dict) else {}


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
        difficulty: str | None = None,
        tags: list[str] | None = None,
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
        return [Patient.from_dict(p) for p in response.get("cases", response.get("patients", []))]

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
        api_key: str | None = None,
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
        import httpx

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Earl-SDK-Validator/1.0",
        }
        if api_key:
            # Support both header formats for broader API compatibility.
            headers["X-API-Key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint_url = api_url.rstrip("/")
        test_payload: dict[str, Any] = {
            "model": "default",
            "messages": [{"role": "user", "content": "Hello, I am testing the connection."}],
            "max_tokens": 50,
        }

        # Warming strategy for cold-start APIs (Modal, Lambda, etc.)
        # 1st attempt: quick check. 2nd: allow cold start. 3rd: retry after warm.
        attempts: list[tuple[float, str]] = [
            (timeout, "initial check"),
            (60.0, "warming (cold start)"),
            (30.0, "retry after warm"),
        ]
        last_error: str | None = None

        with httpx.Client(timeout=timeout, follow_redirects=False) as validator:
            for attempt_timeout, attempt_desc in attempts:
                try:
                    r = validator.post(
                        endpoint_url,
                        json=test_payload,
                        headers=headers,
                        timeout=attempt_timeout,
                    )
                except httpx.TimeoutException:
                    last_error = f"Request timed out after {attempt_timeout}s ({attempt_desc})"
                    continue
                except httpx.TransportError as exc:
                    last_error = f"Cannot connect: {exc}"
                    continue

                if 200 <= r.status_code < 300:
                    return  # Endpoint responded successfully.
                if r.status_code == 401:
                    raise ValidationError(
                        "External doctor API authentication failed (401 Unauthorized).\n"
                        f"URL: {endpoint_url}\n"
                        f"API key provided: {'Yes' if api_key else 'No'}\n"
                        "Please verify your API key is correct."
                    )
                if r.status_code == 403:
                    raise ValidationError(
                        "External doctor API access forbidden (403 Forbidden).\n"
                        f"URL: {endpoint_url}\n"
                        "Please verify your API key has the correct permissions."
                    )
                if r.status_code == 404:
                    raise ValidationError(
                        "External doctor API endpoint not found (404).\n"
                        f"URL: {endpoint_url}\n"
                        "The orchestrator will POST to this exact URL.\n"
                        "Please verify the URL is correct and accepts POST requests."
                    )
                if r.status_code >= 500:
                    last_error = f"Server error ({r.status_code})"
                    continue
                # Any other 4xx means the API is reachable and speaking HTTP;
                # payload format differences don't block pipeline creation.
                return

        raise ValidationError(
            "Cannot reach external doctor API.\n"
            f"URL: {endpoint_url}\n"
            f"Error: {last_error}\n\n"
            f"Tried {len(attempts)} times with increasing timeouts (up to 60s for cold start).\n\n"
            "The orchestrator will POST to this URL during simulations.\n"
            "Please verify:\n"
            "  1. The URL is correct and accessible\n"
            "  2. The service is running and not paused\n"
            "  3. Any firewalls or VPNs allow the connection\n"
            "  4. For serverless APIs (Modal, Lambda): the service may need to be warmed up"
        )

    def validate_doctor_api(
        self,
        api_url: str,
        api_key: str | None = None,
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
        verifier_ids: list[str] | None = None,
        case_id: str | None = None,
        doctor_config: DoctorApiConfig | dict | None = None,
        patient_ids: list[str] | None = None,
        description: str | None = None,
        use_internal_doctor: bool = True,
        validate_doctor: bool = True,
        conversation_initiator: str = "patient",
        max_turns: int = 10,
        verifiers: str = "lumos",
        # Deprecated aliases
        dimension_ids: list[str] | None = None,
        judge_type: str | None = None,
    ) -> Pipeline:
        """
        Create a new evaluation pipeline.

        Args:
            name: Pipeline name (must be unique within your organization).
            verifier_ids: List of verifier IDs to evaluate. Use builtin paths like
                ``"scoring-dimensions/clinical-correctness"`` or
                ``"hard-gates/fabricated-ehr-data"``.
                See available verifiers at https://docs.earl.thelumos.ai/verifiers.
            doctor_config: Configuration for the doctor API.
                Use ``DoctorApiConfig.internal()`` for Earl's built-in doctor,
                ``DoctorApiConfig.external(api_url=..., api_key=...)`` for your API,
                or ``DoctorApiConfig.client_driven()`` for VPN/firewall scenarios.
                If None, uses Earl's internal doctor.
            patient_ids: Optional list of patient IDs to include.
                Use ``client.patients.list()`` to see available patients.
            description: Optional description for the pipeline.
            use_internal_doctor: If True and doctor_config is None, use internal doctor.
            validate_doctor: If True, validates external doctor API before creating pipeline.
            conversation_initiator: Who sends the first message — ``"patient"`` or ``"doctor"``.
            max_turns: Maximum conversation turns (1–50, default 10).
            verifiers: Verifier backend — ``"lumos"`` (next-gen, default) or ``"legacy"``.

        Returns:
            Created Pipeline object.

        Examples:
            ```python
            # Lumos verifiers (default) — hard gates + scoring dimensions
            pipeline = client.pipelines.create(
                name="my-eval",
                verifier_ids=[
                    "hard-gates/fabricated-ehr-data",
                    "scoring-dimensions/clinical-correctness",
                    "scoring-dimensions/communication--empathy",
                ],
                patient_ids=[p.id for p in patients],
            )

            # With external doctor API
            pipeline = client.pipelines.create(
                name="my-eval",
                verifier_ids=["scoring-dimensions/clinical-correctness"],
                doctor_config=DoctorApiConfig.external(
                    api_url="https://my-doctor.com/chat",
                    api_key="my-key",
                ),
            )

            # Doctor-initiated, 30-turn conversations
            pipeline = client.pipelines.create(
                name="thorough-eval",
                verifier_ids=["scoring-dimensions/clinical-correctness"],
                conversation_initiator="doctor",
                max_turns=30,
            )
            ```
        """
        # Handle deprecated aliases
        if dimension_ids is not None and verifier_ids is None:
            verifier_ids = dimension_ids
        if judge_type is not None and verifiers == "lumos":
            verifiers = judge_type
        if verifier_ids is None:
            verifier_ids = []
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

        # Validate doctor type
        doctor_type = doctor.get("type", "internal")

        if doctor_type not in ("internal", "external", "client_driven"):
            raise ValidationError(
                f"Invalid doctor type: '{doctor_type}'. "
                f"Must be 'internal', 'external', or 'client_driven'.\n"
                f"Use DoctorApiConfig.internal(), DoctorApiConfig.external(...), "
                f"or DoctorApiConfig.client_driven()."
            )

        # client_driven is a sub-mode of external doctor (for VPN/firewall scenarios).
        # It does NOT work with internal doctor - the orchestrator would need to
        # call EARL's built-in doctor AND have the customer push responses, which
        # makes no sense. client_driven means the customer has their own external
        # doctor API that is behind a VPN/firewall and cannot be reached directly.

        # Validate external doctor API before creating pipeline
        if doctor_type == "external":
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
                f"Invalid max_turns: {max_turns}. " "Must be an integer between 1 and 50."
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
                "dimensions": verifier_ids,
                "type": verifiers,
                **({"case_id": case_id} if case_id else {}),
            },
        }

        data = {"name": name, "config": config}

        response = self._request("POST", "/pipelines", data=data)
        return Pipeline.from_dict(response)

    def update(
        self,
        pipeline_name: str,
        verifier_ids: list[str] | None = None,
        doctor_config: DoctorApiConfig | dict | None = None,
        patient_ids: list[str] | None = None,
        description: str | None = None,
        conversation_initiator: str | None = None,
        max_turns: int | None = None,
        verifiers: str | None = None,
        dimension_ids: list[str] | None = None,  # deprecated alias
    ) -> Pipeline:
        """
        Update an existing pipeline. Only provided fields are updated.

        Args:
            pipeline_name: The pipeline name to update.
            verifier_ids: New verifier IDs (optional).
            doctor_config: New doctor API config (optional).
            patient_ids: New patient IDs (optional).
            description: New description (optional).
            conversation_initiator: ``"patient"`` or ``"doctor"`` (optional).
            max_turns: Maximum conversation turns 1–50 (optional).
            verifiers: Verifier backend — ``"lumos"`` or ``"legacy"`` (optional).
        """
        # Handle deprecated alias
        if dimension_ids is not None and verifier_ids is None:
            verifier_ids = dimension_ids

        config: dict = {}

        if description is not None:
            config["description"] = description

        if doctor_config is not None:
            if isinstance(doctor_config, dict):
                config["doctor"] = doctor_config
            else:
                config["doctor"] = doctor_config.to_dict()

        if patient_ids is not None:
            config["patients"] = {"patient_ids": patient_ids}

        if verifier_ids is not None:
            judge_cfg: dict = {"enabled": True, "dimensions": verifier_ids}
            if verifiers is not None:
                judge_cfg["type"] = verifiers
            config["judge"] = judge_cfg

        if conversation_initiator is not None or max_turns is not None:
            conv: dict = {}
            if conversation_initiator is not None:
                conv["initiator"] = conversation_initiator
            if max_turns is not None:
                conv["max_turns"] = max_turns
            config["conversation"] = conv

        response = self._request("PUT", f"/pipelines/{pipeline_name}", data={"config": config})
        return Pipeline.from_dict(response)

    def delete(self, pipeline_name: str) -> None:
        """
        Delete a pipeline.

        Args:
            pipeline_name: The pipeline name to delete
        """
        self._request("DELETE", f"/pipelines/{pipeline_name}")


class CasesAPI(BaseAPI):
    """API client for listing pre-defined evaluation cases."""

    def list(self) -> list[dict]:
        """List all available evaluation cases.

        Returns:
            List of case summaries with case_id, name, description,
            patient_id, verifier_count, etc.
        """
        response = self._request("GET", "/cases")
        return response.get("cases", [])

    def get(self, case_id: str) -> dict:
        """Get detailed case definition including verifiers and default judges.

        Args:
            case_id: The case identifier, e.g. ``"carla-hypertension-yasmin"``.
        """
        return self._request("GET", f"/cases/{case_id}")

    def verifiers(self, case_id: str) -> dict:
        """List the case-specific verifiers defined upstream in the Lumos case-service.

        Args:
            case_id: The case identifier.

        Returns:
            ``{"case_id": str, "has_verifiers": bool, "verifiers": [ {name, category, points, activating_condition}, ... ]}``
        """
        return self._request("GET", f"/lumos-catalog/cases/{case_id}/verifiers")


class VerifiersAPI(BaseAPI):
    """API for the platform-wide Lumos verifier catalog (generic, not case-local)."""

    def list(self) -> dict:
        """
        List all generic hard gates and scoring dimensions.

        Calls ``GET /additional-verifiers`` (Lumos case-service contract). Response shape
        may vary; use :func:`earl_sdk.verifiers_catalog.parse_verifiers_list_payload` to normalize.

        Returns:
            Parsed JSON object from the API.
        """
        return self._request("GET", "/additional-verifiers")


class SimulationsAPI(BaseAPI):
    """API client for running and managing simulations."""

    def list(
        self,
        status: SimulationStatus | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> list[Simulation]:
        """
        List simulations for the organization.

        Args:
            status: Filter by status (running, completed, stopped, failed)
            limit: Maximum number of results (default 50)
            skip: Number of results to skip for pagination

        Returns:
            List of Simulation objects
        """
        params: dict[str, Any] = {"limit": limit, "skip": skip}
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
        num_episodes: int | None = None,
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
        timeout: float | None = None,
        on_progress: Callable[[Simulation], None] | None = None,
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
            elif simulation.status in (SimulationStatus.STOPPED, SimulationStatus.CANCELLED):
                from .exceptions import SimulationError

                raise SimulationError(simulation_id, "Simulation was stopped")

            if timeout and (time.time() - start_time) >= timeout:
                raise TimeoutError(f"Simulation {simulation_id} did not complete within {timeout}s")

            time.sleep(poll_interval)

    def stop(self, simulation_id: str) -> Simulation:
        """
        Stop a running simulation.

        Args:
            simulation_id: The simulation ID

        Returns:
            Updated Simulation object (re-fetched after stop)
        """
        self._request("POST", f"/simulations/{simulation_id}/stop")
        # The stop endpoint returns {success, message} — re-fetch full simulation
        return self.get(simulation_id)

    def cancel(self, simulation_id: str) -> Simulation:
        """
        Cancel a running simulation.

        .. deprecated:: Use :meth:`stop` instead. This is an alias for backward compatibility.

        Args:
            simulation_id: The simulation ID

        Returns:
            Updated Simulation object
        """
        return self.stop(simulation_id)

    # =========================================================================
    # Client-Driven Simulation Methods
    # =========================================================================
    # These methods are for client-driven simulations where the customer
    # pushes doctor responses instead of the orchestrator calling their API.
    # =========================================================================

    # get_episode() is defined above in the standard Simulations section;
    # the client-driven section reuses it — no separate definition needed.

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
            data={"message": message},
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


class AuthAPI(BaseAPI):
    """Auth-related endpoints on the orchestrator.

    These are *user-identity* introspection calls — they never mutate anything
    and are separate from Auth0 token exchange (which lives in
    :mod:`earl_sdk.device_flow` + :mod:`earl_sdk.auth`).
    """

    def my_orgs(self) -> list[dict]:
        """List the organizations the current user is a member of.

        Backed by ``GET /api/v1/auth/my-orgs`` on the orchestrator, which:

        - Accepts a valid Auth0 *user* token even if the token has no
          ``org_id`` claim (this is the single endpoint scoped for that
          purpose — it's read-only and returns nothing org-scoped).
        - Rejects M2M tokens (client-credentials grant).
        - Proxies Auth0 Management API:
          ``GET /api/v2/users/{sub}/organizations``.

        Returns:
            List of ``{"id", "name", "display_name"}`` dicts, possibly empty if
            the user isn't a member of any org on this tenant. The list is
            stable-ordered by ``display_name`` (then ``id``) so a CLI picker
            shows the same choice number across invocations.

        Raises:
            AuthenticationError: 401 — token invalid or expired.
            AuthorizationError:  403 — M2M token, or org-discovery explicitly
                                 disabled on this deployment.

        Example:
            ```python
            orgs = client.auth.my_orgs()
            for o in orgs:
                print(o["id"], o["display_name"] or o["name"])
            ```
        """
        response = self._request("GET", "/auth/my-orgs")
        # Orchestrator may return either a bare list or an envelope
        # ``{"organizations": [...]}`` — handle both so we can evolve the
        # server shape without breaking old clients.
        if isinstance(response, dict):
            items = response.get("organizations") or response.get("items") or []
        else:
            items = response
        return list(items) if isinstance(items, list) else []
