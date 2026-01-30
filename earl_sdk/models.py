"""Data models for Earl SDK."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict


class SimulationStatus(str, Enum):
    """Status of a simulation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DoctorApiConfig:
    """
    Configuration for doctor API endpoint.
    
    Supports three modes:
    - Internal: Uses EARL's built-in doctor agent (type="internal")
    - External: Uses customer's own doctor API (type="external")
    - Client-driven: Customer pushes doctor responses via SDK (type="client_driven")
    
    The client_driven mode is useful when:
    - Doctor API is behind a VPN/firewall
    - Customer wants full control over conversation flow
    - Integration with complex internal systems
    """
    type: str = "internal"  # "internal", "external", or "client_driven"
    api_url: Optional[str] = None  # Required for external
    api_key: Optional[str] = None  # Optional API key for external
    prompt: Optional[str] = None  # Optional system prompt override
    
    # Legacy fields for backward compatibility
    url: Optional[str] = None  # Alias for api_url
    auth_header: Optional[str] = None  # Legacy auth header
    auth_type: str = "bearer"  # bearer, api_key, basic
    
    # Configuration for external doctor API calls (used by orchestrator, not SDK directly)
    timeout_seconds: int = 30  # Timeout for doctor API calls
    retry_count: int = 3  # Number of retries for transient errors
    
    def __post_init__(self):
        # Handle legacy 'url' field
        if self.url and not self.api_url:
            self.api_url = self.url
            self.type = "external"
    
    @classmethod
    def internal(cls, prompt: Optional[str] = None) -> "DoctorApiConfig":
        """Create an internal doctor configuration."""
        return cls(type="internal", prompt=prompt)
    
    @classmethod
    def external(
        cls, 
        api_url: str, 
        api_key: Optional[str] = None, 
        auth_type: str = "bearer",
        prompt: Optional[str] = None
    ) -> "DoctorApiConfig":
        """
        Create an external doctor configuration.
        
        Args:
            api_url: URL of the external doctor API
            api_key: API key for authentication (optional)
            auth_type: How to send the API key in requests:
                - "bearer": Authorization: Bearer <token> (OpenAI, Modal, etc.) - DEFAULT
                - "api_key": X-API-Key: <token> (custom APIs)
            prompt: System prompt (not used for external doctors - they manage their own)
        """
        return cls(type="external", api_url=api_url, api_key=api_key, auth_type=auth_type, prompt=prompt)
    
    @classmethod
    def client_driven(cls) -> "DoctorApiConfig":
        """
        Create a client-driven doctor configuration.
        
        In client-driven mode, the orchestrator does NOT call any doctor API.
        Instead, the customer:
        1. Polls for pending patient messages
        2. Calls their own doctor (locally, behind VPN, etc.)
        3. Submits the doctor response via SDK
        
        Example:
            ```python
            # Create client-driven pipeline
            pipeline = client.pipelines.create(
                name="vpn-doctor-eval",
                dimension_ids=["factuality", "empathy"],
                doctor_config=DoctorApiConfig.client_driven(),
            )
            
            # Start simulation
            sim = client.simulations.create(pipeline_name=pipeline.name, num_episodes=3)
            
            # Process episodes - poll until all complete
            while sim.status.value not in ["completed", "failed"]:
                sim = client.simulations.get(sim.id)
                episodes = client.simulations.get_episodes(sim.id)
                
                for ep in episodes:
                    if ep["status"] == "awaiting_doctor":
                        full_ep = client.simulations.get_episode(sim.id, ep["episode_id"])
                        dialogue = full_ep.get("dialogue_history", [])
                        
                        # Call YOUR doctor API (behind VPN)
                        response = my_doctor(dialogue)
                        client.simulations.submit_response(sim.id, ep["episode_id"], response)
            ```
        """
        return cls(type="client_driven")
    
    def to_dict(self) -> dict:
        result = {"type": self.type}
        if self.api_url:
            result["api_url"] = self.api_url
        if self.api_key:
            result["api_key"] = self.api_key
        if self.prompt:
            result["prompt"] = self.prompt
        # Always include auth_type for external doctors (default: bearer)
        if self.type == "external":
            result["auth_type"] = self.auth_type
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "DoctorApiConfig":
        if data is None:
            return cls(type="internal")
        
        # Determine type: explicit > inferred from URL > default internal
        doc_type = data.get("type")
        if not doc_type:
            if data.get("url") or data.get("api_url"):
                doc_type = "external"
            else:
                doc_type = "internal"
        
        return cls(
            type=doc_type,
            api_url=data.get("api_url") or data.get("url"),
            api_key=data.get("api_key"),
            prompt=data.get("prompt"),
            url=data.get("url"),
            auth_header=data.get("auth_header"),
            auth_type=data.get("auth_type", "bearer"),
            timeout_seconds=data.get("timeout_seconds", 30),
            retry_count=data.get("retry_count", 3),
        )
    
    @property
    def is_client_driven(self) -> bool:
        """Check if this is a client-driven configuration."""
        return self.type == "client_driven"


class ConversationInitiator(str, Enum):
    """Who initiates the conversation."""
    PATIENT = "patient"  # Patient sends first message (e.g., "I have a headache...")
    DOCTOR = "doctor"    # Doctor sends first message (e.g., "Hello, what brings you in?")


@dataclass
class ConversationConfig:
    """
    Configuration for how conversations are conducted.
    
    Attributes:
        initiator: Who starts the conversation - "patient" or "doctor"
            - "patient": Patient sends first message describing symptoms/concerns.
                         This is the typical telemedicine scenario.
            - "doctor": Doctor sends first message (greeting/opening).
                         This is useful for proactive care or follow-up calls.
        max_turns: Maximum number of conversation turns (1-50, default 10).
            The conversation ends after this many turns. The patient will 
            indicate they need to leave as the limit approaches.
            Note: System has a hard cap of 250 turns.
    """
    initiator: str = "patient"  # "patient" or "doctor"
    max_turns: int = 10  # Customer-configurable (1-50), system cap is 250
    
    def to_dict(self) -> dict:
        return {
            "initiator": self.initiator,
            "max_turns": self.max_turns,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationConfig":
        if data is None:
            return cls()
        return cls(
            initiator=data.get("initiator", "patient"),
            max_turns=data.get("max_turns", 10),
        )
    
    @classmethod
    def patient_initiated(cls, max_turns: int = 10) -> "ConversationConfig":
        """Create a patient-initiated conversation config."""
        return cls(initiator="patient", max_turns=max_turns)
    
    @classmethod
    def doctor_initiated(cls, max_turns: int = 10) -> "ConversationConfig":
        """Create a doctor-initiated conversation config."""
        return cls(initiator="doctor", max_turns=max_turns)


@dataclass
class Dimension:
    """
    An evaluation dimension for judging doctor responses.
    
    Dimensions define what aspects of a doctor's response are evaluated,
    such as accuracy, empathy, safety, etc.
    """
    id: str
    name: str
    description: str
    category: str
    weight: float = 1.0
    is_custom: bool = False
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "Dimension":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=data.get("category", "general"),
            weight=data.get("weight", 1.0),
            is_custom=data.get("is_custom", False),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )


@dataclass
class Patient:
    """
    A simulated patient for evaluation scenarios.
    
    Patients have conditions, symptoms, and expected behaviors
    that the doctor AI should handle appropriately.
    
    Patients have rich emotional and cognitive state, session-based
    conversation management, termination signals, and internal thoughts.
    
    Attributes:
        scenario: Current scenario/situation description
        behaviors: Behavioral dimensions (looping, validation, etc.)
        occupation: Patient's occupation
        personality: Personality traits
        reactive_traits: How patient reacts under stress
        condition: Medical condition (anxiety, asthma, etc.)
        task: Task type (focused-clinical-encounter, medication-reconciliation, etc.)
    
    Note:
        Not all fields are populated for every patient. Check for None/empty
        before using optional fields.
    """
    id: str
    name: str
    description: str = ""
    simulation_id: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    tags: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    synthea_id: Optional[str] = None
    # Optional detailed fields (may not be populated)
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: Optional[str] = None
    medical_history: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    # Generative patient fields
    scenario: Optional[str] = None  # Current scenario/situation
    behaviors: Optional[Dict[str, Any]] = None  # Behavioral dimensions
    occupation: Optional[str] = None
    personality: Optional[str] = None
    reactive_traits: Optional[str] = None
    condition: Optional[str] = None  # Medical condition (anxiety, asthma, etc.)
    task: Optional[str] = None  # Task type (focused-clinical-encounter, etc.)
    task_display: Optional[str] = None  # Human-readable task name
    
    @classmethod
    def from_dict(cls, data: dict) -> "Patient":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            description=data.get("description", data.get("scenario", "")),
            simulation_id=data.get("simulation_id"),
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", []),
            conditions=data.get("conditions", []),
            synthea_id=data.get("synthea_id"),
            age=data.get("age"),
            gender=data.get("gender"),
            chief_complaint=data.get("chief_complaint"),
            medical_history=data.get("medical_history", []),
            current_medications=data.get("current_medications", []),
            allergies=data.get("allergies", []),
            scenario=data.get("scenario"),
            behaviors=data.get("behaviors"),
            occupation=data.get("occupation"),
            personality=data.get("personality"),
            reactive_traits=data.get("reactive_traits"),
            condition=data.get("condition"),
            task=data.get("task"),
            task_display=data.get("task_display"),
        )


@dataclass
class Pipeline:
    """
    An evaluation pipeline that defines how simulations are run.
    
    Pipelines specify which dimensions to evaluate, the doctor API to test,
    conversation settings, and configuration for the simulation.
    """
    name: str  # Pipeline name (used as ID in API)
    description: Optional[str] = None
    is_default: bool = False
    is_active: bool = True
    has_auth_key: bool = False
    organization_id: Optional[str] = None
    dimension_ids: list[str] = field(default_factory=list)
    patient_ids: list[str] = field(default_factory=list)  # Patient IDs in this pipeline
    doctor_api: Optional[DoctorApiConfig] = None
    conversation: Optional[ConversationConfig] = None  # Who initiates: patient or doctor
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    
    @property
    def id(self) -> str:
        """Pipeline name serves as the ID."""
        return self.name
    
    @property
    def conversation_initiator(self) -> str:
        """Get who initiates the conversation (patient or doctor)."""
        if self.conversation:
            return self.conversation.initiator
        return "patient"  # Default
    
    @property
    def max_turns(self) -> int:
        """Get the maximum number of conversation turns."""
        if self.conversation:
            return self.conversation.max_turns
        return 10  # Default
    
    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        # Parse doctor config from either doctor_api or config.doctor
        doctor_api = None
        if data.get("doctor_api"):
            doctor_api = DoctorApiConfig.from_dict(data["doctor_api"])
        elif data.get("config", {}).get("doctor"):
            doctor_api = DoctorApiConfig.from_dict(data["config"]["doctor"])
        
        # Parse conversation config from config.conversation
        conversation = None
        if data.get("conversation"):
            conversation = ConversationConfig.from_dict(data["conversation"])
        elif data.get("config", {}).get("conversation"):
            conversation = ConversationConfig.from_dict(data["config"]["conversation"])
        
        # Parse patient_ids from either patient_ids or config.patients.patient_ids
        patient_ids = data.get("patient_ids", [])
        if not patient_ids:
            config = data.get("config", {})
            patients = config.get("patients", {})
            patient_ids = patients.get("patient_ids", [])
        
        # Parse dimension_ids from either dimension_ids or config.judge.dimensions
        dimension_ids = data.get("dimension_ids", [])
        if not dimension_ids:
            config = data.get("config", {})
            judge = config.get("judge", {})
            dimension_ids = judge.get("dimensions", [])
        
        # Parse description from either description or config.description
        description = data.get("description")
        if not description:
            description = data.get("config", {}).get("description")
        
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        
        return cls(
            name=data.get("name") or data.get("pipeline_name", ""),
            description=description,
            is_default=data.get("is_default", False) or data.get("config", {}).get("is_default", False),
            is_active=data.get("is_active", True),
            has_auth_key=data.get("has_auth_key", False),
            organization_id=data.get("organization_id"),
            dimension_ids=dimension_ids,
            patient_ids=patient_ids,
            doctor_api=doctor_api,
            conversation=conversation,
            created_at=created_at,
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")) if data.get("updated_at") else None,
            created_by=data.get("created_by"),
        )


@dataclass
class DimensionScore:
    """
    Score for a single dimension in a simulation result.
    
    Attributes:
        dimension_id: Unique identifier for the dimension
        dimension_name: Human-readable dimension name
        score: Score on 1-4 scale (1=poor, 2=fair, 3=good, 4=excellent)
        reasoning: Judge's explanation for the score
        details: Additional scoring details
    """
    dimension_id: str
    dimension_name: str
    score: float  # 1.0 to 4.0 scale
    reasoning: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Simulation:
    """
    A simulation run that evaluates a doctor API against patients.
    
    Simulations are created from a pipeline and run against a set of patients.
    """
    id: str
    pipeline_name: str
    organization_id: str
    status: SimulationStatus
    simulation_type: str = "conversational"
    total_episodes: int = 0
    completed_episodes: int = 0
    current_episode: int = 0
    user_id: Optional[str] = None
    error_message: Optional[str] = None
    summary: Optional[dict] = None
    config_snapshot: Optional[dict] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    
    @property
    def progress(self) -> float:
        """Calculate progress as a ratio."""
        if self.total_episodes == 0:
            return 0.0
        return self.completed_episodes / self.total_episodes
    
    @classmethod
    def _parse_datetime(cls, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
    
    @classmethod
    def from_dict(cls, data: dict) -> "Simulation":
        # Handle different status formats
        status_value = data.get("status", "pending")
        try:
            status = SimulationStatus(status_value)
        except ValueError:
            status = SimulationStatus.PENDING
        
        return cls(
            id=data.get("simulation_id") or data.get("id", ""),
            pipeline_name=data.get("pipeline_name") or data.get("pipeline_id", ""),
            organization_id=data.get("organization_id", ""),
            status=status,
            simulation_type=data.get("simulation_type", "conversational"),
            total_episodes=data.get("total_episodes", 0),
            completed_episodes=data.get("completed_episodes", 0),
            current_episode=data.get("current_episode", 0),
            user_id=data.get("user_id"),
            error_message=data.get("error"),
            summary=data.get("summary"),
            config_snapshot=data.get("config_snapshot"),
            started_at=cls._parse_datetime(data.get("started_at")),
            updated_at=cls._parse_datetime(data.get("updated_at")),
            finished_at=cls._parse_datetime(data.get("finished_at")),
        )

