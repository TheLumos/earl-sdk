"""Exception classes for Earl SDK."""
from __future__ import annotations

from typing import Optional


class EarlError(Exception):
    """Base exception for all Earl SDK errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(EarlError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[dict] = None):
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(EarlError):
    """Raised when the user doesn't have permission."""
    
    def __init__(self, message: str = "Access denied", details: Optional[dict] = None):
        super().__init__(message, status_code=403, details=details)


class NotFoundError(EarlError):
    """Raised when a resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ValidationError(EarlError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, status_code=400, details=details)
        self.field = field


class RateLimitError(EarlError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, status_code=429, details={"retry_after": retry_after})
        self.retry_after = retry_after


class ServerError(EarlError):
    """Raised when the server returns an error."""
    
    def __init__(self, message: str = "Internal server error", details: Optional[dict] = None):
        super().__init__(message, status_code=500, details=details)


class SimulationError(EarlError):
    """Raised when a simulation fails."""
    
    def __init__(self, simulation_id: str, message: str, details: Optional[dict] = None):
        super().__init__(
            f"Simulation {simulation_id} failed: {message}",
            details={"simulation_id": simulation_id, **(details or {})}
        )
        self.simulation_id = simulation_id

