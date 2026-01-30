"""Auth0 M2M authentication for Earl SDK."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict
import urllib.request
import urllib.parse
import json

from .exceptions import AuthenticationError


@dataclass
class TokenInfo:
    """Information about an access token."""
    access_token: str
    token_type: str
    expires_at: float  # Unix timestamp
    scope: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if the token is expired (with 60s buffer)."""
        return time.time() >= (self.expires_at - 60)


class Auth0Client:
    """
    Auth0 M2M (Machine-to-Machine) authentication client.
    
    Handles token acquisition and refresh for the Earl API.
    """
    
    # Earl's Auth0 configuration (customers connect to this)
    DEFAULT_DOMAIN = "auth.onlyevals.com"
    DEFAULT_AUDIENCE = "https://api.onlyevals.com"
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        organization: str,
        domain: Optional[str] = None,
        audience: Optional[str] = None,
    ):
        """
        Initialize Auth0 client.
        
        Args:
            client_id: Auth0 M2M application client ID
            client_secret: Auth0 M2M application client secret
            organization: Auth0 organization ID (org_xxx)
            domain: Auth0 domain (defaults to Earl's domain)
            audience: API audience (defaults to Earl's API)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.organization = organization
        self.domain = domain or self.DEFAULT_DOMAIN
        self.audience = audience or self.DEFAULT_AUDIENCE
        
        self._token_info: Optional[TokenInfo] = None
    
    @property
    def token_url(self) -> str:
        """Get the Auth0 token endpoint URL."""
        return f"https://{self.domain}/oauth/token"
    
    def get_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            A valid access token string
            
        Raises:
            AuthenticationError: If token acquisition fails
        """
        if self._token_info and not self._token_info.is_expired:
            return self._token_info.access_token
        
        self._token_info = self._fetch_token()
        return self._token_info.access_token
    
    def _fetch_token(self) -> TokenInfo:
        """Fetch a new token from Auth0."""
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
        }
        
        # Add organization if specified
        if self.organization:
            payload["organization"] = self.organization
        
        data = urllib.parse.urlencode(payload).encode("utf-8")
        
        req = urllib.request.Request(
            self.token_url,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                return TokenInfo(
                    access_token=result["access_token"],
                    token_type=result.get("token_type", "Bearer"),
                    expires_at=time.time() + result.get("expires_in", 86400),
                    scope=result.get("scope"),
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get("error_description", error_data.get("error", str(e)))
            except json.JSONDecodeError:
                error_msg = error_body or str(e)
            
            raise AuthenticationError(
                f"Failed to authenticate with Auth0: {error_msg}",
                details={"status_code": e.code, "domain": self.domain}
            )
        except Exception as e:
            raise AuthenticationError(
                f"Failed to connect to Auth0: {str(e)}",
                details={"domain": self.domain}
            )
    
    def invalidate_token(self) -> None:
        """Force token refresh on next request."""
        self._token_info = None
    
    def get_headers(self) -> dict[str, str]:
        """Get authorization headers for API requests."""
        token = self.get_token()
        headers = {
            "Authorization": f"Bearer {token}",
        }
        # Only add organization header if set
        if self.organization:
            headers["X-Organization-Id"] = self.organization
        return headers

