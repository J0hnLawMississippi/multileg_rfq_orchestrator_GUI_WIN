#!/usr/bin/env python3
"""
Deribit API Handler Classes
REST API with JSON-RPC 2.0 protocol
Client Credentials authentication with session token
POST request support and connection pooling
"""

import asyncio
import time
from enum import Enum
from typing import Dict, Any, Optional, Union, List
import aiohttp
from aiolimiter import AsyncLimiter


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class DeribitError(Exception):
    """Base exception for all Deribit API errors."""
    def __init__(self, message: str, code: Optional[int] = None):
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}" if code else message)


class DeribitAuthError(DeribitError):
    """Authentication or authorization failures."""
    pass


class DeribitTokenExpiredError(DeribitAuthError):
    """Access token has expired - triggers re-authentication."""
    pass


class DeribitRateLimitError(DeribitError):
    """Rate limit exceeded."""
    pass


class DeribitInvalidParamsError(DeribitError):
    """Invalid request parameters."""
    pass


class DeribitNotFoundError(DeribitError):
    """Requested resource not found."""
    pass


class DeribitInternalError(DeribitError):
    """Deribit internal server error."""
    pass


class DeribitOrderError(DeribitError):
    """Order-related errors."""
    pass


class DeribitInsufficientFundsError(DeribitError):
    """Insufficient funds for operation."""
    pass


# Error code mapping based on Deribit API documentation
DERIBIT_ERROR_MAP = {
    # Authentication errors
    13009: DeribitAuthError,              # invalid_credentials
    13010: DeribitAuthError,              # invalid_token
    13004: DeribitTokenExpiredError,      # token_expired
    13005: DeribitTokenExpiredError,      # token_invalid
    13006: DeribitAuthError,              # unauthorized
    
    # Rate limiting
    10028: DeribitRateLimitError,         # too_many_requests
    
    # Parameter errors
    10001: DeribitInvalidParamsError,     # invalid_params
    10002: DeribitInvalidParamsError,     # invalid_argument
    
    # Resource errors
    11044: DeribitNotFoundError,          # not_found
    11050: DeribitNotFoundError,          # instrument_not_found
    
    # Order errors
    10009: DeribitOrderError,             # not_enough_funds (legacy)
    10010: DeribitInsufficientFundsError, # insufficient_funds
    10011: DeribitOrderError,             # order_not_found
    10012: DeribitOrderError,             # price_too_high
    10013: DeribitOrderError,             # price_too_low
    10014: DeribitOrderError,             # invalid_quantity
    
    # Internal errors
    10000: DeribitInternalError,          # internal_error
    11051: DeribitInternalError,          # system_maintenance
}


def raise_for_deribit_error(error_data: Dict[str, Any]) -> None:
    """Parse error response and raise appropriate exception."""
    code = error_data.get("code")
    message = error_data.get("message", "Unknown error")
    data = error_data.get("data", {})
    
    # Include additional error data in message if present
    if data:
        message = f"{message} - {data}"
    
    exception_class = DERIBIT_ERROR_MAP.get(code, DeribitError)
    raise exception_class(message, code)


# =============================================================================
# REQUEST METHOD ENUM
# =============================================================================

class RequestMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"


# =============================================================================
# CONNECTION POOL CONFIGURATION
# =============================================================================

class ConnectionPoolConfig:
    """Configuration for aiohttp connection pooling."""
    
    def __init__(
        self,
        limit: int = 100,
        limit_per_host: int = 30,
        ttl_dns_cache: int = 300,
        keepalive_timeout: int = 60,
        enable_cleanup_closed: bool = True
    ):
        """
        Args:
            limit: Total number of simultaneous connections
            limit_per_host: Connections per host
            ttl_dns_cache: DNS cache TTL in seconds
            keepalive_timeout: Keep-alive timeout in seconds
            enable_cleanup_closed: Clean up closed connections
        """
        self.limit = limit
        self.limit_per_host = limit_per_host
        self.ttl_dns_cache = ttl_dns_cache
        self.keepalive_timeout = keepalive_timeout
        self.enable_cleanup_closed = enable_cleanup_closed
    
    def create_connector(self) -> aiohttp.TCPConnector:
        """Create configured TCP connector."""
        return aiohttp.TCPConnector(
            limit=self.limit,
            limit_per_host=self.limit_per_host,
            ttl_dns_cache=self.ttl_dns_cache,
            keepalive_timeout=self.keepalive_timeout,
            enable_cleanup_closed=self.enable_cleanup_closed
        )


# Default connection pool configuration
DEFAULT_POOL_CONFIG = ConnectionPoolConfig()


# =============================================================================
# BASE API HANDLER
# =============================================================================

class DeribitApiHandler:
    """
    Base handler for Deribit REST API using JSON-RPC 2.0.
    Manages aiohttp session with connection pooling, rate limiting,
    and request/response handling.
    """
    
    PRODUCTION_URL = "https://www.deribit.com"
    TESTNET_URL = "https://test.deribit.com"
    API_PATH = "/api/v2"
    
    def __init__(
        self,
        testnet: bool = False,
        rate_limit: int = 20,
        time_period: float = 1.0,
        pool_config: Optional[ConnectionPoolConfig] = None,
        timeout: Optional[aiohttp.ClientTimeout] = None
    ):
        """
        Args:
            testnet: Use testnet environment if True
            rate_limit: Maximum requests per time_period
            time_period: Rate limit window in seconds
            pool_config: Connection pool configuration
            timeout: Request timeout configuration
        """
        self._base_url = self.TESTNET_URL if testnet else self.PRODUCTION_URL
        self._rate_limiter = AsyncLimiter(rate_limit, time_period)
        self._pool_config = pool_config or DEFAULT_POOL_CONFIG
        self._timeout = timeout or aiohttp.ClientTimeout(total=30, connect=10)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._request_id_counter = 0
    
    @property
    def is_testnet(self) -> bool:
        """Check if handler is configured for testnet."""
        return self._base_url == self.TESTNET_URL
    
    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self._base_url
    
    def _generate_request_id(self) -> int:
        """Generate timestamp-based unique request ID."""
        timestamp_ms = int(time.time() * 1000)
        self._request_id_counter = (self._request_id_counter + 1) % 10000
        return timestamp_ms * 10000 + self._request_id_counter
    
    def _build_jsonrpc_payload(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Construct JSON-RPC 2.0 request payload."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._generate_request_id(),
            "method": method
        }
        if params:
            filtered_params = {k: v for k, v in params.items() if v is not None}
            if filtered_params:
                payload["params"] = filtered_params
        return payload
    
    async def __aenter__(self):
        """Create session with connection pooling."""
        self._connector = self._pool_config.create_connector()
        self._session = aiohttp.ClientSession(
            base_url=self._base_url,
            connector=self._connector,
            timeout=self._timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session and connector."""
        if self._session:
            await self._session.close()
            self._session = None
        if self._connector:
            await self._connector.close()
            self._connector = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Return headers for request. Override in subclasses for auth."""
        return {"Content-Type": "application/json"}
    
    async def _execute_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        method: RequestMethod
    ) -> Dict[str, Any]:
        """Execute HTTP request and handle response."""
        async with self._rate_limiter:
            if method == RequestMethod.GET:
                async with self._session.get(
                    endpoint,
                    params=payload.get("params"),
                    headers=headers
                ) as response:
                    return await self._handle_response(response)
            else:
                async with self._session.post(
                    endpoint,
                    json=payload,
                    headers=headers
                ) as response:
                    return await self._handle_response(response)
    
    async def _handle_response(
        self,
        response: aiohttp.ClientResponse
    ) -> Dict[str, Any]:
        """Process HTTP response and handle errors."""
        if response.status == 429:
            raise DeribitRateLimitError("HTTP 429: Rate limit exceeded")
        
        response.raise_for_status()
        data = await response.json()
        
        if "error" in data:
            raise_for_deribit_error(data["error"])
        
        return data
    
    async def make_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_method: RequestMethod = RequestMethod.GET
    ) -> Dict[str, Any]:
        """
        Execute JSON-RPC request to Deribit API.
        
        Args:
            method: Deribit API method (e.g., "public/get_instruments")
            params: Method parameters
            request_method: HTTP method (GET or POST)
            
        Returns:
            Full JSON-RPC response dict containing 'jsonrpc', 'id', and 'result'
            
        Raises:
            DeribitError: On API errors
            RuntimeError: If handler not in async context
        """
        if not self._session:
            raise RuntimeError("Handler must be used within an `async with` block.")
        
        payload = self._build_jsonrpc_payload(method, params)
        headers = self._get_headers()
        endpoint = f"{self.API_PATH}/{method}"
        
        print(f"[{time.time():.2f}] Deribit {request_method.value}: {method}")
        
        return await self._execute_request(endpoint, payload, headers, request_method)
    
    async def get(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convenience method for GET requests."""
        return await self.make_request(method, params, RequestMethod.GET)
    
    async def post(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convenience method for POST requests."""
        return await self.make_request(method, params, RequestMethod.POST)


# =============================================================================
# CACHED API HANDLER
# =============================================================================

class CachedDeribitApiHandler(DeribitApiHandler):
    """
    Deribit API handler with response caching.
    Suitable for data that doesn't change frequently.
    """
    
    def __init__(
        self,
        cache_ttl_sec: int = 3600,
        testnet: bool = False,
        **kwargs
    ):
        super().__init__(testnet=testnet, **kwargs)
        self._cache: Dict[str, Any] = {}
        self._cache_age: Dict[str, float] = {}
        self._cache_ttl = cache_ttl_sec
    
    def _generate_cache_key(
        self,
        method: str,
        params: Optional[Dict] = None,
        request_method: RequestMethod = RequestMethod.GET
    ) -> str:
        """Generate unique cache key from method, params, and request type."""
        key_parts = [request_method.value, method]
        if params:
            sorted_params = sorted(
                [(k, v) for k, v in params.items() if v is not None],
                key=lambda x: x[0]
            )
            key_parts.append(str(sorted_params))
        return ":".join(key_parts)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached entry exists and hasn't expired."""
        if key not in self._cache:
            return False
        age = time.time() - self._cache_age.get(key, 0)
        return age < self._cache_ttl
    
    def invalidate_cache(self, method: Optional[str] = None) -> None:
        """
        Invalidate cache entries.
        
        Args:
            method: If provided, invalidate only entries for this method.
                    If None, invalidate entire cache.
        """
        if method is None:
            self._cache.clear()
            self._cache_age.clear()
        else:
            keys_to_remove = [k for k in self._cache if method in k]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_age.pop(key, None)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        now = time.time()
        valid_entries = sum(
            1 for k in self._cache
            if self._is_cache_valid(k)
        )
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "ttl_seconds": self._cache_ttl
        }
    
    async def make_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_method: RequestMethod = RequestMethod.GET,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Execute request with optional caching."""
        cache_key = self._generate_cache_key(method, params, request_method)
        
        if use_cache and self._is_cache_valid(cache_key):
            print(f"[{time.time():.2f}] Returning CACHED data for: {method}")
            return self._cache[cache_key]
        
        data = await super().make_request(method, params, request_method)
        
        if use_cache:
            self._cache[cache_key] = data
            self._cache_age[cache_key] = time.time()
        
        return data
    
    async def get(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Convenience method for GET requests with caching."""
        return await self.make_request(method, params, RequestMethod.GET, use_cache)
    
    async def post(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Convenience method for POST requests with caching."""
        return await self.make_request(method, params, RequestMethod.POST, use_cache)


# =============================================================================
# SIGNED API HANDLER (Private Endpoints)
# =============================================================================

class DeribitSignedApiHandler(DeribitApiHandler):
    """
    Authenticated handler for Deribit private endpoints.
    Uses client credentials flow with automatic re-authentication on token expiry.
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        testnet: bool = False,
        **kwargs
    ):
        super().__init__(testnet=testnet, **kwargs)
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: float = 0
        self._token_scope: Optional[str] = None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if we have a valid (non-expired) access token."""
        if not self._access_token:
            return False
        return time.time() < (self._token_expiry - 30)
    
    @property
    def token_expires_in(self) -> float:
        """Seconds until token expiry. Negative if expired."""
        return self._token_expiry - time.time()
    
    @property
    def token_scope(self) -> Optional[str]:
        """Return the scope of the current access token."""
        return self._token_scope
    
    async def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate using client credentials.
        
        Returns:
            Full authentication response
            
        Raises:
            DeribitAuthError: On authentication failure
        """
        if not self._session:
            raise RuntimeError("Handler must be used within an `async with` block.")
        
        params = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret
        }
        
        payload = self._build_jsonrpc_payload("public/auth", params)
        endpoint = f"{self.API_PATH}/public/auth"
        headers = {"Content-Type": "application/json"}
        
        print(f"[{time.time():.2f}] Deribit authentication request")
        
        # Use POST for authentication
        async with self._rate_limiter:
            async with self._session.post(
                endpoint,
                json=payload,
                headers=headers
            ) as response:
                data = await self._handle_response(response)
                
                result = data.get("result", {})
                self._access_token = result.get("access_token")
                self._refresh_token = result.get("refresh_token")
                self._token_scope = result.get("scope")
                expires_in = result.get("expires_in", 900)
                self._token_expiry = time.time() + expires_in
                
                print(f"[{time.time():.2f}] Authenticated. Token expires in {expires_in}s")
                
                return data
    
    async def refresh_authentication(self) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.
        
        Returns:
            Full authentication response
            
        Raises:
            DeribitAuthError: If no refresh token or refresh fails
        """
        if not self._refresh_token:
            raise DeribitAuthError("No refresh token available", None)
        
        if not self._session:
            raise RuntimeError("Handler must be used within an `async with` block.")
        
        params = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token
        }
        
        payload = self._build_jsonrpc_payload("public/auth", params)
        endpoint = f"{self.API_PATH}/public/auth"
        headers = {"Content-Type": "application/json"}
        
        print(f"[{time.time():.2f}] Deribit token refresh request")
        
        async with self._rate_limiter:
            async with self._session.post(
                endpoint,
                json=payload,
                headers=headers
            ) as response:
                data = await self._handle_response(response)
                
                result = data.get("result", {})
                self._access_token = result.get("access_token")
                self._refresh_token = result.get("refresh_token")
                self._token_scope = result.get("scope")
                expires_in = result.get("expires_in", 900)
                self._token_expiry = time.time() + expires_in
                
                print(f"[{time.time():.2f}] Token refreshed. Expires in {expires_in}s")
                
                return data
    
    async def logout(self) -> Dict[str, Any]:
        """
        Invalidate current session token.
        
        Returns:
            Logout response
        """
        if not self._access_token:
            raise DeribitAuthError("Not authenticated", None)
        
        response = await self.make_request(
            "private/logout",
            request_method=RequestMethod.POST
        )
        
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = 0
        self._token_scope = None
        
        return response
    
    def _get_headers(self) -> Dict[str, str]:
        """Return headers with authorization token."""
        headers = {"Content-Type": "application/json"}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers
    
    async def _execute_with_retry(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        request_method: RequestMethod,
        auto_auth: bool,
        requires_auth: bool
    ) -> Dict[str, Any]:
        """Execute request with automatic re-authentication on token expiry."""
        try:
            return await self._execute_request(
                endpoint, payload, headers, request_method
            )
        except DeribitTokenExpiredError:
            if not auto_auth or not requires_auth:
                raise
            
            print(f"[{time.time():.2f}] Token expired, re-authenticating...")
            self._access_token = None
            await self.authenticate()
            
            # Update headers with new token
            headers = self._get_headers()
            return await self._execute_request(
                endpoint, payload, headers, request_method
            )
    
    async def make_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_method: RequestMethod = RequestMethod.GET,
        auto_auth: bool = True
    ) -> Dict[str, Any]:
        """
        Execute authenticated JSON-RPC request.
        
        Automatically authenticates if needed and re-authenticates on token expiry.
        
        Args:
            method: Deribit API method
            params: Method parameters
            request_method: HTTP method (GET or POST)
            auto_auth: If True, automatically authenticate when needed
            
        Returns:
            Full JSON-RPC response
        """
        if not self._session:
            raise RuntimeError("Handler must be used within an `async with` block.")
        
        requires_auth = method.startswith("private/")
        
        if requires_auth and auto_auth and not self.is_authenticated:
            await self.authenticate()
        
        payload = self._build_jsonrpc_payload(method, params)
        headers = self._get_headers()
        endpoint = f"{self.API_PATH}/{method}"
        
        print(f"[{time.time():.2f}] Deribit {request_method.value}: {method}")
        
        return await self._execute_with_retry(
            endpoint, payload, headers, request_method, auto_auth, requires_auth
        )
    
    async def get(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        auto_auth: bool = True
    ) -> Dict[str, Any]:
        """Convenience method for authenticated GET requests."""
        return await self.make_request(method, params, RequestMethod.GET, auto_auth)
    
    async def post(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        auto_auth: bool = True
    ) -> Dict[str, Any]:
        """Convenience method for authenticated POST requests."""
        return await self.make_request(method, params, RequestMethod.POST, auto_auth)


# =============================================================================
# CACHED SIGNED API HANDLER
# =============================================================================

class CachedDeribitSignedApiHandler(DeribitSignedApiHandler):
    """
    Authenticated Deribit handler with response caching.
    Combines authentication with caching capabilities.
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        cache_ttl_sec: int = 3600,
        testnet: bool = False,
        **kwargs
    ):
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            testnet=testnet,
            **kwargs
        )
        self._cache: Dict[str, Any] = {}
        self._cache_age: Dict[str, float] = {}
        self._cache_ttl = cache_ttl_sec
    
    def _generate_cache_key(
        self,
        method: str,
        params: Optional[Dict] = None,
        request_method: RequestMethod = RequestMethod.GET
    ) -> str:
        """Generate unique cache key."""
        key_parts = [request_method.value, method]
        if params:
            sorted_params = sorted(
                [(k, v) for k, v in params.items() if v is not None],
                key=lambda x: x[0]
            )
            key_parts.append(str(sorted_params))
        return ":".join(key_parts)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached entry exists and hasn't expired."""
        if key not in self._cache:
            return False
        age = time.time() - self._cache_age.get(key, 0)
        return age < self._cache_ttl
    
    def invalidate_cache(self, method: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if method is None:
            self._cache.clear()
            self._cache_age.clear()
        else:
            keys_to_remove = [k for k in self._cache if method in k]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_age.pop(key, None)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        now = time.time()
        valid_entries = sum(1 for k in self._cache if self._is_cache_valid(k))
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "ttl_seconds": self._cache_ttl
        }
    
    async def make_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_method: RequestMethod = RequestMethod.GET,
        auto_auth: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Execute authenticated request with optional caching."""
        cache_key = self._generate_cache_key(method, params, request_method)
        
        if use_cache and self._is_cache_valid(cache_key):
            print(f"[{time.time():.2f}] Returning CACHED data for: {method}")
            return self._cache[cache_key]
        
        data = await super().make_request(method, params, request_method, auto_auth)
        
        if use_cache:
            self._cache[cache_key] = data
            self._cache_age[cache_key] = time.time()
        
        return data
    
    async def get(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        auto_auth: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Convenience method for authenticated GET requests with caching."""
        return await self.make_request(
            method, params, RequestMethod.GET, auto_auth, use_cache
        )
    
    async def post(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        auto_auth: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Convenience method for authenticated POST requests with caching."""
        return await self.make_request(
            method, params, RequestMethod.POST, auto_auth, use_cache
        )


# =============================================================================
# RESPONSE PARSERS
# =============================================================================

class DeribitResponseParser:
    """Base parser for Deribit JSON-RPC responses."""
    
    def __init__(
        self,
        handler: DeribitApiHandler,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_method: RequestMethod = RequestMethod.GET
    ):
        self.handler = handler
        self.method = method
        self.params = params or {}
        self.request_method = request_method
    
    async def _fetch(self) -> Dict[str, Any]:
        """Fetch raw response from handler."""
        return await self.handler.make_request(
            self.method,
            params=self.params,
            request_method=self.request_method
        )


class FullResponseParser(DeribitResponseParser):
    """Returns the full JSON-RPC response."""
    
    async def get(self) -> Dict[str, Any]:
        return await self._fetch()


class ResultParser(DeribitResponseParser):
    """Extracts and returns only the 'result' field."""
    
    async def get(self) -> Any:
        data = await self._fetch()
        return data.get("result")


class ListParser(DeribitResponseParser):
    """Extracts 'result' and validates it's a list."""
    
    async def get(self) -> List[Any]:
        data = await self._fetch()
        result = data.get("result")
        
        if not isinstance(result, list):
            raise ValueError(f"Expected list in result, got {type(result)}")
        
        return result


class DictParser(DeribitResponseParser):
    """Extracts 'result' and validates it's a dict."""
    
    async def get(self) -> Dict[str, Any]:
        data = await self._fetch()
        result = data.get("result")
        
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict in result, got {type(result)}")
        
        return result


class PaginatedParser(DeribitResponseParser):
    """
    Handles paginated responses.
    Automatically fetches all pages and combines results.
    """
    
    def __init__(
        self,
        handler: DeribitApiHandler,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_method: RequestMethod = RequestMethod.GET,
        result_key: str = "trades",
        continuation_key: str = "has_more",
        page_size: int = 1000
    ):
        super().__init__(handler, method, params, request_method)
        self.result_key = result_key
        self.continuation_key = continuation_key
        self.page_size = page_size
    
    async def get(self) -> List[Any]:
        """Fetch all pages and return combined results."""
        all_results = []
        params = dict(self.params)
        params["count"] = self.page_size
        
        while True:
            data = await self.handler.make_request(
                self.method,
                params=params,
                request_method=self.request_method
            )
            
            result = data.get("result", {})
            
            if isinstance(result, dict):
                items = result.get(self.result_key, [])
                all_results.extend(items)
                
                if not result.get(self.continuation_key, False):
                    break
                
                # Update continuation parameter
                if items:
                    last_item = items[-1]
                    if "timestamp" in last_item:
                        params["end_timestamp"] = last_item["timestamp"] - 1
                    elif "trade_id" in last_item:
                        params["end_id"] = last_item["trade_id"]
                    else:
                        break
                else:
                    break
            else:
                all_results.extend(result if isinstance(result, list) else [])
                break
        
        return all_results


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

async def example_public_usage():
    """Example: Public API calls with GET and POST."""
    print("\n=== Public API Example ===\n")
    
    # Custom pool configuration for high-throughput
    pool_config = ConnectionPoolConfig(
        limit=50,
        limit_per_host=20,
        keepalive_timeout=120
    )
    
    async with DeribitApiHandler(testnet=True, pool_config=pool_config) as handler:
        # GET request
        response = await handler.get(
            "public/get_instruments",
            params={"currency": "BTC", "kind": "option"}
        )
        instruments = response.get("result", [])
        print(f"Found {len(instruments)} BTC options via GET")
        
        # POST request (same endpoint, different method)
        response = await handler.post(
            "public/get_index_price",
            params={"index_name": "btc_usd"}
        )
        print(f"BTC Index (POST): {response.get('result')}")
        
        # Using ResultParser for cleaner code
        parser = ResultParser(
            handler,
            "public/get_book_summary_by_currency",
            params={"currency": "BTC", "kind": "future"}
        )
        summary = await parser.get()
        print(f"Futures summary: {len(summary)} instruments")


async def example_cached_usage():
    """Example: Cached API calls."""
    print("\n=== Cached API Example ===\n")
    
    async with CachedDeribitApiHandler(testnet=True, cache_ttl_sec=60) as handler:
        # First call - hits API
        await handler.get("public/get_currencies")
        
        # Second call - cached
        await handler.get("public/get_currencies")
        
        # Check cache stats
        stats = handler.get_cache_stats()
        print(f"Cache stats: {stats}")
        
        # Bypass cache
        await handler.get("public/get_currencies", use_cache=False)
        
        # Invalidate specific method
        handler.invalidate_cache("public/get_currencies")


async def example_private_usage():
    """Example: Authenticated API calls."""
    print("\n=== Private API Example ===\n")
    
    CLIENT_ID = "your_client_id"
    CLIENT_SECRET = "your_client_secret"
    
    async with DeribitSignedApiHandler(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        testnet=True
    ) as handler:
        # Auto-authenticates on first private call
        account = await handler.get(
            "private/get_account_summary",
            params={"currency": "BTC"}
        )
        print(f"Account equity: {account.get('result', {}).get('equity')}")
        
        # POST for trading operations
        # order = await handler.post(
        #     "private/buy",
        #     params={
        #         "instrument_name": "BTC-PERPETUAL",
        #         "amount": 10,
        #         "type": "market"
        #     }
        # )
        
        # Check token status
        print(f"Token expires in: {handler.token_expires_in:.0f}s")
        print(f"Token scope: {handler.token_scope}")


async def example_error_handling():
    """Example: Error handling."""
    print("\n=== Error Handling Example ===\n")
    
    async with DeribitApiHandler(testnet=True) as handler:
        try:
            await handler.get(
                "public/get_instruments",
                params={"currency": "INVALID", "kind": "option"}
            )
        except DeribitInvalidParamsError as e:
            print(f"Invalid params error: {e}")
        except DeribitError as e:
            print(f"Generic Deribit error: {e}")


async def main():
    """Run all examples."""
    await example_public_usage()
    await example_cached_usage()
    await example_error_handling()
    
    # Uncomment with valid credentials
    # await example_private_usage()


if __name__ == "__main__":
    asyncio.run(main())
