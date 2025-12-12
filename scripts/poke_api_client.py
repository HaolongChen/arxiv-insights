#!/usr/bin/env python3
"""
Poke API Client for arXiv Insights

Robust client for integrating with the Poke API. Includes:
- Exponential backoff retry logic with jitter
- Circuit breaker pattern for fault tolerance
- Comprehensive error handling
- Rate limiting
- Request/response logging
- Request compression
- Metrics collection
- Configurable via environment variables

Environment Variables:
    POKE_API_KEY: API key for authentication (required)
    POKE_API_BASE_URL: Base URL for Poke API (default: https://api.poke.example.com)
    POKE_API_TIMEOUT: Request timeout in seconds (default: 30)
    POKE_API_MAX_RETRIES: Maximum number of retry attempts (default: 3)
    POKE_API_RATE_LIMIT: Maximum requests per minute (default: 60)
    POKE_API_CIRCUIT_BREAKER_THRESHOLD: Failed requests before circuit opens (default: 5)
    POKE_API_CIRCUIT_BREAKER_TIMEOUT: Seconds before circuit half-opens (default: 60)

Usage:
    from poke_api_client import PokeAPIClient
    
    client = PokeAPIClient()
    response = client.send_paper_insights(paper_insights)
"""

import os
import time
import logging
import random
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import json
import gzip

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    print("Error: requests package not installed.")
    print("Please install: pip install requests")
    raise


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class PokeAPIError(Exception):
    """Base exception for Poke API errors."""
    pass


class PokeAPIAuthenticationError(PokeAPIError):
    """Raised when authentication fails."""
    pass


class PokeAPIRateLimitError(PokeAPIError):
    """Raised when rate limit is exceeded."""
    pass


class PokeAPIValidationError(PokeAPIError):
    """Raised when request validation fails."""
    pass


class CircuitBreakerOpenError(PokeAPIError):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Will retry after {self.timeout}s"
                )
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.timeout)
    
    def _on_success(self):
        """Handle successful request."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker closing after successful test")
            self.state = CircuitState.CLOSED
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(f"Circuit breaker opening after {self.failure_count} failures")
                self.state = CircuitState.OPEN


class PokeAPIClient:
    """
    Client for interacting with the Poke API for arXiv paper insights.
    
    Features:
    - Automatic retry with exponential backoff and jitter
    - Circuit breaker for fault tolerance
    - Rate limiting
    - Request/response logging
    - Comprehensive error handling
    - Request compression (gzip)
    - Metrics collection
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        rate_limit: Optional[int] = None
    ):
        """
        Initialize Poke API client.
        
        Args:
            api_key: API key for authentication (reads from POKE_API_KEY env var if not provided)
            base_url: Base URL for API (reads from POKE_API_BASE_URL env var if not provided)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
            rate_limit: Maximum requests per minute (default: 60)
        
        Raises:
            PokeAPIAuthenticationError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv('POKE_API_KEY')
        if not self.api_key:
            raise PokeAPIAuthenticationError(
                "API key not provided. Set POKE_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = base_url or os.getenv('POKE_API_BASE_URL', 'https://api.poke.example.com')
        self.timeout = timeout or int(os.getenv('POKE_API_TIMEOUT', '30'))
        self.max_retries = max_retries or int(os.getenv('POKE_API_MAX_RETRIES', '3'))
        self.rate_limit = rate_limit or int(os.getenv('POKE_API_RATE_LIMIT', '60'))
        
        # Rate limiting state
        self._request_times: List[datetime] = []
        
        # Circuit breaker
        circuit_threshold = int(os.getenv('POKE_API_CIRCUIT_BREAKER_THRESHOLD', '5'))
        circuit_timeout = int(os.getenv('POKE_API_CIRCUIT_BREAKER_TIMEOUT', '60'))
        self.circuit_breaker = CircuitBreaker(circuit_threshold, circuit_timeout)
        
        # Metrics collection
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'retries_total': 0,
            'circuit_breaker_opens': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'compression_ratio': 0.0
        }
        
        # Create session with retry strategy
        self.session = self._create_session()
        
        logger.info(f"Initialized PokeAPIClient for arXiv Insights with base_url: {self.base_url}")
    
    def _create_session(self) -> requests.Session:
        """
        Create requests session with retry strategy and compression.
        
        Returns:
            Configured requests.Session object
        """
        session = requests.Session()
        
        # Configure retry strategy at the requests library level
        retry_strategy = Retry(
            total=0,  # We handle retries manually with jitter
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers with compression support
        session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'arXiv-Insights-PokeAPI-Client/1.0',
            'Accept-Encoding': 'gzip, deflate, br',  # Enable compression
            'Content-Encoding': 'gzip'  # Compress outgoing requests
        })
        
        return session
    
    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """
        Compress data using gzip.
        
        Args:
            data: Dictionary to compress
        
        Returns:
            Compressed bytes
        """
        json_str = json.dumps(data)
        json_bytes = json_str.encode('utf-8')
        compressed = gzip.compress(json_bytes)
        
        # Update metrics
        original_size = len(json_bytes)
        compressed_size = len(compressed)
        self.metrics['bytes_sent'] += compressed_size
        if original_size > 0:
            compression_ratio = compressed_size / original_size
            self.metrics['compression_ratio'] = (
                (self.metrics['compression_ratio'] + compression_ratio) / 2
            )
        
        logger.debug(f"Compression: {original_size} bytes -> {compressed_size} bytes ({compression_ratio:.2%})")
        return compressed
    
    def _check_rate_limit(self):
        """
        Check if rate limit would be exceeded and wait if necessary.
        """
        now = datetime.now()
        
        # Remove requests older than 1 minute
        self._request_times = [
            t for t in self._request_times
            if now - t < timedelta(minutes=1)
        ]
        
        # Check if we're at the rate limit
        if len(self._request_times) >= self.rate_limit:
            # Calculate wait time
            oldest_request = min(self._request_times)
            wait_time = 60 - (now - oldest_request).total_seconds()
            
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time + 0.1)  # Add small buffer
                
                # Clear old requests after waiting
                now = datetime.now()
                self._request_times = [
                    t for t in self._request_times
                    if now - t < timedelta(minutes=1)
                ]
        
        # Record this request
        self._request_times.append(now)
    
    def _calculate_backoff_with_jitter(self, retry_count: int, base_delay: int = 1) -> float:
        """
        Calculate exponential backoff with jitter to prevent thundering herd.
        
        Args:
            retry_count: Current retry attempt (0-indexed)
            base_delay: Base delay multiplier in seconds
        
        Returns:
            Wait time in seconds with jitter applied
        """
        # Exponential backoff: base_delay * (2 ^ retry_count)
        exponential_delay = base_delay * (2 ** retry_count)
        
        # Add jitter: random value between 0 and 30% of exponential_delay
        jitter = random.uniform(0, exponential_delay * 0.3)
        
        total_delay = exponential_delay + jitter
        logger.debug(f"Backoff calculation: base={exponential_delay}s, jitter={jitter:.2f}s, total={total_delay:.2f}s")
        
        return total_delay
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Poke API with retry logic, circuit breaker, and compression.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., '/paper-insights')
            data: Request body data
            params: Query parameters
            retry_count: Current retry attempt (0-indexed, internal use)
        
        Returns:
            Response data as dictionary
        
        Raises:
            PokeAPIError: For various API errors
            CircuitBreakerOpenError: When circuit breaker is open
        """
        self.metrics['requests_total'] += 1
        
        def _execute_request():
            # Check rate limit
            self._check_rate_limit()
            
            # Build URL
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            # Log request
            logger.info(f"Making {method} request to {url} (attempt {retry_count + 1}/{self.max_retries + 1})")
            if data:
                logger.debug(f"Request data keys: {list(data.keys())}")
            
            try:
                # Compress data if present
                request_kwargs = {
                    'method': method,
                    'url': url,
                    'params': params,
                    'timeout': self.timeout
                }
                
                if data:
                    compressed_data = self._compress_data(data)
                    request_kwargs['data'] = compressed_data
                    request_kwargs['headers'] = {'Content-Encoding': 'gzip'}
                
                # Make request
                response = self.session.request(**request_kwargs)
                
                # Track received bytes
                if 'content-length' in response.headers:
                    self.metrics['bytes_received'] += int(response.headers['content-length'])
                
                # Log response
                logger.info(f"Response status: {response.status_code}")
                
                # Handle different status codes
                if response.status_code in (200, 201):
                    result = response.json()
                    logger.info("Request successful")
                    self.metrics['requests_success'] += 1
                    return result
                
                elif response.status_code == 401:
                    raise PokeAPIAuthenticationError(
                        f"Authentication failed: {response.text}"
                    )
                
                elif response.status_code == 400:
                    raise PokeAPIValidationError(
                        f"Request validation failed: {response.text}"
                    )
                
                elif response.status_code == 429:
                    # Rate limit exceeded - retry with exponential backoff
                    if retry_count < self.max_retries:
                        self.metrics['retries_total'] += 1
                        wait_time = self._calculate_backoff_with_jitter(retry_count, base_delay=2)
                        logger.warning(
                            f"Rate limit exceeded (429). Retrying in {wait_time:.2f} seconds... "
                            f"(Attempt {retry_count + 1}/{self.max_retries})"
                        )
                        time.sleep(wait_time)
                        return self._make_request(method, endpoint, data, params, retry_count + 1)
                    else:
                        raise PokeAPIRateLimitError(
                            f"Rate limit exceeded and max retries ({self.max_retries}) reached: {response.text}"
                        )
                
                elif response.status_code >= 500:
                    # Server error - retry with exponential backoff
                    if retry_count < self.max_retries:
                        self.metrics['retries_total'] += 1
                        wait_time = self._calculate_backoff_with_jitter(retry_count, base_delay=1)
                        logger.warning(
                            f"Server error {response.status_code}. Retrying in {wait_time:.2f} seconds... "
                            f"(Attempt {retry_count + 1}/{self.max_retries})"
                        )
                        time.sleep(wait_time)
                        return self._make_request(method, endpoint, data, params, retry_count + 1)
                    else:
                        raise PokeAPIError(
                            f"Server error and max retries ({self.max_retries}) reached: {response.status_code} - {response.text}"
                        )
                
                else:
                    raise PokeAPIError(
                        f"Unexpected status code {response.status_code}: {response.text}"
                    )
            
            except requests.exceptions.Timeout:
                if retry_count < self.max_retries:
                    self.metrics['retries_total'] += 1
                    wait_time = self._calculate_backoff_with_jitter(retry_count, base_delay=1)
                    logger.warning(
                        f"Request timeout. Retrying in {wait_time:.2f} seconds... "
                        f"(Attempt {retry_count + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    return self._make_request(method, endpoint, data, params, retry_count + 1)
                else:
                    raise PokeAPIError(
                        f"Request timeout and max retries ({self.max_retries}) reached"
                    )
            
            except requests.exceptions.ConnectionError as e:
                if retry_count < self.max_retries:
                    self.metrics['retries_total'] += 1
                    wait_time = self._calculate_backoff_with_jitter(retry_count, base_delay=1)
                    logger.warning(
                        f"Connection error. Retrying in {wait_time:.2f} seconds... "
                        f"(Attempt {retry_count + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    return self._make_request(method, endpoint, data, params, retry_count + 1)
                else:
                    raise PokeAPIError(
                        f"Connection error and max retries ({self.max_retries}) reached: {str(e)}"
                    )
            
            except requests.exceptions.RequestException as e:
                raise PokeAPIError(f"Request failed: {str(e)}")
        
        # Execute with circuit breaker
        try:
            return self.circuit_breaker.call(_execute_request)
        except CircuitBreakerOpenError:
            self.metrics['circuit_breaker_opens'] += 1
            raise
        except Exception as e:
            self.metrics['requests_failed'] += 1
            raise e
    
    def send_paper_insights(
        self,
        paper_data: Dict[str, Any],
        arxiv_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send processed arXiv paper insights to Poke API.
        
        Args:
            paper_data: Dictionary containing paper insights
            arxiv_id: Optional arXiv paper identifier
        
        Returns:
            API response containing processing results
        
        Example:
            paper_insights = {
                'title': 'Paper Title',
                'authors': ['Author 1', 'Author 2'],
                'abstract': 'Abstract text',
                'categories': ['cs.AI', 'cs.LG'],
                'key_findings': ['Finding 1', 'Finding 2']
            }
            response = client.send_paper_insights(paper_insights, arxiv_id='2301.00001')
        """
        logger.info(f"Sending paper insights to Poke API{' for arXiv ID ' + arxiv_id if arxiv_id else ''}")
        
        # Prepare payload
        payload = {
            'paper_insights': paper_data,
            'timestamp': datetime.now().isoformat(),
            'source': 'arxiv-insights',
            'client_version': '1.0'
        }
        
        if arxiv_id:
            payload['arxiv_id'] = arxiv_id
        
        # Make request
        response = self._make_request('POST', '/paper-insights', data=payload)
        
        logger.info(f"Paper insights sent successfully. Response ID: {response.get('id', 'N/A')}")
        return response
    
    def batch_send_papers(
        self,
        papers_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Send multiple paper insights in a single batch request.
        
        Args:
            papers_list: List of paper insights dictionaries
        
        Returns:
            Batch processing results
        """
        logger.info(f"Sending batch of {len(papers_list)} paper insights to Poke API")
        
        payload = {
            'papers_batch': papers_list,
            'timestamp': datetime.now().isoformat(),
            'source': 'arxiv-insights',
            'batch_size': len(papers_list),
            'client_version': '1.0'
        }
        
        response = self._make_request('POST', '/paper-insights/batch', data=payload)
        
        logger.info(f"Batch sent successfully. Processed: {response.get('processed_count', 0)}")
        return response
    
    def health_check(self) -> bool:
        """
        Check if Poke API is accessible and healthy.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            logger.info("Performing health check")
            response = self._make_request('GET', '/health')
            return response.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get client metrics for monitoring.
        
        Returns:
            Dictionary of metrics including:
            - Request counts (total, success, failed)
            - Retry statistics
            - Circuit breaker state
            - Data transfer metrics
            - Compression ratio
        """
        return {
            **self.metrics,
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'circuit_breaker_failure_count': self.circuit_breaker.failure_count,
            'rate_limit': self.rate_limit,
            'current_requests_in_window': len(self._request_times),
            'success_rate': (
                self.metrics['requests_success'] / self.metrics['requests_total']
                if self.metrics['requests_total'] > 0 else 0.0
            ),
            'avg_compression_ratio': f"{self.metrics['compression_ratio']:.2%}"
        }
    
    def close(self):
        """Close the session and cleanup resources."""
        self.session.close()
        logger.info("PokeAPIClient session closed")
        
        # Log final metrics
        metrics = self.get_metrics()
        logger.info(f"Final metrics: {json.dumps(metrics, indent=2)}")


if __name__ == '__main__':
    # Example usage
    print("Poke API Client for arXiv Insights")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv('POKE_API_KEY'):
        print("Error: POKE_API_KEY environment variable not set")
        print("Set it with: export POKE_API_KEY='your-api-key'")
    else:
        try:
            # Initialize client
            client = PokeAPIClient()
            
            # Health check
            if client.health_check():
                print("✓ API is healthy")
            else:
                print("✗ API health check failed")
            
            # Example paper data
            paper_insights = {
                'title': 'Example arXiv Paper',
                'authors': ['Author One', 'Author Two'],
                'abstract': 'This is an example abstract for testing.',
                'categories': ['cs.AI', 'cs.LG'],
                'key_findings': [
                    'Novel approach to machine learning',
                    'Improved performance on benchmark datasets'
                ]
            }
            
            # Send paper insights
            response = client.send_paper_insights(paper_insights, arxiv_id='2301.00001')
            print(f"\n✓ Paper insights sent successfully")
            print(f"Response: {json.dumps(response, indent=2)}")
            
            # Show metrics
            print(f"\nClient Metrics:")
            print(json.dumps(client.get_metrics(), indent=2))
            
            # Close client
            client.close()
            
        except PokeAPIError as e:
            print(f"\n✗ Poke API Error: {str(e)}")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
