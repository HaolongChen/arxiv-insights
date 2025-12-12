# Poke API Integration for arXiv Insights

Comprehensive documentation for the Poke API integration in the arXiv Insights repository.

## Overview

The Poke API integration allows arXiv Insights to send processed paper data to the Poke API for further analysis, storage, and processing. The integration is built with enterprise-grade reliability features.

## Key Features

### ✅ Exponential Backoff Retry Logic
- Automatic retry with exponential backoff and jitter
- Prevents thundering herd problem
- Configurable retry attempts (default: 3)
- Smart retry only for transient errors (429, 500, 502, 503, 504, timeouts)

### ✅ Circuit Breaker Pattern
- Prevents cascading failures
- Three states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- Configurable failure threshold (default: 5 failures)
- Automatic recovery attempts after timeout (default: 60s)

### ✅ Request Compression
- Gzip compression for all outgoing requests
- Reduces bandwidth usage significantly
- Metrics tracking for compression ratio
- Transparent to API consumers

### ✅ Metrics Collection
- Request counts (total, success, failed)
- Retry statistics
- Circuit breaker state and failure count
- Data transfer metrics (bytes sent/received)
- Compression efficiency
- Success rate calculation

### ✅ Comprehensive Error Handling
- Specific exception types for different error scenarios
- Graceful degradation when API unavailable
- Detailed logging at appropriate levels
- No data loss on transient failures

### ✅ Rate Limiting
- Client-side rate limiting
- Configurable requests per minute (default: 60)
- Automatic throttling when limit approached
- Rolling window implementation

## Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install requests
```

### Configuration

#### 1. Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```bash
# Required
POKE_API_KEY=pk_sK9vjqit8h_q1kuEiu8obdT4VhDcO-7QPBAz0KP0S08

# Optional (with defaults)
POKE_API_BASE_URL=https://api.poke.example.com
POKE_API_TIMEOUT=30
POKE_API_MAX_RETRIES=3
POKE_API_RATE_LIMIT=60
POKE_API_CIRCUIT_BREAKER_THRESHOLD=5
POKE_API_CIRCUIT_BREAKER_TIMEOUT=60
```

#### 2. GitHub Secrets

For GitHub Actions workflows:

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `POKE_API_KEY`
4. Value: `pk_sK9vjqit8h_q1kuEiu8obdT4VhDcO-7QPBAz0KP0S08`
5. Click **Add secret**

## Usage

### Basic Usage

```bash
# Process a paper and send to Poke API
python scripts/process_paper_with_poke.py papers/2301.00001.json --send-to-poke
```

### Programmatic Usage

```python
from poke_api_client import PokeAPIClient, PokeAPIError

try:
    # Initialize client
    client = PokeAPIClient()
    
    # Prepare paper insights
    paper_insights = {
        'title': 'Novel Approach to Machine Learning',
        'authors': ['Author One', 'Author Two'],
        'abstract': 'This paper presents...',
        'categories': ['cs.AI', 'cs.LG'],
        'key_findings': ['Finding 1', 'Finding 2']
    }
    
    # Send to Poke API
    response = client.send_paper_insights(
        paper_insights,
        arxiv_id='2301.00001'
    )
    
    print(f"Success! Processing ID: {response['id']}")
    
    # Get metrics
    metrics = client.get_metrics()
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Compression: {metrics['avg_compression_ratio']}")
    
    # Close client
    client.close()
    
except PokeAPIError as e:
    print(f"Error: {e}")
```

### Batch Processing

```python
from poke_api_client import PokeAPIClient

client = PokeAPIClient()

papers_list = [
    {'title': 'Paper 1', 'authors': ['A1'], 'categories': ['cs.AI']},
    {'title': 'Paper 2', 'authors': ['A2'], 'categories': ['cs.LG']}
]

response = client.batch_send_papers(papers_list)
print(f"Processed {response['processed_count']} papers")

client.close()
```

## Architecture

### Retry Strategy

| Retry | Error Type | Base Wait | With Jitter | Status Codes |
|-------|-----------|-----------|-------------|-------------|
| 1st | Server/Timeout | 1s | 1.0-1.3s | 408, 500, 502, 503, 504 |
| 2nd | Server/Timeout | 2s | 2.0-2.6s | 408, 500, 502, 503, 504 |
| 3rd | Server/Timeout | 4s | 4.0-5.2s | 408, 500, 502, 503, 504 |
| Special | Rate Limit | 2s | 2.0-2.6s | 429 |

**Jitter:** 30% random variation to prevent thundering herd

### Circuit Breaker States

```
CLOSED (Normal)
  ↓ (5 consecutive failures)
OPEN (Rejecting requests)
  ↓ (60 seconds elapsed)
HALF_OPEN (Testing)
  ↓ (1 success)     ↓ (1 failure)
CLOSED           OPEN
```

### Error Handling Hierarchy

```
PokeAPIError (base)
├── PokeAPIAuthenticationError (401)
├── PokeAPIRateLimitError (429)
├── PokeAPIValidationError (400)
└── CircuitBreakerOpenError (circuit open)
```

## Metrics

The client collects comprehensive metrics:

```json
{
  "requests_total": 150,
  "requests_success": 148,
  "requests_failed": 2,
  "retries_total": 5,
  "circuit_breaker_opens": 0,
  "bytes_sent": 524288,
  "bytes_received": 1048576,
  "compression_ratio": 0.35,
  "circuit_breaker_state": "closed",
  "circuit_breaker_failure_count": 0,
  "rate_limit": 60,
  "current_requests_in_window": 12,
  "success_rate": 0.9867,
  "avg_compression_ratio": "35.00%"
}
```

### Key Metrics Explained

- **success_rate**: Percentage of successful requests (target: >95%)
- **avg_compression_ratio**: Average compression efficiency (lower is better)
- **retries_total**: Total number of retries (indicates API stability)
- **circuit_breaker_opens**: Number of times circuit opened (target: 0)
- **bytes_sent/received**: Data transfer volume

## Error Handling

### Exception Types

#### PokeAPIAuthenticationError
Raised when API key is invalid or missing.

```python
try:
    client = PokeAPIClient()
except PokeAPIAuthenticationError:
    print("Check your API key!")
```

#### PokeAPIRateLimitError
Raised when rate limit exceeded and retries exhausted.

```python
try:
    response = client.send_paper_insights(data)
except PokeAPIRateLimitError:
    print("Rate limit exceeded. Wait before retrying.")
```

#### PokeAPIValidationError
Raised when request validation fails (400 status).

```python
try:
    response = client.send_paper_insights(invalid_data)
except PokeAPIValidationError as e:
    print(f"Validation error: {e}")
```

#### CircuitBreakerOpenError
Raised when circuit breaker is open.

```python
try:
    response = client.send_paper_insights(data)
except CircuitBreakerOpenError:
    print("Service temporarily unavailable. Retry later.")
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failed

**Symptom:** `PokeAPIAuthenticationError`

**Solutions:**
- Verify `POKE_API_KEY` is set correctly
- Check API key is valid: `pk_sK9vjqit8h_q1kuEiu8obdT4VhDcO-7QPBAz0KP0S08`
- Ensure no extra spaces in environment variable

```bash
# Check current value
echo $POKE_API_KEY

# Set correctly
export POKE_API_KEY='pk_sK9vjqit8h_q1kuEiu8obdT4VhDcO-7QPBAz0KP0S08'
```

#### 2. Connection Timeout

**Symptom:** Multiple retry attempts, then failure

**Solutions:**
- Check internet connectivity
- Verify API URL is correct
- Increase timeout: `export POKE_API_TIMEOUT=60`
- Check firewall settings

#### 3. Rate Limit Errors

**Symptom:** `PokeAPIRateLimitError`

**Solutions:**
- Increase rate limit if you have higher quota:
  ```bash
  export POKE_API_RATE_LIMIT=120
  ```
- Use batch operations to reduce request count
- Implement request queuing in your application

#### 4. Circuit Breaker Open

**Symptom:** `CircuitBreakerOpenError`

**Solutions:**
- Wait for circuit breaker timeout (default: 60s)
- Check API service status
- Review logs for root cause of failures
- Adjust circuit breaker settings if needed:
  ```bash
  export POKE_API_CIRCUIT_BREAKER_THRESHOLD=10
  export POKE_API_CIRCUIT_BREAKER_TIMEOUT=120
  ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via environment:

```bash
export LOG_LEVEL=DEBUG
python scripts/process_paper_with_poke.py papers/example.json --send-to-poke
```

### Health Check

Verify API connectivity:

```python
from poke_api_client import PokeAPIClient

client = PokeAPIClient()
if client.health_check():
    print("✓ API is healthy")
else:
    print("✗ API is unhealthy")
client.close()
```

## Best Practices

### 1. Always Close Client

```python
client = PokeAPIClient()
try:
    # Use client
    response = client.send_paper_insights(data)
finally:
    client.close()  # Always close
```

### 2. Handle Exceptions Gracefully

```python
from poke_api_client import (
    PokeAPIClient,
    PokeAPIError,
    CircuitBreakerOpenError
)

try:
    client = PokeAPIClient()
    response = client.send_paper_insights(data)
except CircuitBreakerOpenError:
    # Circuit open - service degraded
    logger.warning("Poke API unavailable, continuing without it")
except PokeAPIError as e:
    # Other API errors
    logger.error(f"Poke API error: {e}")
finally:
    if 'client' in locals():
        client.close()
```

### 3. Monitor Metrics

```python
# Periodically check metrics
metrics = client.get_metrics()
if metrics['success_rate'] < 0.95:
    logger.warning(f"Low success rate: {metrics['success_rate']:.2%}")

if metrics['circuit_breaker_opens'] > 0:
    logger.warning(f"Circuit breaker opened {metrics['circuit_breaker_opens']} times")
```

### 4. Use Batch Operations

For multiple papers, use batch operations to reduce overhead:

```python
# Instead of:
for paper in papers:
    client.send_paper_insights(paper)

# Do:
client.batch_send_papers(papers)
```

### 5. Configure for Your Use Case

```python
# High-throughput scenario
client = PokeAPIClient(
    rate_limit=120,  # Higher rate limit
    max_retries=5,   # More retries
    timeout=60       # Longer timeout
)

# Low-latency scenario
client = PokeAPIClient(
    timeout=10,      # Shorter timeout
    max_retries=1,   # Fewer retries
    rate_limit=30    # Conservative rate
)
```

## Security

### ✅ DO:
- Store API key in environment variables or GitHub Secrets
- Use `.env` file for local development (add to `.gitignore`)
- Rotate API keys regularly
- Use different keys for development and production
- Monitor API usage for anomalies

### ❌ DON'T:
- Commit API keys to version control
- Share API keys in issues or PRs
- Hardcode API keys in code
- Use production keys in development
- Log API keys in application logs

## Performance Optimization

### Compression

The client automatically compresses all requests using gzip:

- **Typical compression ratio:** 60-80% size reduction
- **Benefits:** Reduced bandwidth, faster transfers
- **Overhead:** Minimal CPU usage for compression

### Connection Pooling

The requests session reuses HTTP connections:

- **Benefit:** Reduced connection overhead
- **Implementation:** Automatic via `requests.Session`

### Rate Limiting

Client-side rate limiting prevents API throttling:

- **Strategy:** Rolling window, requests per minute
- **Behavior:** Automatic delay when limit approached
- **Configuration:** Adjust based on your API quota

## GitHub Actions Integration

Example workflow:

```yaml
name: Process Papers with Poke API

on:
  push:
    paths:
      - 'papers/**'

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install requests
      
      - name: Process papers
        env:
          POKE_API_KEY: ${{ secrets.POKE_API_KEY }}
          POKE_API_ENABLED: 'true'
        run: |
          python scripts/process_paper_with_poke.py \\
            papers/latest.json \\
            --send-to-poke
```

## Support

For issues or questions:

1. Check this documentation
2. Review the [troubleshooting section](#troubleshooting)
3. Enable debug logging for detailed information
4. Check GitHub Issues for known problems
5. Contact the repository maintainer

## License

This integration is part of the arXiv Insights repository and follows the same license.
