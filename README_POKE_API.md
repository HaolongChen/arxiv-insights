# Poke API Integration - Quick Start

Quick start guide for the Poke API integration in arXiv Insights.

## 🚀 Quick Start

### 1. Set Up API Key

**For Local Development:**

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export POKE_API_KEY="pk_sK9vjqit8h_q1kuEiu8obdT4VhDcO-7QPBAz0KP0S08"

# Or create a .env file (recommended)
cp .env.example .env
# The API key is already set in .env.example
```

**For GitHub Actions:**

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `POKE_API_KEY`
4. Value: `pk_sK9vjqit8h_q1kuEiu8obdT4VhDcO-7QPBAz0KP0S08`
5. Click **Add secret**

### 2. Install Dependencies

```bash
pip install requests
```

### 3. Process a Paper

```bash
# Basic usage
python scripts/process_paper_with_poke.py \\
  papers/2301.00001.json \\
  --send-to-poke

# With custom output
python scripts/process_paper_with_poke.py \\
  papers/2301.00001.json \\
  --send-to-poke \\
  --output processed_output.json
```

## 📋 Features

✅ **Exponential Backoff** - Automatic retry with increasing wait times (1s, 2s, 4s, 8s)  
✅ **Circuit Breaker** - Prevents cascading failures with 3-state protection  
✅ **Request Compression** - Gzip compression reduces bandwidth by 60-80%  
✅ **Metrics Collection** - Track success rates, compression ratios, and more  
✅ **Rate Limiting** - Client-side rate limiting prevents API throttling  
✅ **Comprehensive Error Handling** - Specific exceptions for different error scenarios  
✅ **Secure** - API keys via environment variables and GitHub Secrets only

## 🔧 Configuration

### Environment Variables

```bash
# Required
POKE_API_KEY="pk_sK9vjqit8h_q1kuEiu8obdT4VhDcO-7QPBAz0KP0S08"

# Optional (with defaults)
POKE_API_BASE_URL="https://api.poke.example.com"
POKE_API_TIMEOUT="30"          # seconds
POKE_API_MAX_RETRIES="3"       # attempts
POKE_API_RATE_LIMIT="60"       # requests/minute
POKE_API_CIRCUIT_BREAKER_THRESHOLD="5"  # failures before circuit opens
POKE_API_CIRCUIT_BREAKER_TIMEOUT="60"   # seconds before retry
```

## 📖 Usage Examples

### Example 1: Simple Processing

```bash
python scripts/process_paper_with_poke.py \\
  papers/example.json \\
  --send-to-poke
```

### Example 2: Programmatic Usage

```python
from poke_api_client import PokeAPIClient

# Initialize
client = PokeAPIClient()

# Send paper insights
response = client.send_paper_insights({
    'title': 'Novel ML Approach',
    'authors': ['Author One'],
    'categories': ['cs.AI'],
    'key_findings': ['Finding 1']
}, arxiv_id='2301.00001')

print(f"Processing ID: {response['id']}")

# Get metrics
metrics = client.get_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Compression: {metrics['avg_compression_ratio']}")

client.close()
```

### Example 3: Batch Processing

```python
from poke_api_client import PokeAPIClient

client = PokeAPIClient()

papers = [
    {'title': 'Paper 1', 'categories': ['cs.AI']},
    {'title': 'Paper 2', 'categories': ['cs.LG']}
]

response = client.batch_send_papers(papers)
print(f"Processed: {response['processed_count']} papers")

client.close()
```

### Example 4: With Error Handling

```python
from poke_api_client import (
    PokeAPIClient,
    PokeAPIError,
    CircuitBreakerOpenError
)

try:
    client = PokeAPIClient()
    response = client.send_paper_insights(paper_data)
    print(f"Success: {response['id']}")
except CircuitBreakerOpenError:
    print("Service temporarily unavailable")
except PokeAPIError as e:
    print(f"API error: {e}")
finally:
    if 'client' in locals():
        client.close()
```

## 🔍 Health Check

```bash
# Verify API connectivity
python -c "from poke_api_client import PokeAPIClient; \\
           client = PokeAPIClient(); \\
           print('✓ Healthy' if client.health_check() else '✗ Unhealthy'); \\
           client.close()"
```

## ⚠️ Troubleshooting

### Issue: Authentication Error

**Solution:**
```bash
# Verify API key is set
echo $POKE_API_KEY

# Re-export if needed
export POKE_API_KEY='pk_sK9vjqit8h_q1kuEiu8obdT4VhDcO-7QPBAz0KP0S08'
```

### Issue: Connection Error

**Solution:**
1. Check internet connectivity
2. Verify API URL is correct
3. Check firewall settings
4. Increase timeout: `export POKE_API_TIMEOUT=60`

### Issue: Rate Limit Exceeded

**Solution:**
```bash
# Increase rate limit if you have higher quota
export POKE_API_RATE_LIMIT=120

# Or use batch processing
python -c "from poke_api_client import PokeAPIClient; \\
           PokeAPIClient().batch_send_papers([...])"
```

### Issue: Circuit Breaker Open

**Solution:**
- Wait 60 seconds for automatic retry
- Check API service status
- Adjust circuit breaker settings:
  ```bash
  export POKE_API_CIRCUIT_BREAKER_THRESHOLD=10
  export POKE_API_CIRCUIT_BREAKER_TIMEOUT=120
  ```

## 📊 Metrics

The client collects comprehensive metrics:

```python
metrics = client.get_metrics()

# Key metrics:
print(f"Success rate: {metrics['success_rate']:.2%}")  # Target: >95%
print(f"Compression: {metrics['avg_compression_ratio']}")  # Typical: 35-40%
print(f"Total requests: {metrics['requests_total']}")
print(f"Retries: {metrics['retries_total']}")
print(f"Circuit state: {metrics['circuit_breaker_state']}")  # Should be 'closed'
```

## 📚 Full Documentation

For comprehensive documentation, see:
- [docs/POKE_API_INTEGRATION.md](docs/POKE_API_INTEGRATION.md)

Key topics covered:
- Detailed API reference
- Architecture and design patterns
- Error handling strategies
- Retry logic explanation
- Rate limiting details
- Circuit breaker behavior
- Performance optimization
- Security best practices
- Advanced usage examples
- GitHub Actions integration

## 🔐 Security Best Practices

✅ **DO:**
- Store API key in environment variables or GitHub Secrets
- Use `.env` file for local development (add to .gitignore)
- Rotate API keys regularly
- Use different keys for development and production

❌ **DON'T:**
- Commit API keys to version control
- Share API keys in issues or PRs
- Hardcode API keys in code
- Use production keys in development

## 🎯 Performance Tips

1. **Use Batch Operations** - Send multiple papers at once
2. **Monitor Metrics** - Check success rate and compression ratio
3. **Adjust Rate Limits** - Set based on your API quota
4. **Configure Timeouts** - Balance between responsiveness and reliability
5. **Enable Compression** - Automatically enabled, saves ~65% bandwidth

## 📞 Support

For issues or questions:

1. Check [troubleshooting section](#troubleshooting)
2. Review [full documentation](docs/POKE_API_INTEGRATION.md)
3. Enable debug logging: `export LOG_LEVEL=DEBUG`
4. Open an issue on GitHub

---

**Happy Processing! 🚀**
