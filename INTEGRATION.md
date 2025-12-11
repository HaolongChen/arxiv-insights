# Integration Guide

This guide explains how to integrate your external automation systems with the arXiv Insights repository.

## Quick Start

### Method 1: GitHub Actions Repository Dispatch (Recommended)

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/HaolongChen/arxiv-insights/dispatches \
  -d '{"event_type":"new-arxiv-paper","client_payload":{"content":YOUR_JSON_DATA}}'
```

### Method 2: Direct arXiv ID Processing

```python
import arxiv

def fetch_and_process_arxiv(arxiv_id):
    """Fetch paper from arXiv API and process it."""
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    
    data = {
        "title": paper.title,
        "arxiv_id": arxiv_id,
        "authors": [author.name for author in paper.authors],
        "date": paper.published.strftime("%Y-%m-%d"),
        "field": "cs-ai",  # Determine from category
        "abstract": paper.summary,
        "url": paper.entry_id,
        # Add your analysis...
    }
    
    # Send to GitHub
    trigger_workflow(data)
```

## Data Format

### Expected JSON Structure

```json
{
  "title": "Attention Is All You Need",
  "arxiv_id": "1706.03762",
  "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  "date": "2017-06-12",
  "field": "cs-ai",
  "abstract": "The dominant sequence transduction models...",
  "key_findings": [
    {
      "title": "Transformer Architecture",
      "description": "Introduced a new architecture...",
      "significance": "Revolutionized NLP"
    }
  ],
  "methodology": {
    "description": "Uses self-attention mechanisms...",
    "dataset": "WMT 2014 English-German",
    "techniques": ["Multi-head attention", "Positional encoding"]
  },
  "applications": [
    "Machine translation",
    "Text generation",
    "Language understanding"
  ],
  "related_work": [
    "Previous seq2seq models",
    "Attention mechanisms"
  ],
  "tags": ["transformers", "attention", "nlp", "deep-learning"]
}
```

### Field Codes

- `cs-ai` - Artificial Intelligence
- `cs-ml` - Machine Learning
- `cs-cv` - Computer Vision
- `cs-lg` - Learning
- `math` - Mathematics
- `physics` - Physics
- `q-bio` - Quantitative Biology

## Python Integration Example

```python
#!/usr/bin/env python3
"""
Example integration for arXiv automation
"""

import requests
import arxiv
from datetime import datetime

class ArXivInsightsIntegration:
    def __init__(self, github_token):
        self.token = github_token
        self.repo = "HaolongChen/arxiv-insights"
        self.base_url = "https://api.github.com"
    
    def trigger_workflow(self, paper_data):
        """Trigger GitHub Actions workflow."""
        url = f"{self.base_url}/repos/{self.repo}/dispatches"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        payload = {
            "event_type": "new-arxiv-paper",
            "client_payload": {"content": paper_data}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        return response.status_code == 204
    
    def process_arxiv_id(self, arxiv_id, analysis=None):
        """Fetch and process paper from arXiv."""
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        # Extract field from categories
        field = self._determine_field(paper.categories)
        
        paper_data = {
            "title": paper.title,
            "arxiv_id": arxiv_id,
            "authors": [author.name for author in paper.authors],
            "date": paper.published.strftime("%Y-%m-%d"),
            "field": field,
            "abstract": paper.summary,
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "tags": paper.categories
        }
        
        # Add your custom analysis
        if analysis:
            paper_data.update(analysis)
        
        return self.trigger_workflow(paper_data)
    
    def _determine_field(self, categories):
        """Map arXiv categories to field codes."""
        primary = categories[0] if categories else ""
        
        if primary.startswith("cs.AI"):
            return "cs-ai"
        elif primary.startswith("cs.LG"):
            return "cs-lg"
        elif primary.startswith("cs.CV"):
            return "cs-cv"
        elif primary.startswith("math"):
            return "math"
        elif primary.startswith("physics"):
            return "physics"
        elif primary.startswith("q-bio"):
            return "q-bio"
        else:
            return "other"

# Usage
if __name__ == "__main__":
    integration = ArXivInsightsIntegration("your_github_token")
    
    # Process a paper with custom analysis
    analysis = {
        "key_findings": [
            {
                "title": "Novel Approach",
                "description": "The paper introduces...",
                "significance": "This could impact..."
            }
        ],
        "methodology": {
            "description": "Uses a combination of...",
            "dataset": "Custom dataset",
            "techniques": ["Method 1", "Method 2"]
        },
        "applications": ["Application 1", "Application 2"]
    }
    
    success = integration.process_arxiv_id("1706.03762", analysis)
    print(f"✅ Paper processed!" if success else "❌ Failed!")
```

## Automated arXiv Monitoring

```python
import arxiv
import time

def monitor_arxiv_category(category="cs.AI", max_results=10):
    """Monitor arXiv for new papers in a category."""
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    integration = ArXivInsightsIntegration("your_token")
    
    for paper in search.results():
        arxiv_id = paper.entry_id.split('/')[-1]
        print(f"Processing: {arxiv_id} - {paper.title}")
        
        # Add your analysis logic here
        # ...
        
        integration.process_arxiv_id(arxiv_id)
        time.sleep(1)  # Rate limiting

# Run daily
if __name__ == "__main__":
    monitor_arxiv_category("cs.AI", max_results=5)
```

## Integration with RSS Feeds

```python
import feedparser

def monitor_arxiv_rss():
    """Monitor arXiv RSS feed."""
    feed_url = "http://export.arxiv.org/rss/cs.AI"
    feed = feedparser.parse(feed_url)
    
    integration = ArXivInsightsIntegration("your_token")
    
    for entry in feed.entries[:10]:
        arxiv_id = entry.id.split('/')[-1]
        # Process each paper
        integration.process_arxiv_id(arxiv_id)
```

## Batch Processing

```python
def batch_process_papers(arxiv_ids):
    """Process multiple papers at once."""
    integration = ArXivInsightsIntegration("your_token")
    results = []
    
    for arxiv_id in arxiv_ids:
        try:
            success = integration.process_arxiv_id(arxiv_id)
            results.append((arxiv_id, success))
            time.sleep(1)  # Respectful rate limiting
        except Exception as e:
            print(f"Error processing {arxiv_id}: {e}")
            results.append((arxiv_id, False))
    
    return results

# Usage
papers = ["1706.03762", "1810.04805", "2010.11929"]
results = batch_process_papers(papers)
```

## Scheduled Monitoring

### Using GitHub Actions (Already configured)

Runs every 8 hours. Modify in `.github/workflows/update-papers.yml`:

```yaml
schedule:
  - cron: '0 */8 * * *'  # Every 8 hours
```

### Using Cron

```bash
# Crontab entry - runs twice daily
0 9,21 * * * /path/to/arxiv-monitor.py
```

## Best Practices

1. **Respect arXiv Terms**: Follow arXiv's API usage guidelines
2. **Rate Limiting**: Don't hammer the API - use delays between requests
3. **Error Handling**: arXiv API can be temperamental - implement retries
4. **Deduplication**: Check if paper already exists before processing
5. **Incremental Updates**: If a paper is updated (v2, v3), track versions

## Advanced Features

### Semantic Analysis

```python
from transformers import pipeline

def analyze_paper_semantics(abstract):
    """Extract key themes from abstract using NLP."""
    classifier = pipeline("zero-shot-classification")
    
    candidate_labels = [
        "deep learning", "nlp", "computer vision",
        "reinforcement learning", "optimization"
    ]
    
    result = classifier(abstract, candidate_labels)
    return result['labels'][:3]  # Top 3 themes
```

### Citation Tracking

```python
import scholarly

def get_citations(title):
    """Get citation count for a paper."""
    search = scholarly.search_pubs(title)
    paper = next(search, None)
    
    if paper:
        return paper.get('num_citations', 0)
    return 0
```

## Monitoring & Alerts

```python
def send_notification(paper_data):
    """Send notification for important papers."""
    # Slack, Discord, Email, etc.
    if should_notify(paper_data):
        # Send alert
        pass

def should_notify(paper_data):
    """Determine if paper is significant."""
    # Check authors, keywords, etc.
    return True  # Your logic here
```

## Troubleshooting

### Common Issues

1. **arXiv API timeout**: Implement retry logic with exponential backoff
2. **Rate limiting**: Add delays between requests
3. **Paper not found**: Some IDs might be invalid or embargoed
4. **Version conflicts**: Handle paper updates (v1, v2, etc.)

---

*Last Updated: December 11, 2025*
