#!/usr/bin/env python3
"""
arXiv Insights - Paper Processor
Processes incoming arXiv papers and formats them into structured markdown.
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import re

def sanitize_filename(text):
    """Convert text to safe filename."""
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')

def get_field_path(field):
    """Convert field to directory path."""
    field_map = {
        'cs-ai': 'cs/ai',
        'cs-ml': 'cs/ml',
        'cs-cv': 'cs/cv',
        'cs-lg': 'cs/lg',
        'math': 'math',
        'physics': 'physics',
        'q-bio': 'bio',
    }
    return field_map.get(field.lower(), 'other')

def process_paper(data):
    """
    Process incoming paper data and create markdown file.
    
    Expected data format:
    {
        "title": "Paper Title",
        "arxiv_id": "2301.12345",
        "authors": ["Author 1", "Author 2"],
        "date": "2025-01-15",
        "field": "cs-ai",
        "abstract": "Paper abstract...",
        "key_findings": [...],
        "methodology": {...},
        "applications": [...],
        "tags": [...]
    }
    """
    
    # Create directory structure
    field_path = get_field_path(data['field'])
    paper_dir = Path(f'papers/{field_path}')
    paper_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = f"{data['arxiv_id']}-{sanitize_filename(data['title'][:50])}.md"
    filepath = paper_dir / filename
    
    # Format authors
    authors_str = ', '.join(data['authors'])
    authors_yaml = json.dumps(data['authors'])
    
    # Build markdown content
    content = f"""---
title: "{data['title']}"
arxiv_id: "{data['arxiv_id']}"
authors: {authors_yaml}
publication_date: {data['date']}
field: "{data['field']}"
tags: {json.dumps(data.get('tags', []))}
url: "https://arxiv.org/abs/{data['arxiv_id']}"
pdf: "https://arxiv.org/pdf/{data['arxiv_id']}.pdf"
---

# {data['title']}

## Metadata

- **arXiv ID**: [{data['arxiv_id']}](https://arxiv.org/abs/{data['arxiv_id']})
- **Authors**: {authors_str}
- **Published**: {data['date']}
- **Collected**: {datetime.now().strftime('%Y-%m-%d')}
- **Field**: {data['field'].upper()}
- **PDF**: [Download](https://arxiv.org/pdf/{data['arxiv_id']}.pdf)

## Abstract

{data.get('abstract', 'No abstract available.')}

## Key Findings

"""
    
    # Add key findings
    for i, finding in enumerate(data.get('key_findings', []), 1):
        if isinstance(finding, dict):
            content += f"""### {i}. {finding.get('title', f'Finding {i}')}

{finding.get('description', '')}

**Significance**: {finding.get('significance', 'N/A')}

"""
        else:
            content += f"""### {i}. {finding}

"""
    
    # Add methodology
    if data.get('methodology'):
        method = data['methodology']
        content += "\n## Methodology\n\n"
        content += f"{method.get('description', '')}\n\n"
        if method.get('dataset'):
            content += f"**Dataset**: {method['dataset']}\n\n"
        if method.get('techniques'):
            content += "**Novel Techniques**:\n"
            for tech in method['techniques']:
                content += f"- {tech}\n"
            content += "\n"
    
    # Add applications
    if data.get('applications'):
        content += "\n## Applications\n\n"
        for app in data['applications']:
            content += f"- {app}\n"
        content += "\n"
    
    # Add related work
    if data.get('related_work'):
        content += "\n## Related Work\n\n"
        for work in data['related_work']:
            content += f"- {work}\n"
        content += "\n"
    
    # Add tags
    if data.get('tags'):
        content += "\n## Tags\n\n"
        content += ' '.join([f"`{tag}`" for tag in data['tags']])
    
    content += f"\n\n---\n\n*Processed by automation system on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Created paper analysis: {filepath}")
    return filepath

def main():
    """Main processing function."""
    paper_data = os.getenv('PAPER_DATA')
    
    if not paper_data:
        print("ℹ️  No new paper data provided")
        return
    
    try:
        data = json.loads(paper_data)
        process_paper(data)
        print("✅ Processing complete")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error processing paper: {e}")
        exit(1)

if __name__ == '__main__':
    main()