#!/usr/bin/env python3
"""
arXiv Insights - Paper Fetcher
Fetches recent papers from arXiv based on configured categories and keywords.
"""

import arxiv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Configuration
ARXIV_CATEGORIES = [
    'cs.AI',  # Artificial Intelligence
    'cs.LG',  # Machine Learning
    'cs.CL',  # Computation and Language
    'cs.CV',  # Computer Vision
    'cs.NE',  # Neural and Evolutionary Computing
]

KEYWORDS = [
    'large language model',
    'transformer',
    'deep learning',
    'neural network',
    'reinforcement learning',
    'generative AI',
    'GPT',
    'diffusion model',
]

MAX_PAPERS_PER_RUN = int(os.getenv('MAX_PAPERS', '5'))
DAYS_BACK = int(os.getenv('DAYS_BACK', '2'))


def map_category_to_field(category):
    """Map arXiv category to internal field structure."""
    mapping = {
        'cs.AI': 'cs-ai',
        'cs.LG': 'cs-lg',
        'cs.ML': 'cs-ml',
        'cs.CV': 'cs-cv',
        'cs.CL': 'cs-ai',
        'cs.NE': 'cs-ai',
        'math': 'math',
        'physics': 'physics',
        'q-bio': 'q-bio',
    }
    return mapping.get(category, 'other')


def extract_arxiv_id(entry):
    """Extract clean arXiv ID from entry."""
    # Remove version info (e.g., v1, v2)
    arxiv_id = entry.entry_id.split('/')[-1]
    arxiv_id = arxiv_id.split('v')[0]
    return arxiv_id


def paper_exists(arxiv_id):
    """Check if paper already exists in repository."""
    papers_dir = Path('papers')
    if not papers_dir.exists():
        return False
    
    # Search for any markdown file containing this arXiv ID
    for md_file in papers_dir.rglob('*.md'):
        content = md_file.read_text(encoding='utf-8')
        if arxiv_id in content:
            return True
    return False


def fetch_recent_papers():
    """Fetch recent papers from arXiv based on categories and keywords."""
    papers_found = []
    # Make cutoff_date timezone-aware to match arXiv API's result.published
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    
    print(f"🔍 Fetching papers from last {DAYS_BACK} days...")
    print(f"📚 Categories: {', '.join(ARXIV_CATEGORIES)}")
    print(f"🔑 Keywords: {', '.join(KEYWORDS[:3])}...")
    
    # Build search query
    category_query = ' OR '.join([f'cat:{cat}' for cat in ARXIV_CATEGORIES])
    keyword_query = ' OR '.join([f'all:"{kw}"' for kw in KEYWORDS])
    
    # Combine queries
    query = f"({category_query}) AND ({keyword_query})"
    
    print(f"\n🔎 Query: {query}\n")
    
    # Search arXiv
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=50,  # Fetch more, then filter
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    count = 0
    for result in client.results(search):
        # Check if paper is recent enough
        if result.published < cutoff_date:
            continue
        
        arxiv_id = extract_arxiv_id(result)
        
        # Skip if already processed
        if paper_exists(arxiv_id):
            print(f"⏭️  Skipping {arxiv_id} (already exists)")
            continue
        
        # Extract primary category
        primary_cat = result.primary_category
        field = map_category_to_field(primary_cat)
        
        # Build paper data structure
        paper_data = {
            'title': result.title,
            'arxiv_id': arxiv_id,
            'authors': [author.name for author in result.authors],
            'date': result.published.strftime('%Y-%m-%d'),
            'field': field,
            'abstract': result.summary.replace('\n', ' ').strip(),
            'key_findings': [
                'Automated extraction - requires manual review'
            ],
            'methodology': {
                'description': 'See abstract and full paper for details',
                'dataset': 'Not specified',
                'techniques': []
            },
            'applications': [
                'Refer to paper for specific applications'
            ],
            'tags': [primary_cat] + [cat for cat in result.categories if cat != primary_cat],
            'pdf_url': result.pdf_url,
            'comment': result.comment or ''
        }
        
        papers_found.append(paper_data)
        count += 1
        
        print(f"✅ Found: {arxiv_id} - {result.title[:60]}...")
        
        if count >= MAX_PAPERS_PER_RUN:
            break
    
    return papers_found


def save_papers_batch(papers):
    """Save papers to a JSON file for batch processing."""
    if not papers:
        print("\n📭 No new papers found")
        return None
    
    output_file = Path('papers-batch.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved {len(papers)} papers to {output_file}")
    return output_file


def main():
    """Main function."""
    print("=" * 70)
    print("🔬 arXiv Paper Fetcher")
    print("=" * 70)
    print()
    
    try:
        papers = fetch_recent_papers()
        
        if papers:
            output_file = save_papers_batch(papers)
            
            # Output for GitHub Actions
            if os.getenv('GITHUB_OUTPUT'):
                with open(os.getenv('GITHUB_OUTPUT'), 'a') as f:
                    f.write(f"papers_found={len(papers)}\n")
                    f.write(f"papers_file={output_file}\n")
            
            print("\n" + "=" * 70)
            print(f"✅ Successfully fetched {len(papers)} new papers!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("ℹ️  No new papers to process")
            print("=" * 70)
            
            if os.getenv('GITHUB_OUTPUT'):
                with open(os.getenv('GITHUB_OUTPUT'), 'a') as f:
                    f.write("papers_found=0\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error fetching papers: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
