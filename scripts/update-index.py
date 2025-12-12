#!/usr/bin/env python3
"""
arXiv Insights - Index Updater
Updates index files and generates data for the website.
"""

import os
import json
import yaml
from datetime import datetime, date
from pathlib import Path
from collections import defaultdict

# Add this class before the main functions
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

def parse_frontmatter(content):
    """Extract YAML frontmatter from markdown."""
    if not content.startswith('---'):
        return {}, content
    
    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}, content
    
    try:
        frontmatter = yaml.safe_load(parts[1])
        return frontmatter, parts[2]
    except:
        return {}, content

def collect_papers():
    """Collect all paper files and their metadata."""
    papers = []
    papers_dir = Path('papers')
    
    for md_file in papers_dir.rglob('*.md'):
        if md_file.name in ['README.md', 'index.md']:
            continue
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        frontmatter, body = parse_frontmatter(content)
        
        if frontmatter:
            # Extract abstract from body
            lines = body.strip().split('\n')
            abstract = ''
            in_abstract = False
            for line in lines:
                if '## Abstract' in line:
                    in_abstract = True
                    continue
                if in_abstract:
                    if line.startswith('##'):
                        break
                    if line.strip():
                        abstract = line.strip()[:300]
                        break
            
            papers.append({
                'arxiv_id': frontmatter.get('arxiv_id', ''),
                'title': frontmatter.get('title', ''),
                'authors': frontmatter.get('authors', []),
                'date': frontmatter.get('publication_date', ''),
                'field': frontmatter.get('field', 'other'),
                'tags': frontmatter.get('tags', []),
                'url': frontmatter.get('url', ''),
                'abstract': abstract,
                'file': str(md_file.relative_to('papers'))
            })
    
    papers.sort(key=lambda x: x['date'], reverse=True)
    return papers

def generate_main_index(papers):
    """Generate main index.md file."""
    unique_authors = set()
    for paper in papers:
        unique_authors.update(paper['authors'])
    
    content = f"""# arXiv Papers Index

*Automatically generated index of all analyzed papers*

## Statistics

- **Total Papers**: {len(papers)}
- **Unique Authors**: {len(unique_authors)}
- **Research Fields**: {len(set(p['field'] for p in papers))}
- **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Latest Papers

"""
    
    for paper in papers[:20]:
        authors_str = ', '.join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += ' et al.'
        
        content += f"""### [{paper['title']}](/{paper['file']})
**{authors_str}** · arXiv:{paper['arxiv_id']} · {paper['date']} · `{paper['field']}`

{paper['abstract']}...

"""
    
    # By field
    content += "\n## By Field\n\n"
    by_field = defaultdict(list)
    for paper in papers:
        by_field[paper['field']].append(paper)
    
    for field, items in sorted(by_field.items()):
        content += f"\n### {field.upper()} ({len(items)} papers)\n\n"
        for paper in items[:10]:
            content += f"- [{paper['title']}](/{paper['file']}) - arXiv:{paper['arxiv_id']}\n"
    
    content += "\n---\n\n*This index is automatically updated by the automation system*\n"
    
    with open('papers/index.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated papers/index.md")

def generate_data_json(papers):
    """Generate JSON data file for website."""
    data_dir = Path('docs/data')
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / 'papers.json', 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, cls=DateTimeEncoder)
    
    print("✅ Generated docs/data/papers.json")

def main():
    """Main index update function."""
    print("🔄 Collecting papers...")
    papers = collect_papers()
    print(f"📊 Found {len(papers)} papers")
    
    print("📝 Generating indexes...")
    generate_main_index(papers)
    generate_data_json(papers)
    
    print("✅ Index update complete")

if __name__ == '__main__':
    main()
