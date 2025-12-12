#!/usr/bin/env python3
"""
arXiv Insights - Batch Processor
Processes multiple papers from a JSON batch file.
"""

import json
import sys
from pathlib import Path

# Import the process_paper function from the process-paper module
import importlib.util
spec = importlib.util.spec_from_file_location("process_paper", Path(__file__).parent / "process-paper.py")
process_paper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_paper_module)
process_paper = process_paper_module.process_paper

def main():
    """Process papers from batch file."""
    batch_file = Path('papers-batch.json')
    
    if not batch_file.exists():
        print("ℹ️  No batch file found")
        return 0
    
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        print(f"📄 Processing {len(papers)} papers from batch file...\n")
        
        processed = 0
        for i, paper in enumerate(papers, 1):
            try:
                print(f"[{i}/{len(papers)}] Processing {paper['arxiv_id']}...")
                process_paper(paper)
                processed += 1
            except Exception as e:
                print(f"❌ Error processing {paper.get('arxiv_id', 'unknown')}: {e}")
                continue
        
        # Remove batch file after processing
        batch_file.unlink()
        
        print(f"\n✅ Successfully processed {processed}/{len(papers)} papers")
        return 0
        
    except Exception as e:
        print(f"❌ Error processing batch: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
