#!/usr/bin/env python3
"""
Test script to verify arXiv API connectivity and functionality.
"""

import arxiv
import sys
from datetime import datetime, timedelta

def test_basic_search():
    """Test basic arXiv search functionality."""
    print("=" * 70)
    print("🧪 Testing arXiv API Connection")
    print("=" * 70)
    print()
    
    try:
        # Test 1: Simple search
        print("Test 1: Simple search for recent AI papers...")
        client = arxiv.Client()
        search = arxiv.Search(
            query='cat:cs.AI',
            max_results=3,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(client.results(search))
        print(f"✅ Found {len(results)} papers")
        
        if results:
            print("\n📄 Sample paper:")
            paper = results[0]
            print(f"   Title: {paper.title}")
            print(f"   Authors: {', '.join([a.name for a in paper.authors[:3]])}...")
            print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
            print(f"   arXiv ID: {paper.entry_id.split('/')[-1]}")
            print(f"   Category: {paper.primary_category}")
        
        # Test 2: Keyword search
        print("\n" + "-" * 70)
        print("Test 2: Keyword search for 'large language model'...")
        search = arxiv.Search(
            query='all:"large language model"',
            max_results=3,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(client.results(search))
        print(f"✅ Found {len(results)} papers")
        
        # Test 3: Combined search
        print("\n" + "-" * 70)
        print("Test 3: Combined category and keyword search...")
        search = arxiv.Search(
            query='(cat:cs.AI OR cat:cs.LG) AND all:"transformer"',
            max_results=3,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(client.results(search))
        print(f"✅ Found {len(results)} papers")
        
        # Test 4: Recent papers (last 7 days)
        print("\n" + "-" * 70)
        print("Test 4: Papers from last 7 days...")
        cutoff = datetime.now() - timedelta(days=7)
        search = arxiv.Search(
            query='cat:cs.AI',
            max_results=20,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        recent_count = 0
        for result in client.results(search):
            if result.published >= cutoff:
                recent_count += 1
            else:
                break
        
        print(f"✅ Found {recent_count} papers from last 7 days")
        
        # Test 5: Paper details extraction
        print("\n" + "-" * 70)
        print("Test 5: Extracting paper details...")
        if results:
            paper = results[0]
            print(f"   Title: {paper.title}")
            print(f"   arXiv ID: {paper.entry_id.split('/')[-1].split('v')[0]}")
            print(f"   Authors ({len(paper.authors)}): {', '.join([a.name for a in paper.authors[:3]])}...")
            print(f"   Abstract length: {len(paper.summary)} chars")
            print(f"   Categories: {', '.join(paper.categories)}")
            print(f"   PDF URL: {paper.pdf_url}")
            print(f"   Published: {paper.published.strftime('%Y-%m-%d %H:%M:%S')}")
            if paper.comment:
                print(f"   Comment: {paper.comment[:50]}...")
        
        print("\n" + "=" * 70)
        print("✅ All tests passed! arXiv API is working correctly.")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def test_categories():
    """Test different arXiv categories."""
    print("\n" + "=" * 70)
    print("🏷️  Testing Different Categories")
    print("=" * 70)
    print()
    
    categories = ['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE']
    client = arxiv.Client()
    
    for cat in categories:
        try:
            search = arxiv.Search(
                query=f'cat:{cat}',
                max_results=2,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            results = list(client.results(search))
            print(f"✅ {cat:8s} - {len(results)} papers found")
        except Exception as e:
            print(f"❌ {cat:8s} - Error: {e}")

if __name__ == '__main__':
    print()
    exit_code = test_basic_search()
    test_categories()
    print()
    sys.exit(exit_code)
