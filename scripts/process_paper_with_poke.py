#!/usr/bin/env python3
"""
Process arXiv paper with Poke API integration.

This script extends the paper processing pipeline to optionally send
processed insights to the Poke API for further analysis.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from poke_api_client import PokeAPIClient, PokeAPIError
except ImportError:
    print("Error: Could not import poke_api_client. Make sure it's in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_paper_data(paper_file: Path) -> Dict[str, Any]:
    """
    Load paper data from JSON file.
    
    Args:
        paper_file: Path to paper JSON file
    
    Returns:
        Paper data dictionary
    """
    logger.info(f"Loading paper data from {paper_file}")
    with open(paper_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_insights(paper_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract insights from paper data for Poke API.
    
    Args:
        paper_data: Raw paper data
    
    Returns:
        Formatted insights dictionary
    """
    insights = {
        'title': paper_data.get('title', 'Unknown'),
        'authors': paper_data.get('authors', []),
        'abstract': paper_data.get('abstract', ''),
        'categories': paper_data.get('categories', []),
        'published_date': paper_data.get('published', ''),
        'arxiv_id': paper_data.get('id', ''),
        'pdf_url': paper_data.get('pdf_url', ''),
    }
    
    # Add any custom analysis or key findings
    if 'summary' in paper_data:
        insights['summary'] = paper_data['summary']
    
    if 'key_findings' in paper_data:
        insights['key_findings'] = paper_data['key_findings']
    
    return insights


def send_to_poke_api(paper_insights: Dict[str, Any], arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Send paper insights to Poke API.
    
    Args:
        paper_insights: Formatted paper insights
        arxiv_id: arXiv paper ID
    
    Returns:
        API response or None if failed
    """
    try:
        client = PokeAPIClient()
        logger.info(f"Sending insights for paper {arxiv_id} to Poke API")
        
        response = client.send_paper_insights(paper_insights, arxiv_id=arxiv_id)
        
        logger.info(f"Successfully sent to Poke API. Processing ID: {response.get('id', 'N/A')}")
        
        # Show metrics
        metrics = client.get_metrics()
        logger.info(f"Client metrics - Success rate: {metrics['success_rate']:.2%}, "
                   f"Compression: {metrics['avg_compression_ratio']}")
        
        client.close()
        return response
        
    except PokeAPIError as e:
        logger.error(f"Poke API error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Process arXiv paper with optional Poke API integration'
    )
    parser.add_argument(
        'paper_file',
        type=Path,
        help='Path to paper JSON file'
    )
    parser.add_argument(
        '--send-to-poke',
        action='store_true',
        help='Send insights to Poke API'
    )
    parser.add_argument(
        '--arxiv-id',
        type=str,
        help='arXiv paper ID (extracted from file if not provided)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for processed insights'
    )
    
    args = parser.parse_args()
    
    # Check if paper file exists
    if not args.paper_file.exists():
        logger.error(f"Paper file not found: {args.paper_file}")
        sys.exit(1)
    
    try:
        # Load paper data
        paper_data = load_paper_data(args.paper_file)
        
        # Extract insights
        paper_insights = extract_insights(paper_data)
        arxiv_id = args.arxiv_id or paper_insights.get('arxiv_id', 'unknown')
        
        logger.info(f"Processed paper: {paper_insights['title']}")
        logger.info(f"arXiv ID: {arxiv_id}")
        logger.info(f"Categories: {', '.join(paper_insights['categories'])}")
        
        # Send to Poke API if requested
        poke_response = None
        if args.send_to_poke:
            # Check if API key is set
            if not os.getenv('POKE_API_KEY'):
                logger.warning("POKE_API_KEY not set. Skipping Poke API integration.")
                logger.warning("Set it with: export POKE_API_KEY='your-api-key'")
            else:
                poke_response = send_to_poke_api(paper_insights, arxiv_id)
                if poke_response:
                    paper_insights['poke_api_response'] = poke_response
        
        # Save output if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(paper_insights, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved processed insights to {args.output}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Processing Complete")
        print("=" * 60)
        print(f"Paper: {paper_insights['title']}")
        print(f"arXiv ID: {arxiv_id}")
        print(f"Authors: {len(paper_insights['authors'])}")
        print(f"Categories: {', '.join(paper_insights['categories'])}")
        if poke_response:
            print(f"✓ Sent to Poke API - ID: {poke_response.get('id', 'N/A')}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error processing paper: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
