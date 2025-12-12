#!/bin/bash
# Quick test script to fetch and process papers locally

set -e

echo "=================================="
echo "arXiv Insights - Local Test Run"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Test API Connection
echo -e "${BLUE}Step 1: Testing arXiv API...${NC}"
python scripts/test-arxiv-api.py
echo ""

# Step 2: Fetch Papers
echo -e "${BLUE}Step 2: Fetching papers from arXiv...${NC}"
MAX_PAPERS=3 DAYS_BACK=3 python scripts/fetch-arxiv-papers.py
echo ""

# Check if papers were found
if [ -f "papers-batch.json" ]; then
    echo -e "${GREEN}✓ Papers fetched successfully!${NC}"
    echo ""
    
    # Step 3: Process Papers
    echo -e "${BLUE}Step 3: Processing papers...${NC}"
    python scripts/process-batch.py
    echo ""
    
    # Step 4: Update Indexes
    echo -e "${BLUE}Step 4: Updating indexes...${NC}"
    python scripts/update-index.py
    echo ""
    
    echo -e "${GREEN}=================================="
    echo -e "✓ Test run completed successfully!"
    echo -e "==================================${NC}"
    echo ""
    echo "Check the 'papers/' directory for processed papers."
    echo ""
else
    echo -e "${YELLOW}No new papers found to process.${NC}"
    echo ""
    echo "This could mean:"
    echo "  - All recent papers matching criteria are already processed"
    echo "  - No papers match your keywords/categories in the time window"
    echo ""
    echo "Try:"
    echo "  - Increasing DAYS_BACK: DAYS_BACK=7 $0"
    echo "  - Adjusting keywords in scripts/fetch-arxiv-papers.py"
    echo ""
fi
