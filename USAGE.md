# arXiv Insights - Usage Guide

## 🚀 Quick Start

Your repository is now fully configured to automatically fetch and process papers from arXiv!

## 🔄 Automatic Operation

The system runs automatically every 8 hours via GitHub Actions. No manual intervention needed!

### What Happens Automatically:
1. **Fetches** recent papers from arXiv matching your criteria
2. **Processes** paper metadata, abstracts, and details
3. **Creates** structured markdown files for each paper
4. **Updates** indexes and navigation
5. **Commits** changes to the repository
6. **Deploys** to GitHub Pages

## 🎯 Configuration

### Paper Selection Criteria

Edit `scripts/fetch-arxiv-papers.py` to customize what papers to fetch:

```python
# Categories to monitor
ARXIV_CATEGORIES = [
    'cs.AI',   # Artificial Intelligence
    'cs.LG',   # Machine Learning
    'cs.CL',   # Computation and Language
    'cs.CV',   # Computer Vision
    'cs.NE',   # Neural and Evolutionary Computing
]

# Keywords to filter
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
```

### Adjust Fetch Parameters

You can control how many papers to fetch:
- `MAX_PAPERS_PER_RUN`: Default is 5 papers per run
- `DAYS_BACK`: Default looks back 2 days

## 🧪 Testing

### Test arXiv API Connection

Run the test script to verify everything works:

```bash
cd scripts
python test-arxiv-api.py
```

This will:
- ✅ Test API connectivity
- ✅ Search for papers in different categories
- ✅ Verify data extraction
- ✅ Check recent paper availability

### Manual Paper Fetch

You can manually trigger a paper fetch:

```bash
# Fetch papers
python scripts/fetch-arxiv-papers.py

# Process the fetched papers
python scripts/process-batch.py

# Update indexes
python scripts/update-index.py
```

### Environment Variables

Control fetch behavior with environment variables:

```bash
# Fetch up to 10 papers from last 7 days
MAX_PAPERS=10 DAYS_BACK=7 python scripts/fetch-arxiv-papers.py
```

## 🎮 Manual Triggers

### Trigger Workflow Manually

Go to Actions tab → "Update arXiv Papers" → "Run workflow"

**Options:**
- Leave `content` empty to fetch from arXiv
- Set `max_papers` (default: 5)
- Set `days_back` (default: 2)

### Process Single Paper

If you want to add a specific paper:

```bash
python scripts/process-paper.py
```

Set the `PAPER_DATA` environment variable with JSON data:

```json
{
  "title": "Paper Title",
  "arxiv_id": "2301.12345",
  "authors": ["Author 1", "Author 2"],
  "date": "2025-01-15",
  "field": "cs-ai",
  "abstract": "Paper abstract...",
  "key_findings": ["Finding 1", "Finding 2"],
  "methodology": {
    "description": "Method description",
    "dataset": "Dataset name",
    "techniques": ["Technique 1"]
  },
  "applications": ["Application 1"],
  "tags": ["cs.AI", "machine-learning"]
}
```

## 📊 Output Structure

Papers are organized by field:

```
papers/
├── cs/
│   ├── ai/
│   │   └── 2301.12345-paper-title.md
│   ├── ml/
│   └── cv/
├── math/
├── physics/
└── bio/
```

Each paper includes:
- **Metadata**: arXiv ID, authors, dates, categories
- **Abstract**: Original paper abstract
- **Key Findings**: Main discoveries (auto-extracted)
- **Methodology**: Research approach
- **Applications**: Practical uses
- **Tags**: arXiv categories and custom tags

## 🔍 Monitoring

### Check Workflow Runs

1. Go to **Actions** tab in your repository
2. Click on "Update arXiv Papers" workflow
3. View recent runs and their logs

### View Papers

- **Repository**: Browse `papers/` directory
- **GitHub Pages**: Visit your deployed site (if configured)

## ⚙️ Advanced Configuration

### Change Schedule

Edit `.github/workflows/update-papers.yml`:

```yaml
schedule:
  - cron: '0 */8 * * *'  # Every 8 hours
  # Examples:
  # - cron: '0 */4 * * *'  # Every 4 hours
  # - cron: '0 9 * * *'    # Daily at 9 AM
  # - cron: '0 9 * * 1'    # Weekly on Monday at 9 AM
```

### Add More Categories

Edit `scripts/fetch-arxiv-papers.py` and add to `ARXIV_CATEGORIES`:

```python
ARXIV_CATEGORIES = [
    'cs.AI',
    'cs.LG',
    'stat.ML',      # Statistics - Machine Learning
    'math.OC',      # Optimization and Control
    'eess.SP',      # Signal Processing
    # See https://arxiv.org/category_taxonomy for all categories
]
```

### Customize Paper Format

Edit `scripts/process-paper.py` to modify the markdown template.

## 🐛 Troubleshooting

### No Papers Being Fetched

1. Check if papers match your criteria (categories + keywords)
2. Increase `DAYS_BACK` to look further back
3. Reduce keyword specificity
4. Check workflow logs for errors

### Workflow Failing

1. Check Actions tab for error messages
2. Verify arXiv API is accessible
3. Check Python dependencies are installing correctly

### Duplicate Papers

The system automatically checks for duplicates by arXiv ID. If you see duplicates:
1. Check the deduplication logic in `fetch-arxiv-papers.py`
2. Manually remove duplicate files from `papers/` directory

## 📚 Resources

- **arXiv API Documentation**: https://info.arxiv.org/help/api/index.html
- **arXiv Category Taxonomy**: https://arxiv.org/category_taxonomy
- **Python arxiv Package**: https://github.com/lukasschwab/arxiv.py

## 🎯 Tips

1. **Start Small**: Begin with 2-3 categories and expand
2. **Refine Keywords**: Monitor what papers you get and adjust
3. **Check Regularly**: Review the automated papers weekly
4. **Customize Templates**: Modify paper format to match your needs
5. **Use Tags**: Leverage tags for better organization

## 🔐 Privacy Note

This is a private repository. All content and automation runs are private to you.

---

*Last Updated: December 11, 2025*
