// arXiv Insights - Search and Interaction Functionality

let papersData = [];

// Search functionality
function searchPapers() {
    const query = document.getElementById('searchInput').value.toLowerCase();
    
    if (query.length < 2) {
        alert('Please enter at least 2 characters to search');
        return;
    }
    
    const results = papersData.filter(paper => {
        return paper.title.toLowerCase().includes(query) ||
               paper.authors.some(author => author.toLowerCase().includes(query)) ||
               paper.abstract.toLowerCase().includes(query) ||
               paper.tags.some(tag => tag.toLowerCase().includes(query)) ||
               paper.arxiv_id.toLowerCase().includes(query);
    });
    
    displayResults(results, `Search results for "${query}"`);
}

// Filter by field
function filterByField(field) {
    const results = papersData.filter(paper => 
        paper.field.toLowerCase() === field.toLowerCase()
    );
    
    displayResults(results, `${field.toUpperCase()} Papers`);
}

// Display results
function displayResults(results, title) {
    const latestContent = document.getElementById('latest-content');
    const latestSection = document.getElementById('latest');
    
    latestSection.scrollIntoView({ behavior: 'smooth' });
    latestSection.querySelector('h2').textContent = title;
    
    if (results.length === 0) {
        latestContent.innerHTML = '<p class="placeholder">No results found.</p>';
        return;
    }
    
    latestContent.innerHTML = results.map(paper => `
        <div class="paper-card" onclick="openPaper('${paper.arxiv_id}')">
            <div class="card-header">
                <span class="card-date">${formatDate(paper.date)}</span>
                <span class="card-category">${paper.field}</span>
            </div>
            <h3 class="card-title">${paper.title}</h3>
            <p class="card-authors">by ${formatAuthors(paper.authors)}</p>
            <p class="card-summary">${paper.abstract.substring(0, 150)}...</p>
            <div class="card-tags">
                <span class="tag">arXiv:${paper.arxiv_id}</span>
                ${paper.tags.slice(0, 3).map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
        </div>
    `).join('');
}

// Open paper details
function openPaper(arxivId) {
    window.open(`https://arxiv.org/abs/${arxivId}`, '_blank');
}

// Format date helper
function formatDate(dateString) {
    const date = new Date(dateString);
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

// Format authors
function formatAuthors(authors) {
    if (authors.length <= 2) {
        return authors.join(' and ');
    }
    return `${authors[0]} et al.`;
}

// Load data function
function loadPapersData(data) {
    papersData = data;
    updateStats();
    renderLatestPapers();
}

// Update statistics
function updateStats() {
    const uniqueAuthors = new Set();
    papersData.forEach(paper => {
        paper.authors.forEach(author => uniqueAuthors.add(author));
    });
    
    const stats = {
        total: papersData.length,
        authors: uniqueAuthors.size,
        fields: new Set(papersData.map(p => p.field)).size,
        lastUpdate: papersData.length > 0 ? 
            formatDate(papersData[0].date) : 'N/A'
    };
    
    document.querySelectorAll('.stat-value')[0].textContent = stats.total;
    document.querySelectorAll('.stat-value')[1].textContent = stats.authors;
    document.querySelectorAll('.stat-value')[2].textContent = stats.fields;
    document.querySelectorAll('.stat-value')[3].textContent = stats.lastUpdate;
}

// Render latest papers
function renderLatestPapers() {
    const latest = papersData.slice(0, 12);
    displayResults(latest, 'Latest Papers');
}

// Update field counts
function updateFieldCounts() {
    const fields = {};
    papersData.forEach(paper => {
        fields[paper.field] = (fields[paper.field] || 0) + 1;
    });
    
    Object.keys(fields).forEach(field => {
        const fieldCard = document.querySelector(
            `.category-card[onclick*="${field}"] .count`
        );
        if (fieldCard) {
            fieldCard.textContent = `${fields[field]} papers`;
        }
    });
}

// Handle search on Enter key
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchPapers();
            }
        });
    }
    
    document.getElementById('lastUpdate').textContent = 
        new Date().toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { loadPapersData, searchPapers, filterByField };
}