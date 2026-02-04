
// GraphRAG Custom Frontend - App Logic

// Configuration
const API_BASE_URL = "/api";
let network = null;
let networkData = { nodes: new vis.DataSet([]), edges: new vis.DataSet([]) };

// DOM Elements
const elements = {
    queryInput: document.getElementById('queryInput'),
    queryBtn: document.getElementById('queryBtn'),
    answerContent: document.getElementById('answerContent'),
    queryType: document.getElementById('queryType'),
    sourcesSection: document.getElementById('sourcesSection'),
    sourcesList: document.getElementById('sourcesList'),
    graphContainer: document.getElementById('graphContainer'),
    refreshGraphBtn: document.getElementById('refreshGraphBtn'),
    fullscreenBtn: document.getElementById('fullscreenBtn'),
    fileInput: document.getElementById('fileInput'),
    dropzone: document.getElementById('dropzone'),
    uploadStatus: document.getElementById('uploadStatus'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    stats: {
        nodes: document.getElementById('nodeCount'),
        rels: document.getElementById('relCount'),
        docs: document.getElementById('docCount')
    }
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initGraph();
    loadGraphData();
    updateStats();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Query
    elements.queryBtn.addEventListener('click', handleQuery);
    elements.queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleQuery();
    });

    // Graph Controls
    elements.refreshGraphBtn.addEventListener('click', () => {
        loadGraphData();
        updateStats();
    });

    // File Upload
    elements.dropzone.addEventListener('click', () => elements.fileInput.click());
    elements.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropzone.classList.add('drag-active');
    });
    elements.dropzone.addEventListener('dragleave', () => {
        elements.dropzone.classList.remove('drag-active');
    });
    elements.dropzone.addEventListener('drop', handleFileDrop);
    elements.fileInput.addEventListener('change', handleFileSelect);
}

// -------------------------------------------------------------------------
// QUERY LOGIC
// -------------------------------------------------------------------------

async function handleQuery() {
    const question = elements.queryInput.value.trim();
    if (!question) return;

    showLoading(true);

    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        displayAnswer(data);
    } catch (error) {
        console.error('Query error:', error);
        elements.answerContent.innerHTML = `<div class="error-msg">Error: ${error.message}</div>`;
    } finally {
        showLoading(false);
    }
}

function displayAnswer(data) {
    // Answer text - handle both string and object
    const answerText = typeof data.answer === 'string' ? data.answer : JSON.stringify(data.answer);
    elements.answerContent.innerHTML = marked.parse(answerText);

    // Badge
    const queryType = data.query_type || 'unknown';
    elements.queryType.textContent = queryType.toUpperCase();
    elements.queryType.className = `query-badge ${queryType}`;

    // Sources - robust handling
    if (data.sources && data.sources.length > 0) {
        elements.sourcesSection.classList.remove('hidden');
        elements.sourcesList.innerHTML = data.sources.map(s => {
            // Handle different source formats
            let sourceName = 'Unknown';
            let content = '';
            let score = null;

            if (typeof s === 'string') {
                content = s;
            } else if (typeof s === 'object' && s !== null) {
                sourceName = s.source || s.metadata?.source || s.name || 'Document';
                content = s.content || s.text || s.page_content || JSON.stringify(s).slice(0, 200);
                score = s.score;
            }

            return `
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-doc">${sourceName}</span>
                        ${score ? `<span class="source-score">${Math.round(score * 100)}% Match</span>` : ''}
                    </div>
                    <div class="source-preview">${content}</div>
                </div>
            `;
        }).join('');
    } else {
        elements.sourcesSection.classList.add('hidden');
    }
}

// -------------------------------------------------------------------------
// GRAPH VISUALIZATION (Vis.js)
// -------------------------------------------------------------------------

function initGraph() {
    const options = {
        nodes: {
            shape: 'dot',
            size: 16,
            font: {
                size: 14,
                color: '#e0e0e0',
                face: 'Inter'
            },
            borderWidth: 2,
            shadow: true
        },
        edges: {
            width: 1,
            color: { color: '#404040', highlight: '#00D4FF' },
            smooth: { type: 'continuous' },
            arrows: { to: { enabled: true, scaleFactor: 0.5 } }
        },
        physics: {
            stabilization: false,
            barnesHut: {
                gravitationalConstant: -2000,
                centralGravity: 0.3,
                springLength: 95
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true
        }
    };

    network = new vis.Network(elements.graphContainer, networkData, options);

    // Click event
    network.on("click", function (params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = networkData.nodes.get(nodeId);
            console.log('Clicked node:', node);
            // Future: Show node details sidebar
        }
    });
}

async function loadGraphData() {
    try {
        const response = await fetch(`${API_BASE_URL}/graph/data?limit=500`);
        const data = await response.json();

        // Transform for Vis.js
        const nodes = data.nodes.map(n => ({
            id: n.id,
            label: n.group === 'Document' ? '📄' : n.label, // Use icon for docs
            group: n.group,
            color: getNodeColor(n.group),
            title: `Type: ${n.group}\nName: ${n.label}`
        }));

        const edges = data.edges.map(e => ({
            from: e.from,
            to: e.to,
            label: e.label,
            title: e.label
        }));

        networkData.nodes.clear();
        networkData.edges.clear();
        networkData.nodes.add(nodes);
        networkData.edges.add(edges);

        network.fit();

    } catch (error) {
        console.error('Failed to load graph:', error);
    }
}

function getNodeColor(group) {
    const colors = {
        'Person': '#FF5F56',      // Red
        'Organization': '#00D4FF', // Cyan
        'Location': '#2ecc71',    // Green
        'Document': '#f1c40f',    // Yellow
        'Date': '#9b59b6',        // Purple
        'Unknown': '#95a5a6'      // Grey
    };
    return colors[group] || colors['Unknown'];
}

// -------------------------------------------------------------------------
// FILE UPLOAD
// -------------------------------------------------------------------------

async function handleFileDrop(e) {
    e.preventDefault();
    elements.dropzone.classList.remove('drag-active');
    handleFiles(e.dataTransfer.files);
}

function handleFileSelect(e) {
    handleFiles(e.target.files);
}

async function handleFiles(files) {
    if (files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    setStatus('Uploading...', 'info');

    try {
        const response = await fetch(`${API_BASE_URL}/ingest`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const result = await response.json();
        setStatus(`Success! Processing ${result.files.length} files in background.`, 'success');

        // Poll for updates (simplified)
        setTimeout(() => {
            loadGraphData();
            updateStats();
        }, 5000);

    } catch (error) {
        setStatus(`Error: ${error.message}`, 'error');
    }
}

function setStatus(msg, type) {
    elements.uploadStatus.textContent = msg;
    elements.uploadStatus.className = `upload-status ${type}`;
    elements.uploadStatus.classList.remove('hidden');
}

// -------------------------------------------------------------------------
// UTILS
// -------------------------------------------------------------------------

async function updateStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/graph/stats`);
        const stats = await response.json();

        elements.stats.nodes.textContent = stats.node_count || 0;
        elements.stats.rels.textContent = stats.relationship_count || 0;
        // Mock doc count since API doesn't return it directly in stats yet
        elements.stats.docs.textContent = stats.labels?.Document || 0;
    } catch (e) {
        console.warn('Stats update failed', e);
    }
}

function showLoading(show) {
    if (show) elements.loadingOverlay.classList.remove('hidden');
    else elements.loadingOverlay.classList.add('hidden');
}

function setQuery(text) {
    elements.queryInput.value = text;
    handleQuery();
}
