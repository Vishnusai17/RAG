# GraphRAG: Investigative Document Analysis Tool

A **Knowledge Graph-powered RAG system** for analyzing documents and discovering entity relationships. Built for investigative journalism and legal discovery use cases.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-purple.svg)

## Features

- 🔍 **Entity Extraction** - Hybrid spaCy + LLM extraction of people, organizations, dates, locations
- 🕸️ **Knowledge Graph** - Neo4j-powered graph for relationship discovery
- 💬 **Natural Language Queries** - Ask questions like "How is Person A connected to Company B?"
- 📊 **Graph Visualization** - Interactive vis.js graph explorer
- 📄 **Multi-format Ingestion** - PDF, TXT, EML (email), HTML support
- 🔗 **Hybrid Search** - Combines graph traversal with vector similarity

## Quick Start

### 1. Prerequisites

- Python 3.9+
- Neo4j (local, Docker, or Aura Cloud)
- OpenAI API key

### 2. Installation

```bash
# Clone and install
cd Graph-Rag
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
# NEO4J_URI=bolt://localhost:7687
# NEO4J_PASSWORD=your_password
# OPENAI_API_KEY=sk-your-key
```

### 4. Start Neo4j

**Option A: Docker**
```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

**Option B: Neo4j Desktop** - Download from [neo4j.com](https://neo4j.com/download/)

**Option C: Neo4j Aura** - Free cloud tier at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)

### 5. Run the Application

```bash
# Start the API server
uvicorn api.main:app --reload

# Open http://localhost:8000 in your browser
```

## Usage

### Web Interface

1. Open `http://localhost:8000`
2. Upload documents via drag-and-drop
3. Query relationships: *"Who sent emails to Ken Lay?"*
4. Explore the interactive graph visualization

### CLI Tools

**Ingest Documents:**
```bash
python scripts/ingest.py --path ./sample_data/enron_sample/ --verbose
```

**Query the Graph:**
```bash
# Natural language query
python scripts/query.py --query "How is Andrew Fastow connected to Merrill Lynch?"

# Find path between entities
python scripts/query.py --path "Kenneth Lay" "Arthur Andersen"

# Interactive mode
python scripts/query.py --interactive
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Natural language query |
| `/api/ingest` | POST | Upload documents |
| `/api/entities` | GET | List all entities |
| `/api/entities/{name}` | GET | Entity details |
| `/api/entities/path` | POST | Find paths between entities |
| `/api/graph/data` | GET | Graph data for visualization |
| `/api/graph/stats` | GET | Graph statistics |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Document      │────▶│    Extraction   │────▶│   Knowledge     │
│   Ingestion     │     │  spaCy + LLM    │     │     Graph       │
│  PDF/TXT/EML    │     │                 │     │    (Neo4j)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│   Vector Store  │◀────│    Chunking     │◀─────────────┘
│    (FAISS)      │     │                 │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Query Engine                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Graph Query  │  │Vector Search │  │  LLM Answer Synth    │  │
│  │ (Cypher)     │  │ (Semantic)   │  │  (GPT-4)             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Graph-Rag/
├── api/                    # FastAPI endpoints
├── extraction/             # Entity & relationship extraction
├── graph/                  # Neo4j client & graph builder
├── ingestion/              # Document loading & chunking
├── rag/                    # Query engine & vector store
├── frontend/               # Web interface
├── scripts/                # CLI tools
├── sample_data/            # Example documents
├── config.py               # Configuration
└── requirements.txt        # Dependencies
```

## Example Queries

- *"Who sent emails to Ken Lay?"*
- *"How is Andrew Fastow connected to JPMorgan?"*
- *"What organizations are mentioned in emails from October 2001?"*
- *"Find all people who work for Enron"*
- *"What topics were discussed between Jeff Skilling and Rebecca Mark?"*

## License

MIT License
