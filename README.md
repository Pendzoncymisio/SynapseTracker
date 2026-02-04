# Synapse Tracker

A custom BitTorrent tracker with vector similarity search for the Synapse Protocol.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tracker
./tracker

# Run with options
./tracker --port 8080
./tracker --debug
./tracker --help
```

## Features

- **Standard BitTorrent Protocol**: announce, scrape, peer discovery
- **Vector Search**: Find memory shards by semantic similarity
- **Metadata Storage**: Extended metadata for OpenClaw memory shards
- **REST API**: HTTP endpoints for search and discovery
- **UDP Support**: Fast announce/scrape via UDP

## Architecture

**Lightweight Design**: No ML models or embeddings generation - clients handle that!

```
┌─────────────────────────────────────────────┐
│          Synapse Tracker                    │
│       (No ML Dependencies)                  │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐    ┌─────────────────┐  │
│  │  BitTorrent  │    │  Vector Search  │  │
│  │   Protocol   │    │  (FAISS Index)  │  │
│  └──────────────┘    └─────────────────┘  │
│         │                     │            │
│         ├─────────────────────┤            │
│         ▼                     ▼            │
│  ┌──────────────────────────────────────┐ │
│  │       Metadata Database              │ │
│  │  (SQLite + Vector Index)             │ │
│  └──────────────────────────────────────┘ │
│                                             │
└─────────────────────────────────────────────┘

Clients generate embeddings → send vectors to tracker → get results
```

## Components

### 1. Tracker Server (`tracker_server.py`)
- HTTP/UDP announce and scrape
- Peer management and statistics
- API endpoints

### 2. Vector Search Engine (`vector_search.py`)
- FAISS index for fast similarity search
- Pre-computed embedding storage and retrieval
- Vector similarity queries (no text processing or ML models)

### 3. Metadata Store (`metadata_store.py`)
- SQLite database for shard metadata
- Extended fields (model, dimensions, tags)
- Full-text search support

### 4. API Server (`api_server.py`)
- REST endpoints for search
- Magnet link generation
- Statistics and monitoring

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
cd SynapseTracker

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Run tracker
uv run tracker

# Or run directly
./tracker

# With options
./tracker --port 8080 --debug
./tracker --help
```

### Using pip (Alternative)

```bash
cd SynapseTracker

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tracker
./tracker
```

## Configuration

Edit `config.yaml`:

```yaml
tracker:
  host: "0.0.0.0"
  http_port: 80
  
vector_search:
  index_type: "faiss"
  dimension: 768  # Nomic Embed
  similarity_metric: "cosine"
  
database:
  path: "./tracker.db"
  
server:
  max_peers_returned: 50
  announce_interval: 1800  # 30 minutes
```

## API Endpoints

### BitTorrent Protocol

**HTTP Announce:**
```
GET /announce?info_hash={hash}&peer_id={id}&port={port}&uploaded={bytes}&downloaded={bytes}&left={bytes}
```

**HTTP Scrape:**
```
GET /scrape?info_hash={hash1}&info_hash={hash2}
```

### Synapse Extensions

**Register Shard:**
```
POST /api/register
{
  "info_hash": "...",
  "display_name": "...",
  "model": "nomic-embed-text-v1.5",
  "dimension": 768,
  "tags": ["python", "async"],
  "embedding": [0.1, 0.2, ...]  // Optional: for search
}
```

**Vector Search (DEPRECATED - use /api/search/embedding):**
```
POST /api/search

⚠️  This endpoint is deprecated and returns HTTP 410 Gone.
    Clients must generate embeddings and use /api/search/embedding instead.
    See Synapse client for transparent embedding generation.
```

**Query by Embedding (PRIMARY SEARCH METHOD):**
```
POST /api/search/embedding
{
  "embedding": [0.1, 0.2, ...],  // 768-dim vector (Nomic Embed)
  "limit": 10
}

Response:
{
  "results": [
    {
      "info_hash": "...",
      "display_name": "...",
      "similarity": 0.87,
      "magnet_link": "magnet:?xt=...",
      "seeders": 5,
      "leechers": 2
    }
  ]
}

Note: Clients (like Synapse) generate embeddings from text queries,
      then send the vector to this endpoint for similarity search.
```

**Get Statistics:**
```
GET /api/stats

Response:
{
  "total_shards": 150,
  "total_peers": 45,
  "active_torrents": 120,
  "total_announced": 50000
}
```

## Usage Example

### From Python Client

```python
import requests

# Register a new shard
response = requests.post("http://tracker.example.com/api/register", json={
    "info_hash": "abcdef1234567890",
    "display_name": "React Hooks Guide",
    "model": "nomic-embed-text-v1.5",
    "dimension": 768,
    "tags": ["react", "javascript", "hooks"],
})

# Search for similar shards (using pre-computed embedding)
# In practice, use the Synapse client which handles embeddings automatically
from embeddings import create_embedder  # From Synapse client

embedder = create_embedder(use_onnx=True)
query_embedding = embedder.encode("How to use React hooks").tolist()

response = requests.post("http://tracker.example.com/api/search/embedding", json={
    "embedding": query_embedding,
    "limit": 5,
})

results = response.json()["results"]
for result in results:
    print(f"{result['display_name']}: {result['similarity']:.2f}")
```

### Integration with Synapse Protocol

The synapse-protocol will automatically use this tracker when configured:

```json
{
  "default_trackers": [
    "http://tracker.example.com/announce"
  ],
  "search_endpoint": "http://tracker.example.com/api/search"
}
```

## Development

### Project Structure

```
SynapseTracker/
├── tracker              # Main executable
├── src/                 # Source code
│   ├── __init__.py
│   ├── tracker_server.py
│   ├── metadata_store.py
│   └── vector_search.py
├── config.yaml          # Configuration
├── pyproject.toml       # Dependencies
└── README.md
```

### Code Formatting

```bash
# Format code with black
uv run black src/
```

## Deployment

### Production with uv

```bash
# Install production dependencies
uv sync --no-dev

# Run with gunicorn (production WSGI server)
uv run gunicorn -w 4 -b 0.0.0.0:80 'src.tracker_server:app'
```

### Docker

```bash
docker build -t synapse-tracker .
docker run -p 80:80 synapse-tracker
```

### Systemd Service

```ini
[Unit]
Description=Synapse Tracker
After=network.target

[Service]
Type=simple
User=synapse
WorkingDirectory=/opt/synapse-tracker
ExecStart=/opt/synapse-tracker/tracker
Restart=always

[Install]
WantedBy=multi-user.target
```

## Performance

- **Announce throughput**: 10,000+ req/s (UDP)
- **Search latency**: <50ms for 100k shards
- **Memory usage**: ~200MB + (shards × 3KB)
- **Storage**: ~10KB per shard (metadata + embedding)

## Security

- Rate limiting per IP
- IP whitelist/blacklist support
- Request validation
- SQL injection prevention
- Optional authentication for registration

## Monitoring

Built-in Prometheus metrics:

```
# Tracker metrics
synapse_tracker_announces_total
synapse_tracker_scrapes_total
synapse_tracker_active_peers
synapse_tracker_active_torrents

# Search metrics
synapse_search_queries_total
synapse_search_latency_seconds
synapse_vector_index_size
```

## Quick Reference

### Common Commands

```bash
# Setup
uv sync                          # Install dependencies

# Development
./tracker                        # Start server
./tracker --debug                # Start with debug mode
./tracker --port 8080            # Custom port
uv run black src/                # Format code

# Production
uv sync --no-dev                 # Install only prod dependencies
uv run gunicorn -w 4 src.tracker_server:app  # Run with gunicorn
```

### Environment Files

- **tracker** - Main executable (single entry point)
- **src/** - Source code modules
- **config.yaml** - Tracker configuration  
- **pyproject.toml** - Project dependencies (uv)
- **requirements.txt** - Legacy pip requirements
- **.python-version** - Python version (3.11)

## License

Part of HiveBrain project - see main repository for license.
# SynapseTracker
