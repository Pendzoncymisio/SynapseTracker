# Synapse Tracker

A custom BitTorrent tracker with vector similarity search for the Synapse Protocol.

## Features

- **Standard BitTorrent Protocol**: announce, scrape, peer discovery
- **Vector Search**: Find memory shards by semantic similarity
- **Metadata Storage**: Extended metadata for OpenClaw memory shards
- **REST API**: HTTP endpoints for search and discovery
- **UDP Support**: Fast announce/scrape via UDP

## Architecture

```
┌─────────────────────────────────────────────┐
│          Synapse Tracker                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐    ┌─────────────────┐  │
│  │  BitTorrent  │    │  Vector Search  │  │
│  │   Protocol   │    │     Engine      │  │
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
```

## Components

### 1. Tracker Server (`tracker_server.py`)
- HTTP/UDP announce and scrape
- Peer management and statistics
- API endpoints

### 2. Vector Search Engine (`vector_search.py`)
- FAISS index for fast similarity search
- Embedding storage and retrieval
- Query processing

### 3. Metadata Store (`metadata_store.py`)
- SQLite database for shard metadata
- Extended fields (model, dimensions, tags)
- Full-text search support

### 4. API Server (`api_server.py`)
- REST endpoints for search
- Magnet link generation
- Statistics and monitoring

## Installation

```bash
cd synapse-tracker

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_tracker.py

# Run tracker
python run_tracker.py
```

## Configuration

Edit `config.yaml`:

```yaml
tracker:
  host: "0.0.0.0"
  http_port: 6881
  udp_port: 6969
  
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

**Vector Search:**
```
POST /api/search
{
  "query": "Kubernetes deployment tutorial",
  "limit": 10,
  "filters": {
    "model": "nomic-embed-text-v1.5",
    "tags": ["devops"]
  }
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
```

**Query by Embedding:**
```
POST /api/search/embedding
{
  "embedding": [0.1, 0.2, ...],  // 768-dim vector
  "limit": 10
}
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
response = requests.post("http://tracker.example.com:6881/api/register", json={
    "info_hash": "abcdef1234567890",
    "display_name": "React Hooks Guide",
    "model": "nomic-embed-text-v1.5",
    "dimension": 768,
    "tags": ["react", "javascript", "hooks"],
})

# Search for similar shards
response = requests.post("http://tracker.example.com:6881/api/search", json={
    "query": "How to use React hooks",
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
    "http://tracker.example.com:6881/announce",
    "udp://tracker.example.com:6969/announce"
  ],
  "search_endpoint": "http://tracker.example.com:6881/api/search"
}
```

## Deployment

### Docker

```bash
docker build -t synapse-tracker .
docker run -p 6881:6881 -p 6969:6969/udp synapse-tracker
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
ExecStart=/usr/bin/python3 run_tracker.py
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

## License

Part of HiveBrain project - see main repository for license.
# SynapseTracker
