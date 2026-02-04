"""
Synapse Tracker Server.

Implements BitTorrent announce/scrape protocol with vector search extensions.
"""

import logging
from flask import Flask, request, jsonify, make_response
from typing import Dict, Any, Optional
import struct
import socket
from urllib.parse import unquote

from metadata_store import MetadataStore
from vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)

app = Flask(__name__)
metadata_store = MetadataStore()
search_engine = VectorSearchEngine(dimension=768, index_type="flat")


# ============================================================================
# BitTorrent Protocol Endpoints
# ============================================================================

@app.route('/announce', methods=['GET'])
def announce():
    """
    Standard BitTorrent announce endpoint.
    
    Query params:
        info_hash: 20-byte SHA1 hash (URL encoded)
        peer_id: 20-byte peer identifier
        port: Port number peer is listening on
        uploaded: Total bytes uploaded
        downloaded: Total bytes downloaded
        left: Bytes left to download
        event: 'started', 'completed', 'stopped' (optional)
    """
    try:
        # Parse parameters
        info_hash = request.args.get('info_hash', '')
        peer_id = request.args.get('peer_id', '')
        port = int(request.args.get('port', 0))
        uploaded = int(request.args.get('uploaded', 0))
        downloaded = int(request.args.get('downloaded', 0))
        left = int(request.args.get('left', 0))
        event = request.args.get('event')
        
        # Get client IP
        ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        
        # Convert info_hash to hex if URL encoded
        if len(info_hash) == 20:
            info_hash = info_hash.hex()
        
        # Validate
        if not info_hash or not peer_id or not port:
            return bencode_error("Missing required parameters")
        
        # Update peer in database
        metadata_store.update_peer(
            peer_id=peer_id,
            info_hash=info_hash,
            ip=ip,
            port=port,
            uploaded=uploaded,
            downloaded=downloaded,
            left=left,
        )
        
        # Record announce
        metadata_store.record_announce(info_hash, peer_id, event)
        
        # Get peer list
        peers = metadata_store.get_peers(info_hash, max_peers=50)
        
        # Build response
        response = {
            'interval': 1800,  # 30 minutes
            'min interval': 900,  # 15 minutes
            'complete': sum(1 for p in peers if p.get('is_seeder')),
            'incomplete': sum(1 for p in peers if not p.get('is_seeder')),
            'peers': _format_peers(peers),
        }
        
        return bencode_dict(response)
    
    except Exception as e:
        logger.exception("Announce error")
        return bencode_error(str(e))


@app.route('/scrape', methods=['GET'])
def scrape():
    """
    Standard BitTorrent scrape endpoint.
    
    Query params:
        info_hash: One or more info hashes to scrape
    """
    try:
        # Get info_hashes (can be multiple)
        info_hashes = request.args.getlist('info_hash')
        
        if not info_hashes:
            return bencode_error("No info_hash provided")
        
        # Build response
        files = {}
        for info_hash in info_hashes:
            if len(info_hash) == 20:
                info_hash = info_hash.hex()
            
            shard = metadata_store.get_shard(info_hash)
            if shard:
                files[info_hash] = {
                    'complete': shard.get('seeders', 0),
                    'incomplete': shard.get('leechers', 0),
                    'downloaded': shard.get('completed', 0),
                }
        
        return bencode_dict({'files': files})
    
    except Exception as e:
        logger.exception("Scrape error")
        return bencode_error(str(e))


# ============================================================================
# Synapse Protocol Extensions
# ============================================================================

@app.route('/api/register', methods=['POST'])
def register_shard():
    """
    Register a new memory shard with metadata.
    
    Body:
        {
            "info_hash": "...",
            "display_name": "...",
            "embedding_model": "nomic-embed-text-v1.5",
            "dimension_size": 768,
            "tags": ["tag1", "tag2"],
            "description": "...",
            "entry_count": 100,
            "file_size": 1024000,
            "embedding": [0.1, 0.2, ...]  // Optional: for search
        }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ["info_hash", "display_name", "embedding_model", "dimension_size"]
        if not all(field in data for field in required):
            return jsonify({"error": f"Missing required fields: {required}"}), 400
        
        # Register in metadata store
        success = metadata_store.register_shard(data)
        
        if not success:
            return jsonify({"error": "Failed to register shard"}), 500
        
        # Add embedding to vector index if provided
        if "embedding" in data:
            import numpy as np
            embedding = np.array(data["embedding"], dtype='float32')
            
            if embedding.shape[0] == 768:
                search_engine.add_embedding(data["info_hash"], embedding)
                search_engine.save_index()
            else:
                logger.warning(f"Invalid embedding dimension: {embedding.shape[0]}")
        
        return jsonify({
            "status": "success",
            "info_hash": data["info_hash"],
            "message": "Shard registered successfully"
        })
    
    except Exception as e:
        logger.exception("Register error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search_shards():
    """
    DEPRECATED: Text-based search removed. Use /api/search/embedding instead.
    
    Clients should embed queries locally and send vectors.
    This keeps the tracker lightweight and avoids model dependencies.
    """
    return jsonify({
        "error": "Text search deprecated. Use /api/search/embedding",
        "message": "Please embed your query locally and POST to /api/search/embedding with the vector",
        "example": {
            "url": "/api/search/embedding",
            "body": {
                "embedding": [0.1, 0.2, "... 768 dimensions"],
                "limit": 10
            }
        }
    }), 410  # 410 Gone


@app.route('/api/search/embedding', methods=['POST'])
def search_by_embedding():
    """
    Search directly by embedding vector.
    
    Body:
        {
            "embedding": [0.1, 0.2, ...],  // 768-dim
            "limit": 10
        }
    """
    try:
        data = request.get_json()
        embedding = data.get('embedding')
        limit = data.get('limit', 10)
        
        if not embedding or len(embedding) != 768:
            return jsonify({"error": "Invalid embedding (must be 768-dim)"}), 400
        
        import numpy as np
        query_embedding = np.array(embedding, dtype='float32')
        
        # Search
        vector_results = search_engine.search(
            query_embedding=query_embedding,
            k=limit,
            min_similarity=0.3,
        )
        
        # Get metadata
        results = []
        for info_hash, similarity in vector_results:
            shard = metadata_store.get_shard(info_hash)
            if shard:
                results.append({
                    "info_hash": info_hash,
                    "display_name": shard['display_name'],
                    "similarity": round(similarity, 4),
                    "magnet_link": _build_magnet_link(shard),
                })
        
        return jsonify({
            "status": "success",
            "results": results,
            "count": len(results),
        })
    
    except Exception as e:
        logger.exception("Embedding search error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/peers/<info_hash>', methods=['GET'])
def get_peers_for_shard(info_hash):
    """
    Get list of peers seeding/leeching a shard.
    
    Returns peer_id (which can be agent_id) for client-side trust computation.
    
    Response:
    {
        "info_hash": "abc123",
        "seeders": [
            {"agent_id": "agent_xyz", "ip": "1.2.3.4", "port": 6889},
            ...
        ],
        "leechers": [...]
    }
    """
    try:
        # Get all peers
        peers = metadata_store.get_peers(info_hash, max_peers=200)
        
        # Split into seeders and leechers
        seeders = [
            {
                "agent_id": p.get("peer_id"),  # peer_id = agent_id
                "ip": p.get("ip"),
                "port": p.get("port")
            }
            for p in peers if p.get("is_seeder")
        ]
        
        leechers = [
            {
                "agent_id": p.get("peer_id"),
                "ip": p.get("ip"),
                "port": p.get("port")
            }
            for p in peers if not p.get("is_seeder")
        ]
        
        return jsonify({
            "info_hash": info_hash,
            "seeders": seeders,
            "leechers": leechers,
            "seeder_count": len(seeders),
            "leecher_count": len(leechers)
        })
    
    except Exception as e:
        logger.exception(f"Get peers error for {info_hash}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/quality/submit', methods=['POST'])
def submit_attestation():
    """
    Submit a quality attestation for a shard.
    
    Request body:
    {
        "shard_hash": "abc123...",
        "provider_agent_id": "agent_xyz",
        "consumer_agent_id": "agent_abc",
        "rating": 0.9,
        "feedback": "Excellent shard!",
        "signature": "base64_signature...",
        "timestamp": "2026-02-04T..."
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ['shard_hash', 'provider_agent_id', 'consumer_agent_id', 
                   'rating', 'signature', 'timestamp']
        if not all(field in data for field in required):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Validate rating
        if not (0.0 <= data['rating'] <= 1.0):
            return jsonify({"error": "Rating must be between 0.0 and 1.0"}), 400
        
        # TODO: Verify signature (requires public key lookup)
        # For now, accept attestations
        
        # Submit to metadata store
        success = metadata_store.submit_attestation(data)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Attestation accepted"
            })
        else:
            return jsonify({"error": "Failed to store attestation"}), 500
    
    except Exception as e:
        logger.exception("Attestation submission error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/quality/reputation/<agent_id>', methods=['GET'])
def get_agent_reputation(agent_id):
    """Get reputation for a specific agent."""
    try:
        reputation = metadata_store.get_reputation(agent_id)
        
        if reputation:
            return jsonify({
                "status": "success",
                "reputation": reputation
            })
        else:
            return jsonify({
                "status": "success",
                "reputation": {
                    "agent_id": agent_id,
                    "average_rating": 0.0,
                    "total_downloads": 0,
                    "trust_score": 0.5,
                    "note": "No reputation data available"
                }
            })
    
    except Exception as e:
        logger.exception("Reputation lookup error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/quality/top-agents', methods=['GET'])
def get_top_agents():
    """Get top-rated agents."""
    try:
        limit = int(request.args.get('limit', 10))
        top_agents = metadata_store.get_top_agents(limit)
        
        return jsonify({
            "status": "success",
            "agents": top_agents
        })
    
    except Exception as e:
        logger.exception("Top agents lookup error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/quality/attestations/<shard_hash>', methods=['GET'])
def get_shard_attestations(shard_hash):
    """Get all attestations for a specific shard."""
    try:
        attestations = metadata_store.get_attestations_for_shard(shard_hash)
        
        return jsonify({
            "status": "success",
            "shard_hash": shard_hash,
            "count": len(attestations),
            "attestations": attestations
        })
    
    except Exception as e:
        logger.exception("Attestations lookup error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/agent/register', methods=['POST'])
def register_agent():
    """
    Register an agent identity with the tracker.
    
    Request body:
    {
        "agent_id": "abc123...",
        "public_key": "base64_key...",
        "metadata": {"version": "1.0", ...}
    }
    """
    try:
        data = request.get_json()
        
        if not all(k in data for k in ['agent_id', 'public_key']):
            return jsonify({"error": "Missing required fields"}), 400
        
        metadata_store.register_agent(
            data['agent_id'],
            data['public_key'],
            data.get('metadata')
        )
        
        return jsonify({
            "status": "success",
            "message": "Agent registered"
        })
    
    except Exception as e:
        logger.exception("Agent registration error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get tracker statistics."""
    try:
        stats = metadata_store.get_statistics()
        vector_stats = search_engine.get_statistics()
        quality_stats = metadata_store.get_quality_statistics()
        
        return jsonify({
            "status": "success",
            "tracker": stats,
            "vector_search": vector_stats,
            "quality": quality_stats,
        })
    
    except Exception as e:
        logger.exception("Stats error")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Helper Functions
# ============================================================================

def _format_peers(peers: list) -> list:
    """Format peer list for BitTorrent response."""
    formatted = []
    for peer in peers:
        formatted.append({
            'peer id': peer.get('peer_id', ''),
            'ip': peer.get('ip', ''),
            'port': peer.get('port', 0),
        })
    return formatted


def _build_magnet_link(shard: dict) -> str:
    """Build magnet link from shard metadata."""
    from urllib.parse import quote
    
    info_hash = shard['info_hash']
    name = shard['display_name']
    
    magnet = f"magnet:?xt=urn:btih:{info_hash}&dn={quote(name)}"
    
    # Add OpenClaw extensions
    magnet += f"&x.model={quote(shard['embedding_model'])}"
    magnet += f"&x.dims={shard['dimension_size']}"
    
    if shard.get('tags'):
        magnet += f"&x.tags={quote(','.join(shard['tags']))}"
    
    return magnet


def bencode_dict(data: dict) -> bytes:
    """Simple bencoding for tracker responses."""
    # This is a simplified version - production should use bencodepy
    import json
    # For now, return JSON (most modern clients support it)
    response = make_response(json.dumps(data))
    response.headers['Content-Type'] = 'text/plain'
    return response


def bencode_error(message: str) -> bytes:
    """Return bencoded error message."""
    return bencode_dict({'failure reason': message})


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import os
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=False)
