#!/usr/bin/env python3
"""
Test client for the Synapse Tracker.

Demonstrates registration, search, and announce operations.
"""

import sys
import json
from pathlib import Path

# Add synapse-protocol to path for embeddings
sys.path.insert(0, str(Path(__file__).parent.parent / "synapse-protocol"))

import requests
import numpy as np
from embeddings import create_embedder


TRACKER_URL = "http://localhost:6881"


def test_register_shard():
    """Test registering a memory shard."""
    print("\n=== Test 1: Register Memory Shard ===")
    
    # Create embedder
    embedder = create_embedder(use_onnx=True)
    
    # Generate embedding for a sample knowledge shard
    text = "Complete guide to Kubernetes deployment and orchestration"
    embedding = embedder.encode(text)
    
    # Register shard
    data = {
        "info_hash": "a" * 40,  # Fake hash for demo
        "display_name": "Kubernetes Deployment Guide",
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        "dimension_size": 768,
        "tags": ["kubernetes", "devops", "docker"],
        "description": "Comprehensive guide for deploying applications on Kubernetes",
        "entry_count": 150,
        "file_size": 5242880,  # 5MB
        "embedding": embedding.tolist(),
    }
    
    response = requests.post(f"{TRACKER_URL}/api/register", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_search_query():
    """Test semantic search by text query."""
    print("\n=== Test 2: Search by Query ===")
    
    data = {
        "query": "How to deploy containers in production",
        "limit": 5,
        "filters": {
            "model": "nomic-ai/nomic-embed-text-v1.5"
        }
    }
    
    response = requests.post(f"{TRACKER_URL}/api/search", json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {results['count']} results:")
        for result in results['results']:
            print(f"  - {result['display_name']} (similarity: {result['similarity']})")
            print(f"    Seeders: {result['seeders']}, Leechers: {result['leechers']}")
            print(f"    Magnet: {result['magnet_link'][:80]}...")
    else:
        print(f"Error: {response.text}")


def test_search_embedding():
    """Test search by direct embedding."""
    print("\n=== Test 3: Search by Embedding ===")
    
    # Create a query embedding
    embedder = create_embedder(use_onnx=True)
    query = "React hooks and state management"
    embedding = embedder.encode(query)
    
    data = {
        "embedding": embedding.tolist(),
        "limit": 5,
    }
    
    response = requests.post(f"{TRACKER_URL}/api/search/embedding", json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {results['count']} results:")
        for result in results['results']:
            print(f"  - {result['display_name']} (similarity: {result['similarity']})")
    else:
        print(f"Error: {response.text}")


def test_announce():
    """Test BitTorrent announce."""
    print("\n=== Test 4: BitTorrent Announce ===")
    
    params = {
        "info_hash": "a" * 40,
        "peer_id": "SYNAPSE-TEST-12345",
        "port": 6881,
        "uploaded": 0,
        "downloaded": 0,
        "left": 5242880,
        "event": "started",
    }
    
    response = requests.get(f"{TRACKER_URL}/announce", params=params)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}...")


def test_stats():
    """Test statistics endpoint."""
    print("\n=== Test 5: Tracker Statistics ===")
    
    response = requests.get(f"{TRACKER_URL}/api/stats")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print(json.dumps(stats, indent=2))
    else:
        print(f"Error: {response.text}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Synapse Tracker - Test Client")
    print("=" * 60)
    
    # Check if tracker is running
    try:
        response = requests.get(f"{TRACKER_URL}/api/stats", timeout=2)
        print(f"✓ Tracker is running at {TRACKER_URL}")
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to tracker at {TRACKER_URL}")
        print("  Make sure the tracker is running: python run_tracker.py")
        return
    
    # Run tests
    try:
        test_register_shard()
        test_search_query()
        test_search_embedding()
        test_announce()
        test_stats()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
