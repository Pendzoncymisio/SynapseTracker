#!/usr/bin/env python3
"""
Run the Synapse Tracker server.
"""

import sys
import logging
from pathlib import Path

# Add embeddings module from synapse-protocol
sys.path.insert(0, str(Path(__file__).parent.parent / "synapse-protocol"))

from tracker_server import app

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("tracker.log"),
            logging.StreamHandler()
        ]
    )
    
    print("=" * 60)
    print("Synapse Tracker - Memory Shard P2P Network")
    print("=" * 60)
    print()
    print("HTTP API:  http://0.0.0.0:6881")
    print("Announce:  http://0.0.0.0:6881/announce")
    print("Scrape:    http://0.0.0.0:6881/scrape")
    print("Search:    http://0.0.0.0:6881/api/search")
    print("Stats:     http://0.0.0.0:6881/api/stats")
    print()
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=6881, debug=False)
