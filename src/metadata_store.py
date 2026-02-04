"""
Metadata storage for the Synapse Tracker.

Uses SQLite for metadata and provides integration with vector search.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MetadataStore:
    """
    Persistent storage for memory shard metadata.
    
    Stores:
    - Shard information (hash, name, model, dimensions)
    - Peer statistics (seeders, leechers)
    - Announce history
    - Tags and search metadata
    """
    
    def __init__(self, db_path: str = "./tracker.db"):
        """
        Initialize the metadata store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"MetadataStore initialized: {db_path}")
    
    def _init_database(self):
        """Create database schema if it doesn't exist."""
        with self._get_connection() as conn:
            # Shards table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shards (
                    info_hash TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    dimension_size INTEGER NOT NULL,
                    entry_count INTEGER DEFAULT 0,
                    file_size INTEGER DEFAULT 0,
                    tags TEXT,  -- JSON array
                    description TEXT,
                    created_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    seeders INTEGER DEFAULT 0,
                    leechers INTEGER DEFAULT 0,
                    completed INTEGER DEFAULT 0,
                    metadata TEXT  -- JSON for additional fields
                )
            """)
            
            # Peers table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS peers (
                    peer_id TEXT NOT NULL,
                    info_hash TEXT NOT NULL,
                    ip TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    uploaded INTEGER DEFAULT 0,
                    downloaded INTEGER DEFAULT 0,
                    left_bytes INTEGER DEFAULT 0,
                    last_announce TEXT NOT NULL,
                    is_seeder BOOLEAN DEFAULT 0,
                    PRIMARY KEY (peer_id, info_hash),
                    FOREIGN KEY (info_hash) REFERENCES shards(info_hash)
                )
            """)
            
            # Announces table (for statistics)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS announces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    info_hash TEXT NOT NULL,
                    peer_id TEXT NOT NULL,
                    event TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (info_hash) REFERENCES shards(info_hash)
                )
            """)
            
            # Agent identities table (added 2026 for PQ security)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    public_key TEXT NOT NULL,
                    algorithm TEXT DEFAULT 'ML-DSA-87',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_shares INTEGER DEFAULT 0,
                    metadata TEXT  -- JSON for additional agent info
                )
            """)
            
            # Quality attestations table (added 2026)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attestations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    shard_hash TEXT NOT NULL,
                    provider_agent_id TEXT NOT NULL,
                    consumer_agent_id TEXT NOT NULL,
                    rating REAL NOT NULL,
                    feedback TEXT,
                    signature TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (shard_hash) REFERENCES shards(info_hash),
                    FOREIGN KEY (provider_agent_id) REFERENCES agents(agent_id),
                    FOREIGN KEY (consumer_agent_id) REFERENCES agents(agent_id)
                )
            """)
            
            # Reputation scores table (aggregated view)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reputation (
                    agent_id TEXT PRIMARY KEY,
                    average_rating REAL DEFAULT 0.0,
                    total_downloads INTEGER DEFAULT 0,
                    positive_attestations INTEGER DEFAULT 0,
                    negative_attestations INTEGER DEFAULT 0,
                    trust_score REAL DEFAULT 0.5,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shards_model ON shards(embedding_model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shards_last_seen ON shards(last_seen)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_peers_info_hash ON peers(info_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_peers_last_announce ON peers(last_announce)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agents_last_seen ON agents(last_seen)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attestations_shard ON attestations(shard_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attestations_provider ON attestations(provider_agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attestations_timestamp ON attestations(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reputation_trust ON reputation(trust_score)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def register_shard(self, shard_info: Dict[str, Any]) -> bool:
        """
        Register a new memory shard or update existing one.
        
        Args:
            shard_info: Dictionary with shard metadata
            
        Returns:
            True if successful
        """
        required_fields = ["info_hash", "display_name", "embedding_model", "dimension_size"]
        if not all(field in shard_info for field in required_fields):
            logger.error(f"Missing required fields: {required_fields}")
            return False
        
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            
            # Prepare data
            tags_json = json.dumps(shard_info.get("tags", []))
            metadata_json = json.dumps(shard_info.get("metadata", {}))
            
            # Upsert shard
            conn.execute("""
                INSERT INTO shards (
                    info_hash, display_name, embedding_model, dimension_size,
                    entry_count, file_size, tags, description,
                    created_at, last_seen, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(info_hash) DO UPDATE SET
                    display_name = excluded.display_name,
                    last_seen = excluded.last_seen,
                    tags = excluded.tags,
                    metadata = excluded.metadata
            """, (
                shard_info["info_hash"],
                shard_info["display_name"],
                shard_info["embedding_model"],
                shard_info["dimension_size"],
                shard_info.get("entry_count", 0),
                shard_info.get("file_size", 0),
                tags_json,
                shard_info.get("description", ""),
                now,
                now,
                metadata_json,
            ))
            
            conn.commit()
            logger.info(f"Registered shard: {shard_info['info_hash']}")
            return True
    
    def get_shard(self, info_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get shard metadata by info hash.
        
        Args:
            info_hash: The info hash to look up
            
        Returns:
            Dictionary with shard data or None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM shards WHERE info_hash = ?",
                (info_hash,)
            ).fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def search_shards(
        self,
        model_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for shards with filters.
        
        Args:
            model_filter: Filter by embedding model
            tag_filter: Filter by tags (ANY match)
            limit: Maximum results
            
        Returns:
            List of shard dictionaries
        """
        with self._get_connection() as conn:
            query = "SELECT * FROM shards WHERE 1=1"
            params = []
            
            if model_filter:
                query += " AND embedding_model = ?"
                params.append(model_filter)
            
            if tag_filter:
                # Check if any tag matches
                tag_conditions = " OR ".join(["tags LIKE ?" for _ in tag_filter])
                query += f" AND ({tag_conditions})"
                params.extend([f'%"{tag}"%' for tag in tag_filter])
            
            query += " ORDER BY last_seen DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def update_peer(
        self,
        peer_id: str,
        info_hash: str,
        ip: str,
        port: int,
        uploaded: int = 0,
        downloaded: int = 0,
        left: int = 0,
    ) -> bool:
        """
        Update peer information from announce.
        
        Args:
            peer_id: Peer identifier
            info_hash: Info hash being announced
            ip: Peer IP address
            port: Peer port
            uploaded: Bytes uploaded
            downloaded: Bytes downloaded
            left: Bytes left to download
            
        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            is_seeder = (left == 0)
            
            # Upsert peer
            conn.execute("""
                INSERT INTO peers (
                    peer_id, info_hash, ip, port,
                    uploaded, downloaded, left_bytes,
                    last_announce, is_seeder
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(peer_id, info_hash) DO UPDATE SET
                    ip = excluded.ip,
                    port = excluded.port,
                    uploaded = excluded.uploaded,
                    downloaded = excluded.downloaded,
                    left_bytes = excluded.left_bytes,
                    last_announce = excluded.last_announce,
                    is_seeder = excluded.is_seeder
            """, (peer_id, info_hash, ip, port, uploaded, downloaded, left, now, is_seeder))
            
            # Update shard stats
            self._update_shard_stats(conn, info_hash)
            
            conn.commit()
            return True
    
    def _update_shard_stats(self, conn, info_hash: str):
        """Update seeder/leecher counts for a shard."""
        # Count active seeders and leechers
        stats = conn.execute("""
            SELECT 
                SUM(CASE WHEN is_seeder THEN 1 ELSE 0 END) as seeders,
                SUM(CASE WHEN NOT is_seeder THEN 1 ELSE 0 END) as leechers
            FROM peers
            WHERE info_hash = ?
            AND datetime(last_announce) > datetime('now', '-2 hours')
        """, (info_hash,)).fetchone()
        
        # Update shard
        conn.execute("""
            UPDATE shards
            SET seeders = ?, leechers = ?, last_seen = ?
            WHERE info_hash = ?
        """, (
            stats["seeders"] or 0,
            stats["leechers"] or 0,
            datetime.utcnow().isoformat(),
            info_hash,
        ))
    
    def get_peers(self, info_hash: str, max_peers: int = 50) -> List[Dict[str, Any]]:
        """
        Get active peers for a shard.
        
        Args:
            info_hash: The info hash
            max_peers: Maximum peers to return
            
        Returns:
            List of peer dictionaries
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT peer_id, ip, port, is_seeder
                FROM peers
                WHERE info_hash = ?
                AND datetime(last_announce) > datetime('now', '-1 hour')
                ORDER BY last_announce DESC
                LIMIT ?
            """, (info_hash, max_peers)).fetchall()
            
            return [dict(row) for row in rows]
    
    def record_announce(self, info_hash: str, peer_id: str, event: Optional[str] = None):
        """Record an announce event for statistics."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO announces (info_hash, peer_id, event, timestamp)
                VALUES (?, ?, ?, ?)
            """, (info_hash, peer_id, event, datetime.utcnow().isoformat()))
            
            # Track completed downloads
            if event == "completed":
                conn.execute("""
                    UPDATE shards
                    SET completed = completed + 1
                    WHERE info_hash = ?
                """, (info_hash,))
            
            conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        with self._get_connection() as conn:
            stats = {}
            
            # Total shards
            stats["total_shards"] = conn.execute("SELECT COUNT(*) as count FROM shards").fetchone()["count"]
            
            # Active shards (seen in last 24h)
            stats["active_shards"] = conn.execute("""
                SELECT COUNT(*) as count FROM shards
                WHERE datetime(last_seen) > datetime('now', '-1 day')
            """).fetchone()["count"]
            
            # Total peers
            stats["total_peers"] = conn.execute("""
                SELECT COUNT(DISTINCT peer_id) as count FROM peers
                WHERE datetime(last_announce) > datetime('now', '-1 hour')
            """).fetchone()["count"]
            
            # Total announces
            stats["total_announces"] = conn.execute("SELECT COUNT(*) as count FROM announces").fetchone()["count"]
            
            # Top models
            top_models = conn.execute("""
                SELECT embedding_model, COUNT(*) as count
                FROM shards
                GROUP BY embedding_model
                ORDER BY count DESC
                LIMIT 5
            """).fetchall()
            stats["top_models"] = [dict(row) for row in top_models]
            
            return stats
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary with JSON parsing."""
        data = dict(row)
        
        # Parse JSON fields
        if "tags" in data and data["tags"]:
            data["tags"] = json.loads(data["tags"])
        
        if "metadata" in data and data["metadata"]:
            data["metadata"] = json.loads(data["metadata"])
        
        return data
    
    def cleanup_stale_peers(self, hours: int = 2):
        """Remove peers that haven't announced recently."""
        with self._get_connection() as conn:
            conn.execute("""
                DELETE FROM peers
                WHERE datetime(last_announce) < datetime('now', ?)
            """, (f'-{hours} hours',))
            
            conn.commit()
            logger.info(f"Cleaned up stale peers (>{hours}h)")
    
    # === Quality & Reputation Methods (added 2026) ===
    
    def register_agent(self, agent_id: str, public_key: str, metadata: Optional[Dict] = None):
        """Register a new agent identity."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO agents (agent_id, public_key, first_seen, last_seen, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    metadata = excluded.metadata
            """, (agent_id, public_key, now, now, json.dumps(metadata or {})))
            conn.commit()
            logger.info(f"Registered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent information."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM agents WHERE agent_id = ?
            """, (agent_id,)).fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def submit_attestation(self, attestation_data: Dict) -> bool:
        """
        Submit a quality attestation to the tracker.
        
        Args:
            attestation_data: Dictionary with shard_hash, provider_agent_id,
                            consumer_agent_id, rating, feedback, signature, timestamp
        
        Returns:
            True if accepted
        """
        with self._get_connection() as conn:
            # Insert attestation
            conn.execute("""
                INSERT INTO attestations 
                (shard_hash, provider_agent_id, consumer_agent_id, rating, 
                 feedback, signature, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                attestation_data["shard_hash"],
                attestation_data["provider_agent_id"],
                attestation_data["consumer_agent_id"],
                attestation_data["rating"],
                attestation_data.get("feedback", ""),
                attestation_data["signature"],
                attestation_data["timestamp"]
            ))
            
            # Update provider's share count
            conn.execute("""
                UPDATE agents
                SET total_shares = total_shares + 1
                WHERE agent_id = ?
            """, (attestation_data["provider_agent_id"],))
            
            conn.commit()
            
            # Recalculate reputation
            self._update_reputation(attestation_data["provider_agent_id"])
            
            logger.info(f"Attestation submitted for {attestation_data['shard_hash'][:8]}")
            return True
    
    def _update_reputation(self, agent_id: str):
        """Recalculate reputation score for an agent."""
        with self._get_connection() as conn:
            # Get all attestations for this provider
            rows = conn.execute("""
                SELECT rating, timestamp FROM attestations
                WHERE provider_agent_id = ?
                ORDER BY timestamp DESC
            """, (agent_id,)).fetchall()
            
            if not rows:
                return
            
            ratings = [row["rating"] for row in rows]
            total_downloads = len(ratings)
            average_rating = sum(ratings) / total_downloads
            positive = sum(1 for r in ratings if r >= 0.7)
            negative = sum(1 for r in ratings if r < 0.4)
            
            # Calculate trust score (same formula as QualityTracker)
            volume_factor = min(1.0, total_downloads / 100.0)
            ratio_factor = positive / total_downloads if total_downloads > 0 else 0.5
            trust_score = 0.5 * average_rating + 0.3 * volume_factor + 0.2 * ratio_factor
            
            # Update reputation table
            conn.execute("""
                INSERT INTO reputation 
                (agent_id, average_rating, total_downloads, positive_attestations,
                 negative_attestations, trust_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    average_rating = excluded.average_rating,
                    total_downloads = excluded.total_downloads,
                    positive_attestations = excluded.positive_attestations,
                    negative_attestations = excluded.negative_attestations,
                    trust_score = excluded.trust_score,
                    last_updated = excluded.last_updated
            """, (
                agent_id, average_rating, total_downloads, positive,
                negative, trust_score, datetime.utcnow().isoformat()
            ))
            
            conn.commit()
    
    def get_reputation(self, agent_id: str) -> Optional[Dict]:
        """Get reputation for an agent."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM reputation WHERE agent_id = ?
            """, (agent_id,)).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_top_agents(self, limit: int = 10) -> List[Dict]:
        """Get top-rated agents."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT 
                    r.*,
                    a.metadata
                FROM reputation r
                JOIN agents a ON r.agent_id = a.agent_id
                ORDER BY r.trust_score DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [self._row_to_dict(row) for row in rows]
    
    def get_attestations_for_shard(self, shard_hash: str) -> List[Dict]:
        """Get all attestations for a specific shard."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM attestations
                WHERE shard_hash = ?
                ORDER BY timestamp DESC
            """, (shard_hash,)).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_attestations_by_provider(self, provider_id: str, limit: int = 50) -> List[Dict]:
        """Get recent attestations for a provider."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM attestations
                WHERE provider_agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (provider_id, limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_quality_statistics(self) -> Dict:
        """Get tracker-wide quality statistics."""
        with self._get_connection() as conn:
            stats = {}
            
            stats["total_agents"] = conn.execute(
                "SELECT COUNT(*) as count FROM agents"
            ).fetchone()["count"]
            
            stats["total_attestations"] = conn.execute(
                "SELECT COUNT(*) as count FROM attestations"
            ).fetchone()["count"]
            
            stats["average_rating_global"] = conn.execute(
                "SELECT AVG(rating) as avg FROM attestations"
            ).fetchone()["avg"] or 0.0
            
            stats["average_trust_score"] = conn.execute(
                "SELECT AVG(trust_score) as avg FROM reputation"
            ).fetchone()["avg"] or 0.0
            
            return stats
    
    # ========================================================================
    # TrustRank Support Methods (Added 2026-02-04)
    # ========================================================================
    
    def get_seeders(self, info_hash: str) -> List[Dict]:
        """
        Get all current seeders for a shard.
        
        Args:
            info_hash: Shard hash
        
        Returns:
            List of seeder info with agent_id, last_announce, etc.
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT 
                    p.peer_id,
                    p.ip,
                    p.port,
                    p.last_announce,
                    p.uploaded,
                    p.downloaded,
                    a.agent_id,
                    a.public_key
                FROM peers p
                LEFT JOIN agents a ON p.peer_id = a.agent_id
                WHERE p.info_hash = ?
                AND p.is_seeder = 1
                AND datetime(p.last_announce) > datetime('now', '-2 hours')
            """, (info_hash,)).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_all_shard_hashes(self) -> List[str]:
        """Get all shard hashes in database."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT info_hash FROM shards").fetchall()
            return [row["info_hash"] for row in rows]
    
    def get_agent_reputation(self, agent_id: str) -> float:
        """
        Get normalized reputation score for an agent.
        
        Returns:
            Float 0.0-1.0 representing reputation (0.5 = neutral)
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT trust_score FROM reputation WHERE agent_id = ?
            """, (agent_id,)).fetchone()
            
            if row and row["trust_score"] is not None:
                return float(row["trust_score"])
            
            # Default neutral reputation for new agents
            return 0.5
    
    def get_trusting_agents(self, agent_id: str, min_rating: float = 0.7) -> List[str]:
        """
        Get agents who have positively attested to this agent's shards.
        
        Args:
            agent_id: Target agent
            min_rating: Minimum rating to count as "trusting" (default 0.7)
        
        Returns:
            List of agent IDs who trust this agent
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT DISTINCT consumer_agent_id
                FROM attestations
                WHERE provider_agent_id = ?
                AND rating >= ?
            """, (agent_id, min_rating)).fetchall()
            
            return [row["consumer_agent_id"] for row in rows]
    
    def get_shard_stats(self, info_hash: str) -> Dict:
        """
        Get statistics for a shard.
        
        Returns:
            Dict with completed, seeders, leechers, etc.
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT 
                    seeders,
                    leechers,
                    completed,
                    created_at,
                    last_seen
                FROM shards
                WHERE info_hash = ?
            """, (info_hash,)).fetchone()
            
            if row:
                return dict(row)
            
            return {
                "seeders": 0,
                "leechers": 0,
                "completed": 0,
                "created_at": None,
                "last_seen": None,
            }
    
    def has_attestation(self, agent_a: str, agent_b: str) -> bool:
        """
        Check if agent A has attested to agent B (or vice versa).
        
        Used for trust graph diversity calculation.
        
        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
        
        Returns:
            True if there's any attestation between them
        """
        with self._get_connection() as conn:
            count = conn.execute("""
                SELECT COUNT(*) as count
                FROM attestations
                WHERE (provider_agent_id = ? AND consumer_agent_id = ?)
                   OR (provider_agent_id = ? AND consumer_agent_id = ?)
            """, (agent_a, agent_b, agent_b, agent_a)).fetchone()["count"]
            
            return count > 0

