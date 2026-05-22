"""
Artifact storage for the four-role agent system.

Provides file-based persistence for large/raw payloads under code/state/artifacts/.
Artifacts are identified by deterministic content-hash IDs (art:<sha256-prefix>).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from schemas import Artifact, artifact_id_from_bytes

if TYPE_CHECKING:
    pass

# Default artifact storage directory
STATE_DIR = Path(__file__).parent / "state"
ARTIFACTS_DIR = STATE_DIR / "artifacts"


def _ensure_dirs() -> None:
    """Create state directories if they don't exist."""
    STATE_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)


class ArtifactStore:
    """
    File-based artifact storage.
    
    Each artifact is stored as two files:
    - {artifact_id}.bin  - the raw bytes
    - {artifact_id}.meta.json - the metadata (Artifact model)
    """
    
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or ARTIFACTS_DIR
        _ensure_dirs()
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _blob_path(self, artifact_id: str) -> Path:
        """Path to the artifact's binary content."""
        safe_id = artifact_id.replace(":", "_")
        return self.base_dir / f"{safe_id}.bin"
    
    def _meta_path(self, artifact_id: str) -> Path:
        """Path to the artifact's metadata JSON."""
        safe_id = artifact_id.replace(":", "_")
        return self.base_dir / f"{safe_id}.meta.json"
    
    def put(
        self,
        blob: bytes,
        *,
        content_type: str,
        source: str,
        descriptor: str,
    ) -> str:
        """
        Store a blob and return its artifact ID.
        
        If an artifact with the same content already exists, returns its ID
        without re-writing (content-addressed dedup).
        """
        artifact_id = artifact_id_from_bytes(blob)
        
        # Content-addressed: skip if already stored
        if self.exists(artifact_id):
            return artifact_id
        
        # Create metadata
        meta = Artifact(
            id=artifact_id,
            content_type=content_type,
            size_bytes=len(blob),
            source=source,
            descriptor=descriptor[:300],  # bound descriptor length
        )
        
        # Write blob
        self._blob_path(artifact_id).write_bytes(blob)
        
        # Write metadata
        self._meta_path(artifact_id).write_text(
            meta.model_dump_json(indent=2),
            encoding="utf-8",
        )
        
        return artifact_id
    
    def get_bytes(self, artifact_id: str) -> bytes:
        """Retrieve the raw bytes for an artifact."""
        path = self._blob_path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        return path.read_bytes()
    
    def get_meta(self, artifact_id: str) -> Artifact:
        """Retrieve the metadata for an artifact."""
        path = self._meta_path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"Artifact metadata not found: {artifact_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Artifact.model_validate(data)
    
    def exists(self, artifact_id: str) -> bool:
        """Check if an artifact exists in the store."""
        return self._blob_path(artifact_id).exists()
    
    def list_all(self) -> list[str]:
        """List all artifact IDs in the store."""
        ids = []
        for p in self.base_dir.glob("*.meta.json"):
            # Convert filename back to artifact ID
            safe_id = p.stem.replace(".meta", "")
            artifact_id = safe_id.replace("_", ":", 1)
            ids.append(artifact_id)
        return ids
    
    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact. Returns True if it existed."""
        blob_path = self._blob_path(artifact_id)
        meta_path = self._meta_path(artifact_id)
        existed = blob_path.exists()
        if blob_path.exists():
            blob_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        return existed


# Module-level singleton for convenience
_store: ArtifactStore | None = None


def get_store() -> ArtifactStore:
    """Get the global artifact store instance."""
    global _store
    if _store is None:
        _store = ArtifactStore()
    return _store


def put(blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
    """Store a blob and return its artifact ID."""
    return get_store().put(blob, content_type=content_type, source=source, descriptor=descriptor)


def get_bytes(artifact_id: str) -> bytes:
    """Retrieve the raw bytes for an artifact."""
    return get_store().get_bytes(artifact_id)


def get_meta(artifact_id: str) -> Artifact:
    """Retrieve the metadata for an artifact."""
    return get_store().get_meta(artifact_id)


def exists(artifact_id: str) -> bool:
    """Check if an artifact exists."""
    return get_store().exists(artifact_id)
