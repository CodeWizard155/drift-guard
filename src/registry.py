"""A small file-based model registry (like a mini MLflow Model Registry).

Every trained model is saved as models/model_v<N>.joblib and described in
models/registry.json with its metrics, training data and status.
Exactly one model is in "production" at a time; older ones are "archived".
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from . import config


class ModelRegistry:
    def __init__(self, path: Path | None = None):
        self.dir = Path(path or config.MODEL_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.dir / "registry.json"
        self.entries: list[dict] = (
            json.loads(self.index_file.read_text()) if self.index_file.exists() else []
        )

    # -- helpers -----------------------------------------------------------
    def _save_index(self) -> None:
        self.index_file.write_text(json.dumps(self.entries, indent=2))

    @property
    def next_version(self) -> int:
        return len(self.entries) + 1

    # -- public API --------------------------------------------------------
    def register(self, model, metrics: dict, trained_on: str, note: str = "") -> int:
        version = self.next_version
        file = self.dir / f"model_v{version}.joblib"
        joblib.dump(model, file)
        self.entries.append({
            "version": version,
            "file": file.name,
            "status": "staging",
            "metrics": metrics,
            "trained_on": trained_on,
            "note": note,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        self._save_index()
        return version

    def promote(self, version: int) -> None:
        """Make `version` the production model and archive the previous one."""
        for e in self.entries:
            if e["status"] == "production":
                e["status"] = "archived"
            if e["version"] == version:
                e["status"] = "production"
        self._save_index()

    def reject(self, version: int) -> None:
        for e in self.entries:
            if e["version"] == version:
                e["status"] = "rejected"
        self._save_index()

    def production_entry(self) -> dict | None:
        return next((e for e in self.entries if e["status"] == "production"), None)

    def load(self, version: int | None = None):
        entry = (self.production_entry() if version is None
                 else next(e for e in self.entries if e["version"] == version))
        if entry is None:
            raise LookupError("No production model registered yet")
        return joblib.load(self.dir / entry["file"])
