"""
In-memory job tracker for async forecast jobs.

Jobs are lost on process restart — acceptable for 1-min granularity data where
the next cycle replaces any missed computation. Move to Redis on multi-node scale-out.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    plants_total: int = 0
    plants_completed: int = 0
    plants_failed: int = 0
    errors: List[str] = field(default_factory=list)
    result_summary: Optional[Dict[str, Any]] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "plants_total": self.plants_total,
            "plants_completed": self.plants_completed,
            "plants_failed": self.plants_failed,
            "errors": self.errors[-20:],  # cap error list
            "result_summary": self.result_summary,
        }


class JobManager:
    """Thread-safe in-memory job store."""

    MAX_JOBS = 200  # Oldest jobs evicted when this is exceeded

    def __init__(self):
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> JobRecord:
        async with self._lock:
            job_id = str(uuid.uuid4())
            rec = JobRecord(job_id=job_id)
            self._jobs[job_id] = rec
            await self._evict_if_needed()
            return rec

    async def get(self, job_id: str) -> Optional[JobRecord]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_all(self) -> List[JobRecord]:
        async with self._lock:
            return list(self._jobs.values())

    async def update(self, job_id: str, **kwargs) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if rec:
                for k, v in kwargs.items():
                    if hasattr(rec, k):
                        setattr(rec, k, v)

    async def _evict_if_needed(self) -> None:
        """Remove oldest completed/failed jobs when cap is reached."""
        if len(self._jobs) <= self.MAX_JOBS:
            return
        terminal = [
            j for j in self._jobs.values()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.PARTIAL)
        ]
        terminal.sort(key=lambda j: j.created_at)
        for job in terminal[: len(self._jobs) - self.MAX_JOBS]:
            del self._jobs[job.job_id]


# Module-level singleton
job_manager = JobManager()
