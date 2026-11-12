"""A session's safety latch: what a hazard stop did, and what it blocks until a person acknowledges.

Anything can report a hazard: the VLM monitor, the GUI, or an external caller over
the API. The session stops the control (see ``RuntimeSession.safety_stop``), moves
the follower home or holds it, and latches. While latched, the stopped kind of
control cannot start again; acknowledging clears the latch and restarts nothing.

The latch only keeps state; the session does the stopping and moving, as it owns the
controls and the store. The configuration, with the captured home pose, is kept per
follower robot under ``<storage>/safety/<robot_id>.json`` so it outlives the session.
"""

from __future__ import annotations

import datetime
import json
import time
from collections import Counter, deque
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from schemas.runtime_session import SafetyConfig, SafetyEvent, SafetyStatus, SafetyStopRequest, SessionSafety

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from schemas.runtime_session import ControlConfig, SafetyEventKind

_EVENTS_KEPT = 500


class SafetyLatch:
    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._robot_id: UUID | None = None
        self.config = SafetyConfig()
        self.latched_by: SafetyEvent | None = None
        self._homing_until: float | None = None
        """``time.monotonic()`` when a move home is due to arrive."""
        self._events: deque[SafetyEvent] = deque(maxlen=_EVENTS_KEPT)
        self._counters: Counter[str] = Counter()

    # Configuration, per follower.

    def use_robot(self, robot_id: UUID) -> None:
        """Take the configuration saved for ``robot_id``, the follower now loaded. Keeps the latch."""
        if robot_id == self._robot_id:
            return
        self._robot_id = robot_id
        path = self._path(robot_id)
        try:
            self.config = SafetyConfig.model_validate_json(path.read_text()) if path.is_file() else SafetyConfig()
        except (OSError, ValueError) as error:
            logger.warning("Ignoring unreadable safety config {}: {}", path, error)
            self.config = SafetyConfig()

    def configure(self, config: SafetyConfig, *, source: str) -> SafetyEvent:
        self.config = config
        self._save()
        return self.record("config", source=source)

    def capture_home(self, pose: list[float], *, source: str) -> SafetyEvent:
        self.config = self.config.model_copy(update={"home_pose": pose})
        self._save()
        return self.record("home_captured", source=source)

    def _save(self) -> None:
        if self._robot_id is None:
            return  # nothing loaded to keep it for; it applies to this session only
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._path(self._robot_id).write_text(json.dumps(self.config.model_dump(mode="json"), indent=2))

    def _path(self, robot_id: UUID) -> Path:
        return self._config_dir / f"{robot_id}.json"

    # The latch.

    @property
    def latched(self) -> bool:
        return self.latched_by is not None

    @property
    def homing(self) -> bool:
        return self._homing_until is not None and time.monotonic() < self._homing_until

    def blocks(self, config: ControlConfig | None) -> bool:
        """Whether starting ``config`` would restart what a latched stop ended."""
        return self.latched and config is not None and config.type in self.config.stop_controls

    def latch(
        self,
        request: SafetyStopRequest,
        *,
        stopped_control: str | None,
        motion: str,
        homing_s: float | None,
    ) -> SafetyEvent:
        """Record a stop. The first stop latches; later ones are recorded and keep the first as the cause."""
        event = self.record(
            "stop",
            source=request.source,
            hazard=request.hazard,
            reason=request.reason,
            confidence=request.confidence,
            stopped_control=stopped_control,
            motion=motion,
        )
        if self.latched_by is None:
            self.latched_by = event
        self._homing_until = time.monotonic() + homing_s if homing_s is not None else None
        return event

    def acknowledge(self, *, source: str) -> SafetyEvent | None:
        """Clear the latch; ``None`` when nothing was latched. A move home in progress carries on."""
        if self.latched_by is None:
            return None
        self.latched_by = None
        return self.record("acknowledge", source=source)

    def moving_home(self, homing_s: float, *, source: str) -> SafetyEvent:
        """A move home asked for directly, without a hazard: nothing latches."""
        self._homing_until = time.monotonic() + homing_s
        return self.record("home", source=source, motion="home")

    def stopped_moving(self) -> None:
        self._homing_until = None

    # Reporting.

    def record(self, kind: SafetyEventKind, *, source: str, **fields: object) -> SafetyEvent:
        event = SafetyEvent(
            id=uuid4().hex,
            at=datetime.datetime.now(tz=datetime.UTC),
            kind=kind,
            source=source,
            **fields,  # type: ignore[arg-type]
        )
        self._events.append(event)
        self._counters[kind] += 1
        return event

    def events(self, *, limit: int, stops_only: bool = False) -> list[SafetyEvent]:
        """Newest first."""
        events = [event for event in reversed(self._events) if not stops_only or event.kind == "stop"]
        return events[:limit]

    def summary(self) -> SessionSafety:
        state = "homing" if self.homing else "latched" if self.latched else "clear"
        return SessionSafety(state=state, latched_by=self.latched_by, home_pose_set=self.config.home_pose is not None)

    def status(self, session_id: str) -> SafetyStatus:
        return SafetyStatus(
            **self.summary().model_dump(), session_id=session_id, config=self.config, counters=dict(self._counters)
        )
