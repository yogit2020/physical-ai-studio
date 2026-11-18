"""Runtime session websocket messages and the session listing.

Client to server: one JSON command per message, each with a ``request_id``
that its ``ack`` echoes. Server to client: ``ack``, ``state`` on change and
``error`` as JSON text, and ``observation`` at the sampling rate as one binary
message per tick: ``[u32 big-endian header length][ObservationMessage JSON][jpeg bytes]``,
the JPEGs concatenated in the order of ``frames``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID  # noqa: TC003 - pydantic resolves annotations at runtime

from pydantic import BaseModel, Field

from schemas.base import UTCDatetime
from schemas.environment import CameraEnvironmentConfiguration, RobotEnvironmentConfiguration


class EnvironmentDraft(BaseModel):
    """An environment to load, saved or still being edited in the form."""

    id: UUID | None = None
    """The saved environment this is, if it is one; the session reports it back."""
    name: str = "Unsaved environment"
    robots: list[RobotEnvironmentConfiguration]
    cameras: list[CameraEnvironmentConfiguration] = Field(default_factory=list)


class TeleopControlConfig(BaseModel):
    """Drive the follower from the environment's leader arm."""

    type: Literal["teleop"] = "teleop"
    hz: float = Field(100, gt=0, le=500, description="How often the leader is read; each reading is one command")
    grace_s: float = Field(
        3,
        ge=0,
        le=30,
        description="On every start, the follower closes the gap to the leader over this long instead of jumping",
    )


class ExecutionConfig(BaseModel):
    """How a model's action chunks are requested, as ``physicalai.runtime`` runs them."""

    mode: Literal["sync", "async"] = "sync"
    """``sync`` infers in the control's tick and waits; ``async`` infers alongside, while the queue plays."""
    request_threshold: float = Field(
        default=0.5, ge=0, le=1, description="Request the next chunk once this fraction of the current one is left"
    )


class ModelControlConfig(BaseModel):
    """Drive the follower with the loaded model."""

    type: Literal["model"] = "model"
    task: str = ""
    """What the model is asked to do, for a policy that takes a task."""
    hz: float = Field(30, gt=0, le=500, description="How often an action is taken from the model's queue")
    execution: ExecutionConfig = ExecutionConfig()
    grace_s: float = Field(
        3,
        ge=0,
        le=30,
        description="On every start, the follower closes the gap to the model's actions over this long",
    )


ControlConfig = Annotated[TeleopControlConfig | ModelControlConfig, Field(discriminator="type")]


class LoadEnvironmentCommand(BaseModel):
    command: Literal["load_environment"] = "load_environment"
    request_id: str
    environment: EnvironmentDraft
    unload_incompatible: bool = False
    """Replace a loaded environment even if a loaded part no longer fits it; that part is unloaded.

    Without it such a load fails with ``incompatible_parts`` and changes nothing.
    """


class UnloadEnvironmentCommand(BaseModel):
    command: Literal["unload_environment"] = "unload_environment"
    request_id: str


class ReloadEnvironmentCommand(BaseModel):
    """Connect the loaded environment's devices again, after one failed; the store and dataset stay."""

    command: Literal["reload_environment"] = "reload_environment"
    request_id: str


class ReloadEnvironment(BaseModel):
    """What the session asks the environment, on ``rpc/environment/reload``."""


class LoadModelCommand(BaseModel):
    """Load one of a model's exports to run. Needs an environment whose joints and cameras fit it."""

    command: Literal["load_model"] = "load_model"
    request_id: str
    model_id: UUID
    backend: str
    """One of the model's ``available_backends``."""
    device: str = "auto"


class UnloadModelCommand(BaseModel):
    command: Literal["unload_model"] = "unload_model"
    request_id: str


class LoadDatasetCommand(BaseModel):
    """Record into this dataset. Needs an environment whose joints and cameras fit it."""

    command: Literal["load_dataset"] = "load_dataset"
    request_id: str
    dataset_id: UUID
    fps: int | None = Field(
        None, ge=1, le=100, description="The rate to record a dataset with nothing recorded at; others keep their own"
    )


class UnloadDatasetCommand(BaseModel):
    """Stop recording into the dataset and copy what was recorded back into it."""

    command: Literal["unload_dataset"] = "unload_dataset"
    request_id: str


class StartEpisodeCommand(BaseModel):
    command: Literal["start_episode"] = "start_episode"
    request_id: str
    task: str


class SaveEpisodeCommand(BaseModel):
    command: Literal["save_episode"] = "save_episode"
    request_id: str


class DiscardEpisodeCommand(BaseModel):
    command: Literal["discard_episode"] = "discard_episode"
    request_id: str


HazardKind = Literal["none", "hand_or_person", "unknown_object", "other"]
StoppableControl = Literal["model", "teleop"]


class MonitorConfig(BaseModel):
    """The vision-language model that watches the cameras and stops the control when it sees a hazard.

    It speaks the OpenAI chat API (vLLM, say) and answers one word: ``SAFE``, ``UNSAFE_HAND`` or
    ``UNSAFE_OBJECT``. See docs/feature_plans/remote-inference-and-safety-monitor.md.
    """

    enabled: bool = False
    endpoint: str | None = None
    """The API base, such as ``http://172.18.0.1:18766/v1``."""
    model: str = "qwen-safety"
    cameras: list[str] = Field(default_factory=list)
    """Store camera keys to show the model, in order; empty shows every camera."""
    check_hz: float = Field(
        default=2.0, gt=0, le=10, description="Checks per second; a check still running is not doubled"
    )
    image_max_side: int = Field(default=448, ge=64, le=1920)
    timeout_s: float = Field(default=3.0, gt=0, le=30)
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    """An unsafe answer below this probability does not count."""
    unsafe_consecutive: int = Field(default=1, ge=1, le=10)
    fail_safe_after_s: float = Field(default=2.0, gt=0, le=60)
    """Stop when no check has succeeded for this long while a control the monitor guards drives."""
    max_frame_age_s: float = Field(default=1.0, gt=0, le=10)
    """A frame older than this means a frozen camera: a failed check, not a safe one."""
    prompt: str | None = None
    """The scene description; ``None`` uses a generic one. The one-word answer instruction is added."""


class SafetyConfig(BaseModel):
    """What the session does when something reports a hazard. Kept per follower robot."""

    monitor: MonitorConfig = Field(default_factory=MonitorConfig)

    on_stop: Literal["home", "hold"] = "home"
    """``home`` moves the follower to ``home_pose`` after stopping it; ``hold`` leaves it where it stopped."""
    home_pose: list[float] | None = None
    """In the follower's joint order and units; ``None`` until captured, and the follower then holds instead."""
    home_duration_s: float = Field(default=4.0, gt=0, le=30, description="The move home takes at least this long")
    max_joint_speed: float = Field(
        45.0, gt=0, description="Joint units per second; a longer move home is slowed down to stay under it"
    )
    stop_controls: list[StoppableControl] = Field(default_factory=lambda: list[StoppableControl](["model"]))
    """The controls a stop ends. Any other control keeps driving, and the follower is not moved."""


class SafetyStopRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)
    source: str = Field("api", min_length=1, max_length=100)
    """Who reported the hazard: ``monitor``, ``gui``, or an external caller's own name."""
    hazard: HazardKind = "other"
    confidence: float | None = Field(None, ge=0, le=1)
    hold: bool = False
    """Freeze the follower where it is instead of moving it home; also interrupts a move home."""


SafetyEventKind = Literal["stop", "acknowledge", "home_captured", "home", "config", "verdict", "monitor_fault"]


class SafetyEvent(BaseModel):
    id: str
    at: UTCDatetime
    kind: SafetyEventKind
    source: str
    hazard: HazardKind | None = None
    reason: str | None = None
    confidence: float | None = None
    stopped_control: StoppableControl | None = None
    """The control a stop ended, if one was running."""
    motion: Literal["home", "hold", "none"] | None = None
    """What the follower was told to do: move home, hold where it is, or nothing (another control drives it)."""
    answer: str | None = None
    """The monitor's one-word answer, for a verdict."""
    latency_ms: float | None = None
    frames: list[str] = Field(default_factory=list)
    """Cameras whose frames were saved with a stop; fetch them from ``/safety/events/{id}/frames/{camera}``."""


MonitorHealth = Literal["off", "ok", "degraded", "down"]


class MonitorState(BaseModel):
    health: MonitorHealth = "off"
    """``degraded``: checks are failing but not for ``fail_safe_after_s`` yet; ``down``: they have been."""
    guarding: bool = False
    """Whether a control the monitor would stop is driving now."""
    last_verdict: SafetyEvent | None = None
    error: str | None = None


SafetyLatchState = Literal["clear", "latched", "homing"]


class SessionSafety(BaseModel):
    state: SafetyLatchState
    """``latched`` blocks starting a stopped control until a person acknowledges; ``homing`` is latched and moving."""
    latched_by: SafetyEvent | None = None
    home_pose_set: bool = False
    monitor: MonitorState = MonitorState()


class SafetyStatus(SessionSafety):
    session_id: str
    config: SafetyConfig
    counters: dict[str, int]


class SafetyMessage(BaseModel):
    """A safety event, as it happens; the session's ``state`` follows with the new latch state."""

    type: Literal["safety"] = "safety"
    data: SafetyEvent


class SafetyStopCommand(SafetyStopRequest):
    command: Literal["safety_stop"] = "safety_stop"
    request_id: str


class SafetyAcknowledgeCommand(BaseModel):
    """A person clears the latch. Nothing restarts on its own."""

    command: Literal["safety_acknowledge"] = "safety_acknowledge"
    request_id: str
    source: str = "gui"


class SetSafetyCommand(BaseModel):
    command: Literal["set_safety"] = "set_safety"
    request_id: str
    config: SafetyConfig


class CaptureHomeCommand(BaseModel):
    """Take the follower's current position as its home pose."""

    command: Literal["capture_home"] = "capture_home"
    request_id: str


class GoHomeCommand(BaseModel):
    """Move the follower to its home pose now; refused while a control drives it."""

    command: Literal["go_home"] = "go_home"
    request_id: str


class SetControlCommand(BaseModel):
    """Switch what drives the follower. ``null`` holds it where it is."""

    command: Literal["set_control"] = "set_control"
    request_id: str
    control: ControlConfig | None


RuntimeSessionCommand = Annotated[
    LoadEnvironmentCommand
    | UnloadEnvironmentCommand
    | ReloadEnvironmentCommand
    | SetControlCommand
    | LoadDatasetCommand
    | UnloadDatasetCommand
    | LoadModelCommand
    | UnloadModelCommand
    | StartEpisodeCommand
    | SaveEpisodeCommand
    | DiscardEpisodeCommand
    | SafetyStopCommand
    | SafetyAcknowledgeCommand
    | SetSafetyCommand
    | CaptureHomeCommand
    | GoHomeCommand,
    Field(discriminator="command"),
]


class IncompatiblePart(BaseModel):
    part: Literal["dataset", "model"]
    name: str
    reasons: list[str]


class SessionError(BaseModel):
    message: str
    error_code: str
    parts: list[IncompatiblePart] | None = None
    """With ``incompatible_parts``: what the command would unload, and why."""


class AckMessage(BaseModel):
    type: Literal["ack"] = "ack"
    request_id: str
    ok: bool
    error: SessionError | None = None
    result: dict[str, Any] | None = None


class SessionCamera(BaseModel):
    camera_id: UUID
    name: str
    key: str
    """The store and frame key; also the key a dataset records this camera under."""
    height: int
    width: int


class SessionEnvironment(BaseModel):
    environment_id: UUID | None
    """The saved environment, or ``None`` for an unsaved one from the form."""
    name: str
    robot_id: UUID
    leader_id: UUID | None
    joint_names: list[str]
    cameras: list[SessionCamera]


class SessionDataset(BaseModel):
    dataset_id: UUID
    name: str
    fps: int
    fps_fixed: bool
    """Whether the dataset already has episodes, and so its own rate; a new one takes the rate it is loaded with."""
    is_recording: bool
    episodes_recorded: int
    """Saved in this session and not yet copied into the dataset, which happens when it is unloaded."""


class SessionModel(BaseModel):
    model_id: UUID
    name: str
    backend: str
    device: str


class SessionState(BaseModel):
    session_id: str
    environment: SessionEnvironment | None = None
    control: ControlConfig | None = None
    """What drives the follower; ``None`` holds it still."""
    dataset: SessionDataset | None = None
    model: SessionModel | None = None
    safety: SessionSafety = SessionSafety(state="clear")


class StateMessage(BaseModel):
    type: Literal["state"] = "state"
    data: SessionState


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    component: str
    message: str
    error_code: str
    session_alive: bool
    """False when the session has ended; the socket closes after this message."""


ComponentState = Literal["starting", "running", "failed", "stopped"]


class StatusMessage(BaseModel):
    """A part of the session or one of its devices saying how it is, on ``evt/status/<component>``.

    ``component`` is a path: ``environment``, ``control`` and ``dataset`` are the
    session's parts, ``environment/follower``, ``environment/leader`` and
    ``environment/<camera key>`` the environment's devices. Devices report only
    ``starting``, ``running`` and ``failed``: a part that starts or stops takes its
    devices with it, so their earlier statuses no longer apply.
    """

    type: Literal["status"] = "status"
    component: str
    state: ComponentState
    error: str | None = None
    """Why it failed."""


SessionEvent = Annotated[StateMessage | ErrorMessage | SafetyMessage, Field(discriminator="type")]
"""What the session publishes on ``evt/session`` for its socket to forward."""


class CommandDone(BaseModel):
    """The reply to a command that has no result."""


class ObservationFrame(BaseModel):
    key: str
    timestamp: float
    size: int
    """Bytes of this JPEG in the message body."""


class ObservationMessage(BaseModel):
    """The header of a binary observation message; ``frames`` lists only frames new since the last one."""

    type: Literal["observation"] = "observation"
    timestamp: float
    state: dict[str, float] | None
    action: dict[str, float] | None
    frames: list[ObservationFrame]


RuntimeSessionServerMessage = Annotated[
    AckMessage | StateMessage | ErrorMessage | StatusMessage | ObservationMessage | SafetyMessage,
    Field(discriminator="type"),
]


class RuntimeSessionInfo(BaseModel):
    """One live runtime session, for the footer and the robot list."""

    session_id: str
    project_id: UUID
    project_name: str
    started_at: UTCDatetime
    environment_name: str | None
    robot_ids: list[UUID]
    camera_ids: list[UUID]


class RuntimeSessionsMessage(BaseModel):
    """Sent by ``/api/runtime/sessions/ws`` on connect and whenever the list changes."""

    sessions: list[RuntimeSessionInfo]
