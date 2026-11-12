"""Watch and stop the runtime sessions of this API process, and drive their safety latch.

Not project-scoped: the footer shows every session, because a session in one
project can hold a camera another project wants. The ``/safety`` routes let
anything that watches the robot (the VLM monitor, a light curtain gateway, a
script) stop a session's control for a hazard, as the GUI does over the session
socket. See docs/feature_plans/remote-inference-and-safety-monitor.md.
"""

from typing import Annotated

import anyio
from fastapi import APIRouter, Query, WebSocket, status
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect

from api.dependencies import RuntimeSessionRegistryDep
from exceptions import ResourceNotFoundError, ResourceType
from runtime_session.registry import RuntimeSessionRegistry
from runtime_session.session import RuntimeSession
from schemas.runtime_session import RuntimeSessionsMessage, SafetyConfig, SafetyEvent, SafetyStatus, SafetyStopRequest

router = APIRouter(prefix="/api/runtime/sessions", tags=["Runtime Sessions"])


@router.get(
    "/ws",
    tags=["WebSocket"],
    summary="Live runtime sessions (WebSocket)",
    status_code=426,
    responses={200: {"model": RuntimeSessionsMessage, "description": "Sent on connect and on every change"}},
)
async def runtime_sessions_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def runtime_sessions_websocket(websocket: WebSocket, registry: RuntimeSessionRegistryDep) -> None:  # noqa: D103 - documented on the GET stub
    await websocket.accept()
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(_send_on_change, websocket, registry)
            tasks.start_soon(_until_closed, websocket)
    except* WebSocketDisconnect:
        pass


async def _until_closed(websocket: WebSocket) -> None:
    """The client sends nothing; receiving is how a closed socket is noticed while idle."""
    while True:
        await websocket.receive_text()


async def _send_on_change(websocket: WebSocket, registry: RuntimeSessionRegistry) -> None:
    with registry.watch() as changes:
        while True:
            message = RuntimeSessionsMessage(sessions=[session.info() for session in registry.sessions()])
            await websocket.send_json(message.model_dump(mode="json"))
            await changes.get()


@router.post("/{session_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_runtime_session(session_id: str, registry: RuntimeSessionRegistryDep) -> None:
    """Close a session, releasing its devices. Its websocket gets a final ``session_closed`` error."""
    await _session(registry, session_id).stop()


def _session(registry: RuntimeSessionRegistry, session_id: str) -> RuntimeSession:
    session = registry.get(session_id)
    if session is None:
        raise ResourceNotFoundError(ResourceType.RUNTIME_SESSION, session_id)
    return session


@router.get("/{session_id}/safety", tags=["Safety"])
async def get_safety(session_id: str, registry: RuntimeSessionRegistryDep) -> SafetyStatus:
    """The session's safety latch, its configuration for the loaded follower, and event counts."""
    return _session(registry, session_id).safety_status()


@router.put("/{session_id}/safety", tags=["Safety"])
async def set_safety(session_id: str, config: SafetyConfig, registry: RuntimeSessionRegistryDep) -> SafetyStatus:
    """Set what a stop does for the loaded follower; kept for that follower across sessions."""
    try:
        return await _session(registry, session_id).set_safety(config, source="api")
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.post("/{session_id}/safety/stop", tags=["Safety"])
async def safety_stop(session_id: str, request: SafetyStopRequest, registry: RuntimeSessionRegistryDep) -> SafetyStatus:
    """Stop the control for a hazard, move the follower home (or hold it), and latch.

    Idempotent: a stop while latched is recorded and keeps the first stop as the cause.
    ``409`` without a loaded environment.
    """
    return await _session(registry, session_id).safety_stop(request)


@router.post("/{session_id}/safety/acknowledge", tags=["Safety"])
async def safety_acknowledge(
    session_id: str, registry: RuntimeSessionRegistryDep, source: Annotated[str, Query(max_length=100)] = "api"
) -> SafetyStatus:
    """A person clears the latch. Nothing restarts on its own."""
    return await _session(registry, session_id).safety_acknowledge(source)


@router.get("/{session_id}/safety/events", tags=["Safety"])
async def safety_events(
    session_id: str,
    registry: RuntimeSessionRegistryDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    stops_only: bool = False,
) -> list[SafetyEvent]:
    """The session's safety events, newest first."""
    return _session(registry, session_id).safety_events(limit=limit, stops_only=stops_only)


@router.post("/{session_id}/safety/home/capture", tags=["Safety"])
async def capture_home(session_id: str, registry: RuntimeSessionRegistryDep) -> SafetyStatus:
    """Take the follower's current position as its home pose."""
    return await _session(registry, session_id).capture_home(source="api")


@router.post("/{session_id}/safety/home/go", tags=["Safety"])
async def go_home(session_id: str, registry: RuntimeSessionRegistryDep) -> SafetyStatus:
    """Move the follower to its home pose now. ``409`` while a control drives it, or with no home pose."""
    return await _session(registry, session_id).go_home(source="api")
