from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from exceptions import RuntimeSessionCommandError
from runtime_session.store import ACTION, STATE
from schemas.runtime_session import (
    ModelControlConfig,
    MonitorConfig,
    SafetyConfig,
    SafetyStopRequest,
    TeleopControlConfig,
)
from tests.workers.fake_model import FakeModelWorker

from .fixtures import make_environment
from .test_model import _model, _until


@pytest.fixture
def fake_model_worker():
    with patch("runtime_session.model.ModelWorker", FakeModelWorker):
        yield


@pytest.fixture(autouse=True)
def own_storage(tmp_path, monkeypatch) -> None:
    """Home poses are kept per follower, and every test's follower has the same id."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))


HAND = SafetyStopRequest(reason="hand above the red bin", source="monitor", hazard="hand_or_person", confidence=0.93)


async def _driving(new_session, tmp_path, *, home_pose: list[float] | None = None):
    """A session whose model drives the follower, with ``home_pose`` set if given."""
    session = await new_session()
    await session.load_environment(make_environment())
    await _until(lambda: session.store.read(STATE) is not None)
    if home_pose is not None:
        await session.set_safety(SafetyConfig(home_pose=home_pose, home_duration_s=0.5))
    await session.load_model(_model(tmp_path), "torch", "cpu")
    await session.set_control(ModelControlConfig(hz=50, grace_s=0))
    return session


async def test_a_hazard_stops_the_model_and_sends_the_follower_home(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session = await _driving(new_session, tmp_path, home_pose=[0.0, 0.0])

    status = await session.safety_stop(HAND)

    assert session.state().control is None
    assert status.state == "homing"
    assert status.latched_by.source == "monitor" and status.latched_by.hazard == "hand_or_person"
    assert (status.latched_by.stopped_control, status.latched_by.motion) == ("model", "home")
    action = session.store.read(ACTION)
    np.testing.assert_allclose(action.value, [0.0, 0.0])
    assert action.timestamp > time.monotonic()  # due ahead: the robot worker glides there
    # Once there, it stays latched.
    await _until(lambda: session.state().safety.state == "latched")


async def test_a_latched_stop_blocks_the_model_until_acknowledged(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session = await _driving(new_session, tmp_path)
    await session.safety_stop(HAND)

    with pytest.raises(RuntimeSessionCommandError, match="hand above the red bin") as refused:
        await session.set_control(ModelControlConfig(hz=50))
    assert refused.value.error_code == "safety_latched"
    # Teleop is not what was stopped, so an operator can still move the arm clear.
    await session.set_control(TeleopControlConfig(grace_s=0))
    await session.set_control(None)

    status = await session.safety_acknowledge("gui")
    assert status.state == "clear"
    assert session.state().control is None  # acknowledging restarts nothing
    await session.set_control(ModelControlConfig(hz=50))
    assert session.state().control.type == "model"


async def test_without_a_home_pose_the_follower_holds(new_session, fake_cameras, fake_model_worker, tmp_path) -> None:
    session = await _driving(new_session, tmp_path)

    status = await session.safety_stop(HAND)

    assert status.state == "latched" and status.latched_by.motion == "hold"
    held = session.store.read(ACTION).value
    np.testing.assert_allclose(held, session.store.read(STATE).value, atol=1.0)


async def test_a_hold_stop_freezes_a_move_home(new_session, fake_cameras, fake_model_worker, tmp_path) -> None:
    session = await _driving(new_session, tmp_path, home_pose=[-50.0, -50.0])
    await session.set_safety(SafetyConfig(home_pose=[-50.0, -50.0], home_duration_s=10))
    await session.safety_stop(HAND)
    assert session.state().safety.state == "homing"

    status = await session.safety_stop(SafetyStopRequest(reason="still a hand", source="monitor", hold=True))

    assert status.state == "latched"
    assert status.latched_by.reason == "hand above the red bin"  # the first stop stays the cause
    assert session.store.read(ACTION).timestamp <= time.monotonic()  # sent at once, not glided
    assert status.counters["stop"] == 2


async def test_a_hazard_while_idle_still_latches(new_session, fake_cameras) -> None:
    session = await new_session()
    with pytest.raises(RuntimeSessionCommandError, match="Load an environment"):
        await session.safety_stop(HAND)
    await session.load_environment(make_environment())

    status = await session.safety_stop(HAND)

    assert status.state == "latched" and status.latched_by.stopped_control is None
    await session.safety_acknowledge()
    assert (await session.safety_acknowledge()).counters["acknowledge"] == 1  # a second one is a no-op


async def test_another_control_keeps_driving_and_the_follower_is_not_moved(new_session, fake_cameras) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.set_control(TeleopControlConfig(grace_s=0))

    status = await session.safety_stop(HAND)

    assert session.state().control.type == "teleop"
    assert status.latched_by.motion == "none"


async def test_the_home_pose_is_kept_for_the_follower(new_session, fake_cameras) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await _until(lambda: session.store.read(STATE) is not None)
    status = await session.capture_home()
    pose = status.config.home_pose
    np.testing.assert_allclose(pose, session.store.read(STATE).value)
    await session.stop()

    later = await new_session()
    await later.load_environment(make_environment())
    assert later.safety_status().config.home_pose == pose
    assert later.state().safety.home_pose_set


async def test_a_new_session_watches_with_the_followers_saved_monitor(new_session, fake_cameras) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.set_safety(SafetyConfig(monitor=MonitorConfig(enabled=True, endpoint="http://qwen.invalid/v1")))
    await session.stop()

    later = await new_session()
    await later.load_environment(make_environment())
    # No safety call yet: the monitor reads this config, so loading the environment must load it.
    assert later._safety.config.monitor.enabled
    assert later._safety.config.monitor.endpoint == "http://qwen.invalid/v1"


async def test_going_home_by_hand(new_session, fake_cameras) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await _until(lambda: session.store.read(STATE) is not None)
    await session.set_safety(SafetyConfig(home_pose=None))
    with pytest.raises(RuntimeSessionCommandError, match="No home pose"):
        await session.go_home()
    await session.set_safety(SafetyConfig(home_pose=[1.0, 2.0], home_duration_s=0.2))
    await session.set_control(TeleopControlConfig(grace_s=0))
    with pytest.raises(RuntimeSessionCommandError, match="Stop the control"):
        await session.go_home()
    await session.set_control(None)

    status = await session.go_home()

    assert status.state == "homing" and status.latched_by is None
    np.testing.assert_allclose(session.store.read(ACTION).value, [1.0, 2.0])
    await _until(lambda: session.state().safety.state == "clear")


async def test_the_move_home_is_slowed_to_the_joint_speed_limit(new_session, fake_cameras) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await _until(lambda: session.store.read(STATE) is not None)
    far = (session.store.read(STATE).value + 100).tolist()
    await session.set_safety(SafetyConfig(home_pose=far, home_duration_s=1.0, max_joint_speed=50.0))

    before = time.monotonic()
    await session.go_home()

    assert session.store.read(ACTION).timestamp - before == pytest.approx(2.0, abs=0.2)  # 100 units at 50/s


async def test_a_home_pose_must_fit_the_follower(new_session, fake_cameras) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    with pytest.raises(ValueError, match="3 joints, the follower has 2"):
        await session.set_safety(SafetyConfig(home_pose=[0.0, 0.0, 0.0]))


async def test_safety_events_are_kept_newest_first(new_session, fake_cameras) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.safety_stop(HAND)
    await session.safety_acknowledge("gui")

    events = session.safety_events()
    assert [event.kind for event in events] == ["acknowledge", "stop"]
    assert [event.kind for event in session.safety_events(stops_only=True)] == ["stop"]
