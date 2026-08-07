import os

from loguru import logger
from physicalai.data import Observation

from control.inference_poller import InferencePoller
from control.queue_mixer import QueueMixer
from workers.model_worker import ModelWorker


def _action_steps_limit() -> int | None:
    """Steps of each predicted chunk to execute before re-running inference.

    A policy's action horizon is fixed in its exported graph (Pi0.5 emits 50),
    so this trims the chunk on execution. Fewer steps means the policy
    re-observes sooner, which matters on real hardware where the tail of a long
    chunk goes stale --- at 30 fps, 50 steps commits the arm to 1.67 s of motion
    before it looks at the cameras again.

    Prefer setting ``n_action_steps`` on the policy so export emits an
    ``action_chunk_trimmer`` postprocessor instead; this override exists for
    models whose manifest was produced without one, so the horizon can be tried
    without re-exporting.

    Unset or non-positive keeps the model's full horizon.
    """
    raw = os.environ.get("PHYSICALAI_ACTION_STEPS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer PHYSICALAI_ACTION_STEPS={!r}", raw)
        return None
    if value <= 0:
        return None
    return value


class SyncMixedModelIntegration:
    model_worker: ModelWorker
    queue_mixer: QueueMixer
    inference_poller: InferencePoller
    fps: int
    use_synchronous: bool = True

    def __init__(self, model_worker: ModelWorker, fps: int):
        self.model_worker = model_worker
        self.fps = fps
        self.action_steps = _action_steps_limit()

        # Communication layer to model worker. It ensures no queue.
        self.inference_poller = InferencePoller(self.model_worker.observation_queue, self.model_worker.output_queue)

        # Queue mixer to move to new inference result while still executing previous.
        self.queue_mixer = QueueMixer(lerp_duration=self.fps)

    def select_action(self, observation: Observation) -> list[list[float]] | None:
        if self.inference_poller.has_result():
            inference_result = self.inference_poller.get_result()
            offset = int(inference_result.time * self.fps)
            actions = inference_result.data
            if self.action_steps is not None:
                # Slice plainly: the mixer only applies ``offset`` when blending
                # into a still-draining queue, and keeps the first chunk of an
                # episode whole. Adding ``offset`` here would execute more than
                # the requested number of steps on that first chunk.
                actions = actions[: max(self.action_steps, 1)]
            self.queue_mixer.add(actions, offset)
            self.queue_mixer.lerp_duration = max(offset, 1)

        # if self.use_synchronous we wait for the queue_mixer to empty first.
        # else just send inference when its no longer busy.
        synchronous = self.queue_mixer.empty() if self.use_synchronous else True

        if synchronous and not self.inference_poller.busy:
            self.inference_poller.run_inference(observation)

        if not self.queue_mixer.empty():
            return self.queue_mixer.pop().tolist()

        return None

    def reset(self) -> None:
        self.inference_poller.reset()

    async def setup(self) -> None:
        # Worker is already running (pre-spawned); just wait for model to load.
        if self.action_steps is not None:
            logger.info("Executing {} action steps per chunk (PHYSICALAI_ACTION_STEPS)", self.action_steps)
        await self.model_worker.wait_for_loading_to_complete()

    def teardown(self) -> None:
        self.inference_poller.reset()
