import asyncio
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from physicalai.capture import SharedCamera

from control.environment_integration import EnvironmentIntegration


def _make_mock_camera():
    camera = MagicMock(spec=SharedCamera)
    camera.connect = Mock()
    camera.disconnect = Mock()
    mock_frame = MagicMock()
    mock_frame.data = np.zeros([480, 640, 3], dtype=np.uint8)
    camera.read_latest = Mock(return_value=mock_frame)
    return camera


@pytest.fixture
def mock_camera():
    return _make_mock_camera()


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def inference_environment_integration(event_loop, mock_robot_client_factory, mock_camera, test_environment):
    factory = mock_robot_client_factory

    with patch(
        "control.environment_integration.build_shared_camera",
        return_value=mock_camera,
    ) as mock_build:
        subject = EnvironmentIntegration(test_environment, factory)
        event_loop.run_until_complete(subject.setup())
        for call_args in mock_build.call_args_list:
            assert call_args.kwargs.get("validate_on_connect") is True
            assert call_args.kwargs.get("overwrite_settings") is True
        yield subject
        event_loop.run_until_complete(subject.teardown())


class TestInferenceEnvironmentIntegration:
    def test_get_observation(self, inference_environment_integration: EnvironmentIntegration, event_loop):
        observation = event_loop.run_until_complete(inference_environment_integration.get_observation())
        assert observation is not None
        assert "shoulder_pan.pos" in observation
        assert "3ed60255-04ae-407b-8e2c-c3281847a4e0" in observation  # camera id 1
        assert "4629e172-2aa7-4fde-86b1-e19eb1d210ff" in observation  # camera id 2

    def test_transform_observation_to_model_input(
        self, inference_environment_integration: EnvironmentIntegration, event_loop
    ):
        observation = event_loop.run_until_complete(inference_environment_integration.get_observation())
        assert observation is not None
        phy_ai_obs = inference_environment_integration.format_model_input_observation(observation)
        assert phy_ai_obs.state is not None
        assert phy_ai_obs.state.shape == torch.Size([1, 6])
        assert phy_ai_obs.images is not None
        assert "front" in phy_ai_obs.images
        assert "grabber" in phy_ai_obs.images

    def test_state_values_follow_action_keys_order_not_dict_insertion_order(
        self, inference_environment_integration: EnvironmentIntegration
    ):
        action_keys = inference_environment_integration.action_keys
        unique_values = {key: float(i) for i, key in enumerate(action_keys)}

        # Build observation with joint keys in reversed insertion order — differs from action_keys order.
        # The old code (iterating observation.items()) would have produced reversed state values.
        observation = {
            **{key: unique_values[key] for key in reversed(action_keys)},
            "3ed60255-04ae-407b-8e2c-c3281847a4e0": np.zeros([480, 640, 3], dtype=np.uint8),
            "4629e172-2aa7-4fde-86b1-e19eb1d210ff": np.zeros([480, 640, 3], dtype=np.uint8),
        }

        result = inference_environment_integration.format_model_input_observation(observation)

        expected = [unique_values[k] for k in action_keys]
        np.testing.assert_array_equal(result.state[0], expected)

    def test_model_input_images_follow_camera_name_order_not_db_order(
        self, inference_environment_integration: EnvironmentIntegration, event_loop
    ):
        """Image slots must be ordered by camera name, matching dataset feature order.

        Policies like Pi0.5 stack images positionally and drop the names, so a
        mismatch here feeds each camera into the slot the other one trained as.
        The fixture lists "grabber" before "front" in database order, which is
        the reverse of their sorted order — so iterating the raw list would put
        the wrong camera in slot 0.
        """
        observation = event_loop.run_until_complete(inference_environment_integration.get_observation())
        assert observation is not None

        result = inference_environment_integration.format_model_input_observation(observation)

        assert list(result.images.keys()) == ["front", "grabber"]

    def test_dataset_image_features_follow_camera_name_order(
        self, inference_environment_integration: EnvironmentIntegration
    ):
        """Recorded datasets must declare image features in the same canonical order.

        Training reads slot order from these feature names, so recording in a
        different order than inference builds its inputs would reintroduce the
        same crossed-inputs bug for every dataset captured here.
        """
        features = inference_environment_integration.build_lerobot_dataset_features()

        image_keys = [key for key in features if key.startswith("observation.images.")]
        assert image_keys == ["observation.images.front", "observation.images.grabber"]

    def test_transform_observation_to_report_to_ui(
        self, inference_environment_integration: EnvironmentIntegration, event_loop
    ):
        observation = event_loop.run_until_complete(inference_environment_integration.get_observation())
        assert observation is not None
        report_obs = inference_environment_integration.format_observation_for_reporting(observation, 0)
        assert "shoulder_pan.pos" in report_obs["state"]
        assert "3ed60255-04ae-407b-8e2c-c3281847a4e0" in report_obs["cameras"]  # camera id 1
        assert "4629e172-2aa7-4fde-86b1-e19eb1d210ff" in report_obs["cameras"]  # camera id 2

    def test_teardown_disconnects_robot_and_stops_cameras(
        self,
        inference_environment_integration,
        mock_robot_client,
        mock_camera,
        event_loop,
    ):
        event_loop.run_until_complete(inference_environment_integration.teardown())
        mock_robot_client.disconnect.assert_called_once()
        assert mock_camera.disconnect.call_count == 2  # one per camera
