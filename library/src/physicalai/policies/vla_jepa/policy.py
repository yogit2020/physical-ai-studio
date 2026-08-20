# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast

import torch

from physicalai.data.observation import ACTION, STATE, TASK
from physicalai.export import ExportablePolicyMixin
from physicalai.export.backends import ExportBackend
from physicalai.inference.data import InferenceFeature
from physicalai.inference.data.features import InferenceFeatureDtype, InferenceFeatureType
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler

from .config import VlaJepaConfig
from .model import VlaJepaModel

if TYPE_CHECKING:
    from physicalai.data import Observation

logger = logging.getLogger(__name__)


class VlaJepa(ExportablePolicyMixin, Policy):
    """VLA-JEPA Policy.

    Lightning wrapper for fine-tuning and inference with VLA-JEPA
    (Qwen3-VL-2B + FlowmatchingActionHead with <|embodied_action|> tokens).

    Architecture:
      - 32 <|embodied_action|> tokens injected into Qwen3-VL user prompt
      - Qwen hidden states at those positions → vl_embs (B, 32, 2048)
      - FlowmatchingActionHead (DiT-B, 16 layers, Beta noise) denoises actions

    Args:
        qwen_model_name: HF repo ID or local path for Qwen3-VL-2B-Instruct.
        chunk_size: Number of future actions to predict.
        n_action_steps: Number of actions to execute per invocation.
        action_dim: Action dimensionality (6 for SO-ARM 101).
        state_dim: Proprioceptive state dimensionality.
        freeze_qwen_vision: Freeze Qwen3-VL vision tower.
        freeze_qwen_llm: Freeze Qwen3-VL LM backbone (recommended for <500 episodes).
        pretrained_checkpoint: Path to VLA-JEPA-pretrain.pt (empty = random init).
        dataset_stats: For eager initialization (e.g. after checkpoint load).
    """

    model: VlaJepaModel | None

    def __init__(  # noqa: PLR0913
        self,
        qwen_model_name: str = "/checkpoints/Qwen3-VL-2B-Instruct",
        chunk_size: int = 50,
        n_action_steps: int = 50,
        action_dim: int = 6,
        state_dim: int = 6,
        *,
        freeze_qwen_vision: bool = False,
        freeze_qwen_llm: bool = True,
        lr_backbone: float = 1e-5,
        lr_action_head: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-8,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 500,
        scheduler_decay_steps: int = 10_000,
        scheduler_decay_lr: float = 1e-6,
        num_diffusion_steps_train: int = 1000,
        num_diffusion_steps_infer: int = 4,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        image_resize: tuple[int, int] = (224, 224),
        tokenizer_max_length: int = 512,
        enable_gradient_checkpointing: bool = False,
        num_embodied_action_tokens: int = 32,
        pretrained_checkpoint: str = "/checkpoints/vla-jepa-pretrain/Pretrain/checkpoints/VLA-JEPA-pretrain.pt",
        gripper_dim: int = 5,
        gripper_threshold: float = 0.5,
        binarize_gripper_action: bool = False,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(n_action_steps=n_action_steps)

        self.config = VlaJepaConfig(
            qwen_model_name=qwen_model_name,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            state_dim=state_dim,
            freeze_qwen_vision=freeze_qwen_vision,
            freeze_qwen_llm=freeze_qwen_llm,
            lr_backbone=lr_backbone,
            lr_action_head=lr_action_head,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
            num_diffusion_steps_train=num_diffusion_steps_train,
            num_diffusion_steps_infer=num_diffusion_steps_infer,
            noise_beta_alpha=noise_beta_alpha,
            noise_beta_beta=noise_beta_beta,
            noise_s=noise_s,
            image_resize=image_resize,
            tokenizer_max_length=tokenizer_max_length,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            num_embodied_action_tokens=num_embodied_action_tokens,
            pretrained_checkpoint=pretrained_checkpoint,
            gripper_dim=gripper_dim,
            gripper_threshold=gripper_threshold,
            binarize_gripper_action=binarize_gripper_action,
        )

        self.save_hyperparameters(
            {
                "qwen_model_name": qwen_model_name,
                "chunk_size": chunk_size,
                "n_action_steps": n_action_steps,
                "action_dim": action_dim,
                "state_dim": state_dim,
                "freeze_qwen_vision": freeze_qwen_vision,
                "freeze_qwen_llm": freeze_qwen_llm,
                "lr_backbone": lr_backbone,
                "lr_action_head": lr_action_head,
                "optimizer_betas": optimizer_betas,
                "optimizer_eps": optimizer_eps,
                "optimizer_weight_decay": optimizer_weight_decay,
                "optimizer_grad_clip_norm": optimizer_grad_clip_norm,
                "scheduler_warmup_steps": scheduler_warmup_steps,
                "scheduler_decay_steps": scheduler_decay_steps,
                "scheduler_decay_lr": scheduler_decay_lr,
                "num_diffusion_steps_train": num_diffusion_steps_train,
                "num_diffusion_steps_infer": num_diffusion_steps_infer,
                "noise_beta_alpha": noise_beta_alpha,
                "noise_beta_beta": noise_beta_beta,
                "noise_s": noise_s,
                "image_resize": image_resize,
                "tokenizer_max_length": tokenizer_max_length,
                "enable_gradient_checkpointing": enable_gradient_checkpointing,
                "num_embodied_action_tokens": num_embodied_action_tokens,
                "pretrained_checkpoint": pretrained_checkpoint,
                "gripper_dim": gripper_dim,
                "gripper_threshold": gripper_threshold,
                "binarize_gripper_action": binarize_gripper_action,
            }
        )

        self.model: VlaJepaModel | None = None
        self._dataset_stats = dataset_stats
        self._action_steps: int | None = None  # resolved lazily; see _executed_action_steps

        if dataset_stats is not None:
            self._build_model(dataset_stats)

    @staticmethod
    def get_supported_export_backends() -> list[ExportBackend]:
        return [ExportBackend.TORCH]

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        if self._dataset_stats is None:
            return None
        schema: list[InferenceFeature] = [
            InferenceFeature(
                ftype=InferenceFeatureType.STATE,
                shape=(self.config.state_dim,),
                name=STATE,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
            InferenceFeature(
                ftype=InferenceFeatureType.VISUAL,
                shape=(3, *self.config.image_resize),
                name="images",
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            ),
        ]
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        if self._dataset_stats is None:
            return None
        action_shape = cast("tuple", self._dataset_stats.get(ACTION, {}).get("shape", (self.config.action_dim,)))
        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, *action_shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    def _build_model(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        self.model = VlaJepaModel(self.config, dataset_stats)
        self._dataset_stats = dataset_stats
        self.hparams["dataset_stats"] = dataset_stats

    def _load_pretrained_checkpoint(self) -> None:
        """Load weights from VLA-JEPA-pretrain.pt into self.model."""
        from .pretrained_utils import load_vla_jepa_pretrain  # noqa: PLC0415

        ckpt_path = self.config.pretrained_checkpoint
        if not ckpt_path:
            return

        logger.info("Loading pretrained VLA-JEPA weights from %s", ckpt_path)
        loaded, skipped = load_vla_jepa_pretrain(self.model, ckpt_path)
        logger.info("Pretrain load: %d tensors loaded, %d skipped (shape mismatch)", loaded, skipped)

    def setup(self, stage: str) -> None:
        """Lazy initialization — called by Lightning before training."""
        del stage

        from physicalai.data.dataset import Dataset  # noqa: PLC0415
        from physicalai.train.utils import reformat_dataset_to_match_policy  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats = train_dataset.stats
        self.hparams["dataset_stats"] = stats

        # If self.model is already built, this policy was loaded from a Lightning checkpoint
        # via load_from_checkpoint() (dataset_stats was passed to __init__). Skip the
        # pretrained checkpoint load — the fine-tuned weights are already in place.
        loaded_from_checkpoint = self.model is not None

        if self.model is not None:
            self.model.set_dataset_stats(stats)
        else:
            self._build_model(stats)

        if self.config.pretrained_checkpoint and not loaded_from_checkpoint:
            self._load_pretrained_checkpoint()

        reformat_dataset_to_match_policy(self, datamodule)

    # ------------------------------------------------------------------
    # Core forward / predict
    # ------------------------------------------------------------------

    def _obs_to_model_batch(self, batch: Observation) -> dict[str, Any]:
        """Convert physicalai Observation to the dict VlaJepaModel expects."""
        import numpy as np  # noqa: PLC0415
        from PIL import Image as PILImage  # noqa: PLC0415

        obs_dict = batch.to_dict()

        # state: (B, state_dim)
        state = obs_dict.get(f"observation.{STATE}", obs_dict.get(STATE))

        # task strings
        task_raw = obs_dict.get(TASK, ["pick and place"] * state.size(0))
        if isinstance(task_raw, torch.Tensor):
            task = ["pick and place"] * state.size(0)
        else:
            task = list(task_raw)

        # images: extract first camera frame as PIL list
        images_raw = obs_dict.get("observation.images", obs_dict.get("images", None))
        if images_raw is None:
            img_keys = [k for k in obs_dict if "image" in k.lower() or "visual" in k.lower()]
            images_raw = obs_dict.get(img_keys[0]) if img_keys else None

        target_hw = self.config.image_resize  # (H, W), e.g. (224, 224)
        images = []
        if images_raw is not None:
            for i in range(state.size(0)):
                frame = images_raw[i]
                if isinstance(frame, torch.Tensor):
                    frame_np = (frame.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(frame_np)
                else:
                    pil_img = frame
                h, w = target_hw
                if pil_img.size != (w, h):
                    pil_img = pil_img.resize((w, h))
                images.append(pil_img)
        else:
            images = [PILImage.new("RGB", self.config.image_resize)] * state.size(0)

        model_batch: dict[str, Any] = {
            "images": images,
            "task": task,
            "state": state.to(self.device),
        }

        if ACTION in obs_dict:
            model_batch["action"] = obs_dict[ACTION].to(self.device)

        return model_batch

    def forward(
        self, batch: Observation
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if self.model is None:
            msg = "Model not initialized — call setup() or provide dataset_stats"
            raise RuntimeError(msg)

        model_batch = self._obs_to_model_batch(batch)

        if self.training:
            return self.model(model_batch)
        return self.predict_action_chunk(batch)

    def _executed_action_steps(self) -> int:
        """How many chunk steps to emit at inference, clamped to ``chunk_size``.

        ``PHYSICALAI_ACTION_STEPS`` overrides ``n_action_steps`` so the executed
        horizon can be retuned without re-exporting the checkpoint.
        """
        if self._action_steps is not None:
            return self._action_steps

        steps = self.config.n_action_steps
        raw = os.environ.get("PHYSICALAI_ACTION_STEPS")
        if raw is not None:
            try:
                steps = int(raw)
            except ValueError:
                logger.warning("Ignoring non-integer PHYSICALAI_ACTION_STEPS=%r", raw)

        clamped = max(1, min(steps, self.config.chunk_size))
        if clamped != steps:
            logger.warning(
                "Requested %d action steps but chunk_size is %d — clamped to %d",
                steps,
                self.config.chunk_size,
                clamped,
            )

        self._action_steps = clamped
        return clamped

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        if self.model is None:
            msg = "Model not initialized"
            raise RuntimeError(msg)

        model_batch = self._obs_to_model_batch(batch)
        actions = self.model.predict_action_chunk(model_batch)

        # The deployment path calls the policy directly (inference/adapters/pytorch.py
        # does `self._policy(observation)` on an eval()-mode module), so
        # Policy.select_action — and with it the n_action_steps action queue — never
        # runs and the full chunk is emitted regardless of n_action_steps. Truncate
        # here instead, which is what the backend queue mixer actually consumes.
        steps = self._executed_action_steps()
        if steps >= self.config.chunk_size:
            return actions
        return actions[:, :steps] if actions.dim() == 3 else actions[:steps]  # noqa: PLR2004

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        return loss

    def validation_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        del batch_idx
        model_batch = self._obs_to_model_batch(batch)
        loss, loss_dict = self.model(model_batch)
        self.log("val/loss", loss_dict["loss"], prog_bar=True, sync_dist=False)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        if self.model is None:
            msg = "Model not initialized"
            raise RuntimeError(msg)

        cfg = self.config

        # Separate Qwen (backbone, lower LR) from action head (higher LR)
        action_head_params = list(self.model.action_head.parameters())
        action_head_ids = {id(p) for p in action_head_params}

        backbone_params = [p for p in self.parameters() if id(p) not in action_head_ids and p.requires_grad]

        param_groups = [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": [p for p in action_head_params if p.requires_grad], "lr": cfg.lr_action_head},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=cfg.optimizer_betas,
            eps=cfg.optimizer_eps,
            weight_decay=cfg.optimizer_weight_decay,
        )

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=cfg.lr_action_head,
            decay_lr=cfg.scheduler_decay_lr,
            num_warmup_steps=cfg.scheduler_warmup_steps,
            num_decay_steps=cfg.scheduler_decay_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )
