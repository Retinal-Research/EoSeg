from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.lightning_module import LightningModule
from training.mask_classification_loss import MaskClassificationLoss


class QueryBinaryMaskClassification(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool = False,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 5e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 5e-4,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        metric_threshold: float = 0.5,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ) -> None:
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.metric_threshold = metric_threshold
        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

    @staticmethod
    def _condseg_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        intersection = float((y_true * y_pred).sum())
        return (intersection + 1e-15) / (float(y_pred.sum()) + 1e-15)

    @staticmethod
    def _condseg_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        intersection = float((y_true * y_pred).sum())
        return (intersection + 1e-15) / (float(y_true.sum()) + 1e-15)

    @classmethod
    def _condseg_f2(cls, y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> float:
        precision = cls._condseg_precision(y_true, y_pred)
        recall = cls._condseg_recall(y_true, y_pred)
        return (1 + beta**2) * (precision * recall) / (
            beta**2 * precision + recall + 1e-15
        )

    @staticmethod
    def _condseg_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        intersection = float((y_true * y_pred).sum())
        return (2 * intersection + 1e-15) / (
            float(y_true.sum()) + float(y_pred.sum()) + 1e-15
        )

    @staticmethod
    def _condseg_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        intersection = float((y_true * y_pred).sum())
        union = float(y_true.sum()) + float(y_pred.sum()) - intersection
        return (intersection + 1e-15) / (union + 1e-15)

    def _targets_to_binary_masks(self, targets: list[dict]) -> torch.Tensor:
        binary_masks = []
        for target in targets:
            if target["masks"].numel() == 0:
                height, width = target["masks"].shape[-2:]
                binary_mask = torch.zeros(
                    (height, width),
                    dtype=torch.float32,
                    device=target["labels"].device,
                )
            else:
                binary_mask = target["masks"].any(dim=0).float()
            binary_masks.append(binary_mask)
        return torch.stack(binary_masks, dim=0).unsqueeze(1)

    def _binary_scores_from_outputs(
        self, mask_logits: torch.Tensor, class_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_probs = class_logits.softmax(dim=-1)
        if class_probs.shape[-1] < 2:
            raise ValueError("Binary query setup expects foreground + no-object logits.")

        # Binary query training uses one foreground class (index 0) plus the
        # no-object class (last index). Aggregate the foreground contribution
        # from all queries into a single per-pixel foreground score/probability.
        foreground_query_probs = class_probs[..., 0:1]
        foreground_probs = torch.einsum(
            "bqhw,bqk->bkhw",
            mask_logits.sigmoid(),
            foreground_query_probs,
        )
        foreground_probs = foreground_probs.clamp(1e-6, 1 - 1e-6)
        foreground_logits = torch.logit(foreground_probs)
        return foreground_logits, foreground_probs

    def _batch_metrics(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        preds = (probs >= self.metric_threshold).float().detach().cpu().numpy()
        targets = (targets > 0.5).float().detach().cpu().numpy()

        jaccard_scores = []
        dice_scores = []
        recall_scores = []
        precision_scores = []
        acc_scores = []
        f2_scores = []

        for pred, target in zip(preds, targets):
            pred = pred.reshape(-1).astype(np.uint8)
            target = target.reshape(-1).astype(np.uint8)

            jaccard_scores.append(self._condseg_jaccard(target, pred))
            dice_scores.append(self._condseg_dice(target, pred))
            recall_scores.append(self._condseg_recall(target, pred))
            precision_scores.append(self._condseg_precision(target, pred))
            acc_scores.append(float(np.mean(target == pred)))
            f2_scores.append(self._condseg_f2(target, pred))

        return {
            "jaccard": torch.tensor(np.mean(jaccard_scores), device=self.device, dtype=torch.float32),
            "dice": torch.tensor(np.mean(dice_scores), device=self.device, dtype=torch.float32),
            "recall": torch.tensor(np.mean(recall_scores), device=self.device, dtype=torch.float32),
            "precision": torch.tensor(np.mean(precision_scores), device=self.device, dtype=torch.float32),
            "acc": torch.tensor(np.mean(acc_scores), device=self.device, dtype=torch.float32),
            "f2": torch.tensor(np.mean(f2_scores), device=self.device, dtype=torch.float32),
        }

    def _combine_query_losses(self, losses_all_blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for loss_key, loss in losses_all_blocks.items():
            if "mask" in loss_key:
                weighted = loss * self.criterion.mask_coefficient
            elif "dice" in loss_key:
                weighted = loss * self.criterion.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted = loss * self.criterion.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")
            total = weighted if total is None else total + weighted
        if total is None:
            raise RuntimeError("No losses were produced for query-based binary training.")
        return total

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch
        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        total_loss = self._combine_query_losses(losses_all_blocks)
        batch_size = imgs.shape[0]
        self.log(
            f"{log_prefix}/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        last_mask_logits = mask_logits_per_block[-1]
        last_class_logits = class_logits_per_block[-1]
        _, foreground_probs = self._binary_scores_from_outputs(
            last_mask_logits, last_class_logits
        )
        binary_targets = self._targets_to_binary_masks(targets).to(foreground_probs.device)
        metrics = self._batch_metrics(foreground_probs, binary_targets)

        self.log(
            f"metrics/{log_prefix}_dice",
            metrics["dice"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"metrics/{log_prefix}_iou",
            metrics["jaccard"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"metrics/{log_prefix}_jaccard",
            metrics["jaccard"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"metrics/{log_prefix}_recall",
            metrics["recall"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"metrics/{log_prefix}_precision",
            metrics["precision"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"metrics/{log_prefix}_acc",
            metrics["acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"metrics/{log_prefix}_f2",
            metrics["f2"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
