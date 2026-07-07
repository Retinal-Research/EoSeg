from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.lightning_module import LightningModule


class WeightedBCE(nn.Module):
    def __init__(self, weights: tuple[float, float] = (0.5, 0.5)) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pos = (targets > 0.5).float()
        neg = (targets < 0.5).float()

        pos_weight = pos.sum().clamp_min(1e-12)
        neg_weight = neg.sum().clamp_min(1e-12)

        return (
            self.weights[0] * pos * loss / pos_weight
            + self.weights[1] * neg * loss / neg_weight
        ).sum()


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights: tuple[float, float] = (0.5, 0.5)) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-5,
    ) -> torch.Tensor:
        batch_size = probs.shape[0]
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        weights = targets.detach()
        weights = weights * (self.weights[1] - self.weights[0]) + self.weights[0]

        probs = weights * probs
        targets = weights * targets

        intersection = (probs * targets).sum(-1)
        union = (probs * probs).sum(-1) + (targets * targets).sum(-1)
        dice = 1 - (2 * intersection + smooth) / (union + smooth)

        return dice.mean()


class WeightedDiceBCE(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.6,
        bce_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.bce_loss = WeightedBCE()
        self.dice_loss = WeightedDiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        dice = self.dice_loss(probs, targets)
        bce = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce


class MedicalBinarySegmentation(LightningModule):
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
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        metric_threshold: float = 0.5,
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
        self.criterion = WeightedDiceBCE(dice_weight=0.6, bce_weight=0.4)

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

    def _foreground_logits_and_probs(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self(imgs)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            first, second = outputs

            if isinstance(first, list) and isinstance(second, list):
                mask_logits = first[-1]
                class_logits = second[-1]
                per_pixel_scores = self.to_per_pixel_logits_semantic(
                    mask_logits, class_logits
                )

                if per_pixel_scores.shape[1] < 2:
                    raise ValueError("Binary semantic setup expects at least 2 classes.")

                # `to_per_pixel_logits_semantic` returns query-aggregated class scores,
                # not a guaranteed probability map. For binary training we therefore
                # recover foreground probability via class-channel softmax, and use
                # the fg-bg score difference as the BCE logit.
                foreground_logits = per_pixel_scores[:, 1:2] - per_pixel_scores[:, 0:1]
                foreground = torch.softmax(per_pixel_scores, dim=1)[:, 1:2]
            elif torch.is_tensor(first):
                logits = first
                if logits.shape[1] == 1:
                    foreground_logits = logits
                    foreground = torch.sigmoid(logits)
                else:
                    foreground_logits = logits[:, 1:2] - logits[:, 0:1]
                    foreground = torch.sigmoid(foreground_logits)
            else:
                raise TypeError("Unsupported network output format.")
        elif torch.is_tensor(outputs):
            if outputs.shape[1] == 1:
                foreground_logits = outputs
                foreground = torch.sigmoid(outputs)
            else:
                foreground_logits = outputs[:, 1:2] - outputs[:, 0:1]
                foreground = torch.sigmoid(foreground_logits)
        else:
            raise TypeError("Unsupported network output format.")

        if foreground.shape[-2:] != imgs.shape[-2:]:
            foreground_logits = F.interpolate(
                foreground_logits,
                size=imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            foreground = F.interpolate(
                foreground,
                size=imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return foreground_logits, foreground.clamp(0.0, 1.0)

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

        return (
            {
                "jaccard": torch.tensor(
                    np.mean(jaccard_scores), device=self.device, dtype=torch.float32
                ),
                "dice": torch.tensor(
                    np.mean(dice_scores), device=self.device, dtype=torch.float32
                ),
                "recall": torch.tensor(
                    np.mean(recall_scores), device=self.device, dtype=torch.float32
                ),
                "precision": torch.tensor(
                    np.mean(precision_scores), device=self.device, dtype=torch.float32
                ),
                "acc": torch.tensor(
                    np.mean(acc_scores), device=self.device, dtype=torch.float32
                ),
                "f2": torch.tensor(
                    np.mean(f2_scores), device=self.device, dtype=torch.float32
                ),
            }
        )

    def _shared_step(self, batch, log_prefix: str):
        imgs, targets = batch
        foreground_logits, foreground_probs = self._foreground_logits_and_probs(imgs)
        binary_targets = self._targets_to_binary_masks(targets).to(foreground_probs.device)

        loss = self.criterion(foreground_logits, foreground_probs, binary_targets)
        metrics = self._batch_metrics(foreground_probs, binary_targets)

        batch_size = imgs.shape[0]
        self.log(
            f"{log_prefix}/loss",
            loss,
            on_step=(log_prefix == "train"),
            on_epoch=True,
            prog_bar=(log_prefix != "train"),
            sync_dist=True,
            batch_size=batch_size,
        )
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

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx=0):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")
