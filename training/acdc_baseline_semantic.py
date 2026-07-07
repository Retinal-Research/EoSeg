from typing import List, Optional

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities import rank_zero_info
from medpy import metric
from scipy.ndimage import zoom

from training.lightning_module import LightningModule


class BaselineDiceLoss(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tensor_list.append((input_tensor == i).unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes


class ACDCBaselineSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool = False,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 1e-4,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        ce_weight: float = 0.3,
        dice_weight: float = 0.7,
    ):
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
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.NLLLoss(ignore_index=ignore_idx)
        self.dice_loss = BaselineDiceLoss(num_classes)
        self.eval_epoch_metrics: list[dict[str, float]] = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _forward_probs(self, imgs: torch.Tensor) -> torch.Tensor:
        mask_logits_per_layer, class_logits_per_layer = self(imgs)
        probs = self.to_per_pixel_logits_semantic(
            mask_logits_per_layer[-1],
            class_logits_per_layer[-1],
        )
        probs = probs.clamp_min(1e-6)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return probs.contiguous()

    def _batch_to_imgs_targets(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]

        if not isinstance(batch, dict):
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        imgs = batch["image"]
        labels = batch["label"]

        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(1)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        targets = []
        case_names = batch.get("case_name")
        for i in range(labels.shape[0]):
            cls_ids = torch.unique(labels[i])
            masks = []
            cls_list = []
            for cls_id in cls_ids:
                cls_val = int(cls_id.item())
                if cls_val == self.ignore_idx:
                    continue
                masks.append(labels[i] == cls_id)
                cls_list.append(cls_val)

            if masks:
                target = {
                    "masks": torch.stack(masks).bool(),
                    "labels": torch.tensor(cls_list, dtype=torch.long, device=labels.device),
                    "is_crowd": torch.zeros(len(cls_list), dtype=torch.bool, device=labels.device),
                }
            else:
                h, w = labels[i].shape[-2:]
                target = {
                    "masks": torch.zeros((0, h, w), dtype=torch.bool, device=labels.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=labels.device),
                    "is_crowd": torch.zeros((0,), dtype=torch.bool, device=labels.device),
                }

            if case_names is not None:
                target["case_name"] = case_names[i] if isinstance(case_names, (list, tuple)) else case_names
            targets.append(target)

        return imgs, targets

    def training_step(self, batch, batch_idx):
        imgs, targets = self._batch_to_imgs_targets(batch)
        batch_size = imgs.shape[0]
        probs = self._forward_probs(imgs).contiguous()
        targets_pp = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        targets_pp = torch.stack(targets_pp).to(probs.device).contiguous()
        log_probs = torch.log(probs).contiguous()

        loss_ce = self.ce_loss(log_probs, targets_pp.long().contiguous())
        loss_dice = self.dice_loss(probs, targets_pp, softmax=False)
        loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        if batch_idx == 0 and (self.current_epoch == 0 or self.current_epoch % 10 == 0):
            pred = torch.argmax(probs, dim=1)
            logging.info(
                "[ACDC DEBUG][train] epoch=%d img_range=(%.4f, %.4f) "
                "target_unique=%s pred_unique=%s prob_max=%.4f prob_mean=%.4f",
                int(self.current_epoch),
                float(imgs.min().item()),
                float(imgs.max().item()),
                torch.unique(targets_pp).detach().cpu().tolist(),
                torch.unique(pred).detach().cpu().tolist(),
                float(probs.max().item()),
                float(probs.mean().item()),
            )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train/loss_ce", loss_ce, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/loss_dice", loss_dice, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    @staticmethod
    def calculate_metric_percase_baseline(pred: np.ndarray, gt: np.ndarray):
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum() > 0:
            return float(metric.binary.dc(pred, gt)), 0.0, 0.0, 0.0
        if pred.sum() > 0 and gt.sum() == 0:
            return 1.0, 0.0, 1.0, 0.0
        return 0.0, 0.0, 0.0, 0.0

    def _run_case_inference(self, img_vol_np: np.ndarray) -> np.ndarray:
        patch_h, patch_w = self.img_size

        if len(img_vol_np.shape) == 3:
            prediction = np.zeros_like(img_vol_np, dtype=np.int32)
            for ind in range(img_vol_np.shape[0]):
                slc = img_vol_np[ind]
                x, y = slc.shape
                if x != patch_h or y != patch_w:
                    slc = zoom(slc, (patch_h / x, patch_w / y), order=3)

                inp = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float()
                inp = inp.repeat(1, 3, 1, 1).to(self.device)

                with torch.no_grad():
                    probs = self._forward_probs(inp)[0]
                    out = torch.argmax(probs, dim=0).cpu().numpy()

                if x != patch_h or y != patch_w:
                    pred = zoom(out, (x / patch_h, y / patch_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
            return prediction

        inp = torch.from_numpy(img_vol_np).unsqueeze(0).unsqueeze(0).float()
        inp = inp.repeat(1, 3, 1, 1).to(self.device)
        with torch.no_grad():
            probs = self._forward_probs(inp)[0]
            return torch.argmax(probs, dim=0).cpu().numpy()

    def _compute_case_metrics(self, prediction: np.ndarray, gt_vol_np: np.ndarray) -> dict[str, float]:
        metric_list = []
        for c in range(1, self.num_classes):
            metric_list.append(
                self.calculate_metric_percase_baseline(prediction == c, gt_vol_np == c)
            )

        metric_arr = np.array(metric_list)
        return {
            "mean_dice": float(metric_arr[:, 0].mean()),
            "mean_hd95": float(metric_arr[:, 1].mean()),
            "mean_jacard": float(metric_arr[:, 2].mean()),
            "mean_asd": float(metric_arr[:, 3].mean()),
        }

    def on_validation_epoch_start(self):
        self.eval_epoch_metrics = []

    def validation_step(self, batch, batch_idx=0):
        case = batch["case_name"]
        img_vol = batch["image"]
        gt_vol = batch["label"]

        if img_vol.dim() == 4:
            img_vol = img_vol[0]
        if gt_vol.dim() == 4:
            gt_vol = gt_vol[0]

        img_vol_np = img_vol.cpu().numpy()
        gt_vol_np = gt_vol.cpu().numpy().astype(np.int32)
        prediction = self._run_case_inference(img_vol_np)
        case_metrics = self._compute_case_metrics(prediction, gt_vol_np)
        self.eval_epoch_metrics.append(case_metrics)

        if batch_idx == 0 and (self.current_epoch == 0 or self.current_epoch % 10 == 0):
            logging.info(
                "[ACDC DEBUG][eval] epoch=%d case=%s input_range=(%.4f, %.4f) "
                "gt_unique=%s pred_unique=%s",
                int(self.current_epoch),
                str(case),
                float(img_vol.min().item()),
                float(img_vol.max().item()),
                np.unique(gt_vol_np).tolist(),
                np.unique(prediction).tolist(),
            )

        rank_zero_info(
            f"[EPOCH-TEST] {case} mean_dice={case_metrics['mean_dice']:.4f} "
            f"mean_hd95={case_metrics['mean_hd95']:.4f} "
            f"mean_jacard={case_metrics['mean_jacard']:.4f} "
            f"mean_asd={case_metrics['mean_asd']:.4f}"
        )
        return case_metrics

    def on_validation_epoch_end(self):
        if not self.eval_epoch_metrics:
            epoch_metrics = {
                "mean_dice": 0.0,
                "mean_hd95": 0.0,
                "mean_jacard": 0.0,
                "mean_asd": 0.0,
            }
        else:
            epoch_metrics = {
                key: float(np.mean([m[key] for m in self.eval_epoch_metrics]))
                for key in self.eval_epoch_metrics[0]
            }

        self.log("metrics/test_mean_dice", epoch_metrics["mean_dice"], prog_bar=True, sync_dist=True, batch_size=1)
        self.log("metrics/test_mean_hd95", epoch_metrics["mean_hd95"], sync_dist=True, batch_size=1)
        self.log("metrics/test_mean_jacard", epoch_metrics["mean_jacard"], sync_dist=True, batch_size=1)
        self.log("metrics/test_mean_asd", epoch_metrics["mean_asd"], sync_dist=True, batch_size=1)
        # Compatibility alias for older ACDC configs that still monitor val_dice.
        self.log("metrics/val_dice", epoch_metrics["mean_dice"], sync_dist=True, batch_size=1)

        rank_zero_info(
            f"[EPOCH-TEST] mean_dice={epoch_metrics['mean_dice']:.4f} "
            f"mean_hd95={epoch_metrics['mean_hd95']:.4f} "
            f"mean_jacard={epoch_metrics['mean_jacard']:.4f} "
            f"mean_asd={epoch_metrics['mean_asd']:.4f}"
        )

    def test_step(self, batch, batch_idx):
        case = batch["case_name"]
        img_vol = batch["image"]
        gt_vol = batch["label"]

        if img_vol.dim() == 4:
            img_vol = img_vol[0]
        if gt_vol.dim() == 4:
            gt_vol = gt_vol[0]

        img_vol_np = img_vol.cpu().numpy()
        gt_vol_np = gt_vol.cpu().numpy().astype(np.int32)
        prediction = self._run_case_inference(img_vol_np)
        case_metrics = self._compute_case_metrics(prediction, gt_vol_np)

        self.log("test/mean_dice", case_metrics["mean_dice"], prog_bar=True, sync_dist=True, batch_size=1)
        self.log("test/mean_hd95", case_metrics["mean_hd95"], sync_dist=True, batch_size=1)
        self.log("test/mean_jacard", case_metrics["mean_jacard"], sync_dist=True, batch_size=1)
        self.log("test/mean_asd", case_metrics["mean_asd"], sync_dist=True, batch_size=1)

        rank_zero_info(
            f"[TEST] {case} mean_dice={case_metrics['mean_dice']:.4f} "
            f"mean_hd95={case_metrics['mean_hd95']:.4f} "
            f"mean_jacard={case_metrics['mean_jacard']:.4f} "
            f"mean_asd={case_metrics['mean_asd']:.4f}"
        )
        return {
            "case": case,
            **case_metrics,
        }
