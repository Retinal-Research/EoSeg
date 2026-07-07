from typing import List, Optional

import numpy as np
from lightning.fabric.utilities import rank_zero_info
from scipy.ndimage import zoom
import torch
import torch.nn as nn

from training.mask_classification_semantic import MaskClassificationSemantic


class ACDCMaskClassificationSemantic(MaskClassificationSemantic):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            ignore_idx=ignore_idx,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            no_object_coefficient=no_object_coefficient,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )
        self.eval_epoch_metrics: list[dict[str, float]] = []

    @staticmethod
    def calculate_metric_percase_baseline(pred: np.ndarray, gt: np.ndarray):
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum() > 0:
            dice = float((2.0 * (pred & gt).sum()) / (pred.sum() + gt.sum()))
            return dice, 0.0, 0.0, 0.0
        if pred.sum() > 0 and gt.sum() == 0:
            return 1.0, 0.0, 1.0, 0.0
        return 0.0, 0.0, 0.0, 0.0

    def _forward_pixel_scores(self, inp: torch.Tensor) -> torch.Tensor:
        mask_logits_per_block, class_logits_per_block = self(inp)
        return self.to_per_pixel_logits_semantic(
            mask_logits_per_block[-1],
            class_logits_per_block[-1],
        )

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
                    pixel_scores = self._forward_pixel_scores(inp)[0]
                    out = torch.argmax(pixel_scores, dim=0).cpu().numpy()

                if x != patch_h or y != patch_w:
                    pred = zoom(out, (x / patch_h, y / patch_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
            return prediction

        inp = torch.from_numpy(img_vol_np).unsqueeze(0).unsqueeze(0).float()
        inp = inp.repeat(1, 3, 1, 1).to(self.device)
        with torch.no_grad():
            pixel_scores = self._forward_pixel_scores(inp)[0]
            return torch.argmax(pixel_scores, dim=0).cpu().numpy()

    def _compute_case_metrics(self, prediction: np.ndarray, gt_vol_np: np.ndarray) -> dict[str, float]:
        metric_list = []
        class_dice = []
        for c in range(1, self.num_classes):
            scores = self.calculate_metric_percase_baseline(prediction == c, gt_vol_np == c)
            metric_list.append(scores)
            class_dice.append(scores[0])

        metric_arr = np.array(metric_list)
        result = {
            "mean_dice": float(metric_arr[:, 0].mean()),
            "mean_hd95": float(metric_arr[:, 1].mean()),
            "mean_jacard": float(metric_arr[:, 2].mean()),
            "mean_asd": float(metric_arr[:, 3].mean()),
        }
        for idx, score in enumerate(class_dice, start=1):
            result[f"dice_class_{idx}"] = float(score)
        return result

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
        return case_metrics

    def on_validation_epoch_end(self):
        if not self.eval_epoch_metrics:
            return

        epoch_metrics = {
            key: float(np.mean([m[key] for m in self.eval_epoch_metrics]))
            for key in self.eval_epoch_metrics[0]
        }

        self.log("metrics/test_mean_dice", epoch_metrics["mean_dice"], prog_bar=True, sync_dist=True, batch_size=1)
        self.log("metrics/test_mean_hd95", epoch_metrics["mean_hd95"], sync_dist=True, batch_size=1)
        self.log("metrics/test_mean_jacard", epoch_metrics["mean_jacard"], sync_dist=True, batch_size=1)
        self.log("metrics/test_mean_asd", epoch_metrics["mean_asd"], sync_dist=True, batch_size=1)
        self.log("metrics/val_dice", epoch_metrics["mean_dice"], sync_dist=True, batch_size=1)

        for c in range(1, self.num_classes):
            key = f"dice_class_{c}"
            if key in epoch_metrics:
                self.log(f"metrics/test_dice_class_{c}", epoch_metrics[key], sync_dist=True, batch_size=1)

        rank_zero_info(
            f"[EPOCH-TEST] mean_dice={epoch_metrics['mean_dice']:.4f} "
            + " ".join(
                [
                    f"class{c}={epoch_metrics[f'dice_class_{c}']:.4f}"
                    for c in range(1, self.num_classes)
                    if f"dice_class_{c}" in epoch_metrics
                ]
            )
        )

    def on_validation_end(self):
        # Disable the parent class semantic mIoU summary hook.
        # ACDC evaluation here follows the old project protocol:
        # test-set Dice logging only, no val_iou_all metric.
        return

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

        for c in range(1, self.num_classes):
            key = f"dice_class_{c}"
            if key in case_metrics:
                self.log(f"test/dice_class_{c}", case_metrics[key], sync_dist=True, batch_size=1)
        return {"case": case, **case_metrics}
