from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom
from medpy import metric
from lightning.fabric.utilities import rank_zero_info

from training.lightning_module import LightningModule


class DiceCELoss(nn.Module):
    def __init__(self, ignore_idx=255):
        super().__init__()
        self.ignore_idx = ignore_idx
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_idx)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        valid_mask = (targets != self.ignore_idx).unsqueeze(1)
        targets_safe = torch.clamp(targets, 0, num_classes - 1)

        targets_one_hot = (
            F.one_hot(targets_safe, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        dice_loss = 0.0
        for c in range(1, num_classes):
            p_c = probs[:, c, :, :].reshape(-1)
            t_c = targets_one_hot[:, c, :, :].reshape(-1)

            intersection = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum()

            dice_score = (2.0 * intersection + 1e-5) / (union + 1e-5)
            dice_loss += (1.0 - dice_score)

        dice_loss = dice_loss / (num_classes - 1)

        return 0.5 * ce_loss + 0.5 * dice_loss


class PixelClassificationSemantic(LightningModule):
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
        weight_decay: float = 0.05,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
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
        self.criterion = DiceCELoss(ignore_idx=ignore_idx)
        self.init_metrics_semantic(ignore_idx, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits, _ = self.network(imgs)
        targets_res = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        if isinstance(targets_res, list):
            try:
                targets_res = torch.stack(targets_res).to(logits.device)
            except RuntimeError:
                targets_res = [
                    F.interpolate(
                        t.unsqueeze(0).unsqueeze(0).float(),
                        size=logits.shape[-2:],
                        mode="nearest",
                    ).long().squeeze()
                    for t in targets_res
                ]
                targets_res = torch.stack(targets_res).to(logits.device)

        loss = self.criterion(logits, targets_res)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

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
        D, H, W = img_vol_np.shape

        patch_h, patch_w = self.img_size

        pred_vol = np.zeros((D, H, W), dtype=np.int32)

        self.eval()
        with torch.no_grad():
            for z in range(D):
                slc = img_vol_np[z]
                x, y = slc.shape

                if (x != patch_h) or (y != patch_w):
                    slc_rs = zoom(slc, (patch_h / x, patch_w / y), order=3)
                else:
                    slc_rs = slc

                inp = torch.from_numpy(slc_rs).float().unsqueeze(0).unsqueeze(0)
                inp = inp.repeat(1, 3, 1, 1).to(self.device)

                logits, _ = self(inp)
                per_pix = logits[0]

                if per_pix.shape[-2:] != (patch_h, patch_w):
                    per_pix = torch.nn.functional.interpolate(
                        per_pix.unsqueeze(0),
                        size=(patch_h, patch_w),
                        mode="bilinear",
                        align_corners=False
                    )[0]

                pred_patch = torch.argmax(per_pix, dim=0).to(torch.int32).cpu().numpy()

                if (x != patch_h) or (y != patch_w):
                    pred_hw = zoom(pred_patch, (x / patch_h, y / patch_w), order=0)
                else:
                    pred_hw = pred_patch

                pred_vol[z] = pred_hw

        metric_list = []
        for c in range(1, self.num_classes):
            dice, hd95 = self.calculate_metric_percase(pred_vol == c, gt_vol_np == c)
            metric_list.append((dice, hd95))
            self.log(f"test/dice_class_{c}", dice, sync_dist=True, batch_size=1)
            self.log(f"test/hd95_class_{c}", hd95, sync_dist=True, batch_size=1)

        metric_arr = np.array(metric_list)  # [(C-1),2]
        mean_dice = float(metric_arr[:, 0].mean())
        mean_hd95 = float(metric_arr[:, 1].mean())

        self.log("test/mean_dice", mean_dice, prog_bar=True, sync_dist=True, batch_size=1)
        self.log("test/mean_hd95", mean_hd95, prog_bar=True, sync_dist=True, batch_size=1)

        rank_zero_info(f"[TEST] {case} mean_dice={mean_dice:.4f} mean_hd95={mean_hd95:.4f}")
        return {"case": case, "mean_dice": mean_dice, "mean_hd95": mean_hd95}

    #     self._on_eval_end_semantic("val")
