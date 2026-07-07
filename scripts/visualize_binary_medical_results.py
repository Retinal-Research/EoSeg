import argparse
import csv
import importlib
import inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


DEFAULT_FOREGROUND_COLOR = "#d65252"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize binary medical segmentation predictions for GlaS or MoNuSeg."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Folder containing img/ and labelcol/ subfolders.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--foreground-color", type=str, default=DEFAULT_FOREGROUND_COLOR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-mask-npy", action="store_true")
    parser.add_argument("--skip-panel", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument(
        "--save-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Optional output PNG size. Example: --save-size 224 224",
    )
    return parser.parse_args()


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_from_config(config_node):
    if isinstance(config_node, dict) and "class_path" in config_node:
        cls = import_class(config_node["class_path"])
        init_args = config_node.get("init_args", {})
        kwargs = {k: instantiate_from_config(v) for k, v in init_args.items()}
        sig = inspect.signature(cls.__init__)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not accepts_var_kwargs:
            valid_keys = {
                name
                for name, p in sig.parameters.items()
                if name != "self"
                and p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return cls(**kwargs)
    if isinstance(config_node, dict):
        return {k: instantiate_from_config(v) for k, v in config_node.items()}
    if isinstance(config_node, list):
        return [instantiate_from_config(v) for v in config_node]
    return config_node


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_style_links(config: dict, ckpt_path: Path):
    config = yaml.safe_load(yaml.safe_dump(config))

    data_init = config.get("data", {}).get("init_args", {})
    model_init = config.get("model", {}).setdefault("init_args", {})
    network_init = model_init.setdefault("network", {}).setdefault("init_args", {})
    encoder_init = network_init.setdefault("encoder", {}).setdefault("init_args", {})

    if "num_classes" in data_init:
        model_init.setdefault("num_classes", data_init["num_classes"])
        network_init.setdefault("num_classes", data_init["num_classes"])

    if "img_size" in data_init:
        model_init.setdefault("img_size", data_init["img_size"])
        encoder_init.setdefault("img_size", data_init["img_size"])

    encoder_init.setdefault("ckpt_path", str(ckpt_path))
    model_init["ckpt_path"] = None
    return config


def load_model(config: dict, ckpt_path: Path, device: torch.device):
    resolved_config = apply_cli_style_links(config, ckpt_path)
    model_cfg = resolved_config["model"]
    model = instantiate_from_config(model_cfg)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model


def normalize_image(image_np: np.ndarray):
    image_np = image_np.astype(np.float32)
    if image_np.ndim == 2:
        lo = np.percentile(image_np, 1)
        hi = np.percentile(image_np, 99)
        if hi <= lo:
            lo = float(image_np.min())
            hi = float(image_np.max())
        image_np = np.clip(image_np, lo, hi)
        if hi > lo:
            image_np = (image_np - lo) / (hi - lo)
        else:
            image_np = np.zeros_like(image_np)
        return image_np

    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    return np.clip(image_np, 0.0, 1.0)


def image_to_rgb(image_np: np.ndarray):
    image_np = normalize_image(image_np)
    if image_np.ndim == 2:
        return np.stack([image_np, image_np, image_np], axis=-1)
    if image_np.shape[0] in (1, 3):
        return np.transpose(image_np, (1, 2, 0))
    return image_np


def overlay_binary_mask(image_np: np.ndarray, mask_np: np.ndarray, color: str, alpha: float = 0.45):
    base = image_to_rgb(image_np).copy()
    tint = np.array(mcolors.to_rgb(color), dtype=np.float32)
    fg = mask_np.astype(bool)
    base[fg] = (1.0 - alpha) * base[fg] + alpha * tint
    return np.clip(base, 0.0, 1.0)


def binary_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    intersection = np.sum(pred * gt)
    return float((2.0 * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps))


def binary_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((intersection + eps) / (union + eps))


def align_prediction_polarity(pred: np.ndarray, gt: np.ndarray):
    direct_dice = binary_dice(pred, gt)
    inverted = 1 - pred.astype(np.uint8)
    inverted_dice = binary_dice(inverted, gt)
    if inverted_dice > direct_dice:
        return inverted, True
    return pred, False


def collect_pairs(test_dir: Path):
    image_dir = test_dir / "img"
    mask_dir = test_dir / "labelcol"
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def preprocess_image(image_path: Path, img_size: tuple[int, int]):
    image = Image.open(image_path).convert("RGB")
    orig_np = np.array(image, dtype=np.uint8)
    resized = image.resize((img_size[1], img_size[0]), resample=Image.BILINEAR)
    tensor = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return tensor, orig_np


def load_mask(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    mask_np = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
    return mask_np


def infer_binary_mask(model, image_tensor: torch.Tensor, output_size: tuple[int, int], threshold: float, device: torch.device):
    with torch.no_grad():
        _, probs = model._foreground_logits_and_probs(image_tensor.unsqueeze(0).to(device))
        pred_small = (probs[0, 0] >= threshold).to(torch.uint8).cpu().numpy()

    pred_img = Image.fromarray(pred_small * 255)
    pred_img = pred_img.resize((output_size[1], output_size[0]), resample=Image.NEAREST)
    pred_np = (np.array(pred_img, dtype=np.uint8) > 0).astype(np.uint8)
    return pred_np


def binary_mask_to_image(mask_np: np.ndarray):
    return (mask_np.astype(np.uint8) * 255)


def resize_for_saving(arr: np.ndarray, save_size, is_mask: bool):
    if save_size is None:
        return arr
    height, width = save_size
    pil = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = pil.resize((width, height), resample=resample)
    return np.array(resized)


def save_outputs(output_dir: Path, stem: str, orig_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray, foreground_color: str, skip_panel: bool, save_overlay: bool, save_size):
    image_rgb = image_to_rgb(orig_np)
    gt_mask = binary_mask_to_image(gt_np)
    pred_mask = binary_mask_to_image(pred_np)

    image_to_save = resize_for_saving((image_rgb * 255).astype(np.uint8), save_size, is_mask=False)
    gt_to_save = resize_for_saving(gt_mask, save_size, is_mask=True)
    pred_to_save = resize_for_saving(pred_mask, save_size, is_mask=True)

    plt.imsave(output_dir / f"{stem}_image.png", image_to_save)
    plt.imsave(output_dir / f"{stem}_gt.png", gt_to_save, cmap="gray", vmin=0, vmax=255)
    plt.imsave(output_dir / f"{stem}_pred.png", pred_to_save, cmap="gray", vmin=0, vmax=255)

    if save_overlay:
        gt_overlay = overlay_binary_mask(orig_np, gt_np, foreground_color)
        pred_overlay = overlay_binary_mask(orig_np, pred_np, foreground_color)
        gt_overlay_to_save = resize_for_saving((gt_overlay * 255).astype(np.uint8), save_size, is_mask=False)
        pred_overlay_to_save = resize_for_saving((pred_overlay * 255).astype(np.uint8), save_size, is_mask=False)
        plt.imsave(output_dir / f"{stem}_gt_overlay.png", gt_overlay_to_save)
        plt.imsave(output_dir / f"{stem}_pred_overlay.png", pred_overlay_to_save)

    if skip_panel:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(image_to_save)
    axes[0].set_title("image")
    axes[1].imshow(gt_to_save, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("ground truth")
    axes[2].imshow(pred_to_save, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("prediction")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_panel.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config = load_config(args.config)
    model = load_model(config, args.ckpt, device)
    img_size = tuple(config["data"]["init_args"]["img_size"])

    pairs = collect_pairs(args.test_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    rows = []
    for image_path, mask_path in tqdm(pairs, desc="Visualizing", total=len(pairs)):
        image_tensor, orig_np = preprocess_image(image_path, img_size)
        gt_np = load_mask(mask_path)
        pred_np = infer_binary_mask(model, image_tensor, gt_np.shape, args.threshold, device)
        pred_np, flipped = align_prediction_polarity(pred_np, gt_np)

        stem = image_path.stem
        save_outputs(
            args.output_dir,
            stem,
            orig_np,
            gt_np,
            pred_np,
            args.foreground_color,
            args.skip_panel,
            args.save_overlay,
            args.save_size,
        )

        if args.save_mask_npy:
            np.save(args.output_dir / f"{stem}_pred.npy", pred_np)

        rows.append(
            {
                "image_name": image_path.name,
                "dice": binary_dice(pred_np, gt_np),
                "iou": binary_iou(pred_np, gt_np),
                "polarity_flipped": int(flipped),
            }
        )

    csv_path = args.output_dir / "metrics_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] saved visualizations to {args.output_dir}")
        print(f"[done] saved metrics to {csv_path}")
        print(f"[done] mean dice: {np.mean([r['dice'] for r in rows]):.4f}")
        print(f"[done] mean iou: {np.mean([r['iou'] for r in rows]):.4f}")
    else:
        print("[done] no matched image/mask pairs found")


if __name__ == "__main__":
    main()
