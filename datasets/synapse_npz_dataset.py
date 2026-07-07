from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors


def label_to_eomt_target(label_2d: torch.Tensor, ignore_index: int = 255, drop_background: bool = False):
    masks: List[torch.Tensor] = []
    labels: List[int] = []
    is_crowd: List[bool] = []

    for cls_id_t in torch.unique(label_2d):
        cls_id = int(cls_id_t.item())
        if cls_id == ignore_index:
            continue
        if drop_background and cls_id == 0:
            continue
        masks.append(label_2d == cls_id_t)
        labels.append(cls_id)
        is_crowd.append(False)

    if not masks:
        h, w = label_2d.shape
        empty_masks = torch.zeros((0, h, w), dtype=torch.bool)
        empty_labels = torch.zeros((0,), dtype=torch.long)
        empty_crowd = torch.zeros((0,), dtype=torch.bool)
        return {
            "masks": tv_tensors.Mask(empty_masks),
            "labels": empty_labels,
            "is_crowd": empty_crowd,
        }

    return {
        "masks": tv_tensors.Mask(torch.stack(masks), dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long),
        "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
    }


class SynapseNPZ2DSliceDataset(Dataset):
    """
    Train split: read 2D slices from .npz and output (img, target) for EoMT.
    """
    def __init__(
        self,
        base_dir: str,
        list_dir: str,
        split: str,
        transforms=None,
        ignore_index: int = 255,
        drop_background: bool = False,
        make_rgb: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.list_dir = Path(list_dir)
        self.split = split
        self.transforms = transforms
        self.ignore_index = ignore_index
        self.drop_background = drop_background
        self.make_rgb = make_rgb

        txt = self.list_dir / f"{split}.txt"
        self.sample_list = [x.strip() for x in txt.read_text().splitlines() if x.strip()]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx: int):
        name = self.sample_list[idx]
        data_path = self.base_dir / f"{name}.npz"
        data = np.load(str(data_path))
        image = data["image"]
        label = data["label"]

        img_t = torch.from_numpy(image.astype(np.float32))
        lbl_t = torch.from_numpy(label.astype(np.int64))

        if self.make_rgb:
            img_t = img_t.unsqueeze(0).repeat(3, 1, 1)
        else:
            img_t = img_t.unsqueeze(0)

        img = tv_tensors.Image(img_t)

        target = label_to_eomt_target(lbl_t, ignore_index=self.ignore_index, drop_background=self.drop_background)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target["case_name"] = name
        return img, target


class SynapseH5VolumeCaseDataset(Dataset):
    """
    Test split: return a whole 3D volume per item (case-level),
    so we can compute standard Synapse metrics like TransUNet (per-case 3D Dice/HD95).
    """
    def __init__(
        self,
        base_dir: str,
        list_dir: str,
        split: str,
        transforms=None,
        ignore_index: int = 255,
        make_rgb: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.list_dir = Path(list_dir)
        self.split = split
        self.transforms = transforms
        self.ignore_index = ignore_index
        self.make_rgb = make_rgb

        txt = self.list_dir / f"{split}.txt"
        self.cases = [x.strip() for x in txt.read_text().splitlines() if x.strip()]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int):
        case = self.cases[idx]
        fp = self.base_dir / f"{case}.npy.h5"
        with h5py.File(fp, "r") as f:
            image = f["image"][:]
            label = f["label"][:]

        img = torch.from_numpy(image.astype(np.float32))
        lbl = torch.from_numpy(label.astype(np.int64))

        return {
            "image": img,
            "label": lbl,
            "case_name": case,
        }
