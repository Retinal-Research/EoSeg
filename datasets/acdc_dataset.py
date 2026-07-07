from pathlib import Path
from typing import List

import numpy as np
import random
from scipy import ndimage
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class ACDCBaselineTransform:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image: np.ndarray, label: np.ndarray):
        image = image.astype(np.float32)
        img_min = float(image.min())
        img_max = float(image.max())
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=3,
            )
            label = zoom(
                label,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=0,
            )
        return image, label


def label_to_eomt_target(
    label_2d: torch.Tensor,
    ignore_index: int = 255,
    drop_background: bool = False,
):
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


class ACDCSliceDataset(Dataset):
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
        data_path = self.base_dir / self.split / name
        data = np.load(str(data_path))
        image = data["img"].astype(np.float32)
        label = data["label"]

        if self.transforms is not None:
            image, label = self.transforms(image, label)
        else:
            img_min = float(image.min())
            img_max = float(image.max())
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            else:
                image = np.zeros_like(image, dtype=np.float32)

        img_t = torch.from_numpy(image.astype(np.float32))
        lbl_t = torch.from_numpy(label.astype(np.int64))

        if self.make_rgb:
            img_t = img_t.unsqueeze(0).repeat(3, 1, 1)
        else:
            img_t = img_t.unsqueeze(0)

        img = tv_tensors.Image(img_t)
        target = label_to_eomt_target(
            lbl_t,
            ignore_index=self.ignore_index,
            drop_background=self.drop_background,
        )

        target["case_name"] = name
        return img, target


class ACDCVolumeDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        list_dir: str,
        split: str = "test",
        ignore_index: int = 255,
    ):
        self.base_dir = Path(base_dir)
        self.list_dir = Path(list_dir)
        self.split = split
        self.ignore_index = ignore_index

        txt = self.list_dir / f"{split}.txt"
        self.cases = [x.strip() for x in txt.read_text().splitlines() if x.strip()]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int):
        case = self.cases[idx]
        fp = self.base_dir / self.split / case
        data = np.load(str(fp))

        image_np = data["img"].astype(np.float32)
        img_min = float(image_np.min())
        img_max = float(image_np.max())
        if img_max > img_min:
            image_np = (image_np - img_min) / (img_max - img_min)
        else:
            image_np = np.zeros_like(image_np, dtype=np.float32)

        image = torch.from_numpy(image_np)
        label = torch.from_numpy(data["label"].astype(np.int64))

        return {
            "image": image,
            "label": label,
            "case_name": case,
        }
