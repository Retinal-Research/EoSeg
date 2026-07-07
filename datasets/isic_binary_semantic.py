from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from datasets.lightning_data_module import LightningDataModule


class ISICBinaryTransforms:
    def __init__(self, img_size: tuple[int, int], train: bool) -> None:
        self.img_size = img_size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            if random.random() > 0.5:
                k = random.randint(0, 3)
                angle = 90 * k
                image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

                if random.random() > 0.5:
                    image = F.hflip(image)
                    mask = F.hflip(mask)
                else:
                    image = F.vflip(image)
                    mask = F.vflip(mask)
            elif random.random() > 0.5:
                angle = random.uniform(-20.0, 20.0)
                image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        image = F.resize(image, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)

        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        mask = (mask > 0).to(torch.uint8)

        return image, mask


class ISICBinaryPairedDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: tuple[int, int],
        train: bool,
        image_suffix: str = ".jpg",
        mask_suffix: str = "_segmentation.png",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = ISICBinaryTransforms(img_size=img_size, train=train)
        self.image_suffix = image_suffix.lower()
        self.mask_suffix = mask_suffix

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.name.startswith("."):
                continue
            if image_path.name.startswith("._"):
                continue
            if image_path.suffix.lower() != self.image_suffix:
                continue

            mask_path = self.mask_dir / f"{image_path.stem}{self.mask_suffix}"
            if mask_path.is_file():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No matched image/mask pairs found between {self.image_dir} and {self.mask_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = self.transforms(image, mask)

        target = {
            "masks": mask.unsqueeze(0).bool(),
            "labels": torch.tensor([0], dtype=torch.long),
            "is_crowd": torch.tensor([False]),
            "image_name": image_path.name,
        }
        return image, target


class ISICBinarySemantic(LightningDataModule):
    def __init__(
        self,
        train_image_dir: str,
        train_mask_dir: str,
        val_image_dir: str,
        val_mask_dir: str,
        test_image_dir: Optional[str] = None,
        test_mask_dir: Optional[str] = None,
        img_size: tuple[int, int] = (256, 256),
        num_classes: int = 2,
        batch_size: int = 12,
        num_workers: int = 8,
        check_empty_targets: bool = False,
        ignore_idx: int = 255,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_suffix: str = ".jpg",
        mask_suffix: str = "_segmentation.png",
    ) -> None:
        super().__init__(
            path="",
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            ignore_idx=ignore_idx,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.save_hyperparameters(ignore=["_class_path"])

    def _make_dataset(self, image_dir: str, mask_dir: str, train: bool):
        return ISICBinaryPairedDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            img_size=self.img_size,
            train=train,
            image_suffix=self.hparams.image_suffix,
            mask_suffix=self.hparams.mask_suffix,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset(
                self.hparams.train_image_dir,
                self.hparams.train_mask_dir,
                train=True,
            )
            self.val_dataset = self._make_dataset(
                self.hparams.val_image_dir,
                self.hparams.val_mask_dir,
                train=False,
            )

        if stage == "validate":
            self.val_dataset = self._make_dataset(
                self.hparams.val_image_dir,
                self.hparams.val_mask_dir,
                train=False,
            )

        if stage == "test" or stage is None:
            self.test_dataset = self._make_dataset(
                self.hparams.test_image_dir or self.hparams.val_image_dir,
                self.hparams.test_mask_dir or self.hparams.val_mask_dir,
                train=False,
            )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=self.train_collate,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.dataloader_kwargs["pin_memory"],
            persistent_workers=self.dataloader_kwargs["persistent_workers"],
        )
