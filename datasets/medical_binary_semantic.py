from pathlib import Path
import random
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from datasets.lightning_data_module import LightningDataModule


class MedicalBinaryTransforms:
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


class MedicalBinaryFolderDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        img_size: tuple[int, int],
        train: bool,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "img"
        self.mask_dir = self.dataset_dir / "labelcol"
        self.transforms = MedicalBinaryTransforms(img_size=img_size, train=train)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.image_paths = []
        for path in sorted(self.image_dir.iterdir()):
            if not path.is_file():
                continue

            mask_path = self.mask_dir / f"{path.stem}.png"
            if mask_path.is_file():
                self.image_paths.append(path)

        if not self.image_paths:
            raise FileNotFoundError(
                f"No matched image/mask pairs found in {self.dataset_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

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


class MedicalBinarySplitDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str,
        img_size: tuple[int, int],
        train: bool,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.mask_dir = self.dataset_dir / "masks"
        self.split_path = self.dataset_dir / f"{split}.txt"
        self.transforms = MedicalBinaryTransforms(img_size=img_size, train=train)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        if not self.split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")

        self.samples: list[tuple[Path, Path]] = []
        for raw_name in self.split_path.read_text(encoding="utf-8").splitlines():
            name = raw_name.strip()
            if not name:
                continue

            image_path = self._resolve_file(self.image_dir, name)
            mask_path = self._resolve_file(self.mask_dir, name)
            self.samples.append((image_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No matched image/mask pairs found using split {self.split_path}"
            )

    @staticmethod
    def _resolve_file(folder: Path, stem_or_name: str) -> Path:
        base = folder / stem_or_name
        if base.is_file():
            return base

        for ext in (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"):
            candidate = folder / f"{stem_or_name}{ext}"
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(f"Could not resolve file for {stem_or_name} under {folder}")

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


class MedicalBinarySemantic(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: Optional[str] = None,
        img_size: tuple[int, int] = (224, 224),
        num_classes: int = 2,
        batch_size: int = 14,
        num_workers: int = 8,
        check_empty_targets: bool = False,
        ignore_idx: int = 255,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        dataset_dir: Optional[str] = None,
        train_split: str = "train",
        val_split: str = "val",
        test_split: Optional[str] = None,
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

    def setup(self, stage: Union[str, None] = None):
        use_split_dataset = self.hparams.dataset_dir is not None

        if stage == "fit" or stage is None:
            if use_split_dataset:
                self.train_dataset = MedicalBinarySplitDataset(
                    dataset_dir=self.hparams.dataset_dir,
                    split=self.hparams.train_split,
                    img_size=self.img_size,
                    train=True,
                )
                self.val_dataset = MedicalBinarySplitDataset(
                    dataset_dir=self.hparams.dataset_dir,
                    split=self.hparams.val_split,
                    img_size=self.img_size,
                    train=False,
                )
            else:
                self.train_dataset = MedicalBinaryFolderDataset(
                    dataset_dir=self.hparams.train_dir,
                    img_size=self.img_size,
                    train=True,
                )
                self.val_dataset = MedicalBinaryFolderDataset(
                    dataset_dir=self.hparams.val_dir,
                    img_size=self.img_size,
                    train=False,
                )

        if stage == "validate":
            if use_split_dataset:
                self.val_dataset = MedicalBinarySplitDataset(
                    dataset_dir=self.hparams.dataset_dir,
                    split=self.hparams.val_split,
                    img_size=self.img_size,
                    train=False,
                )
            else:
                self.val_dataset = MedicalBinaryFolderDataset(
                    dataset_dir=self.hparams.val_dir,
                    img_size=self.img_size,
                    train=False,
                )

        if stage == "test" or stage is None:
            if use_split_dataset:
                self.test_dataset = MedicalBinarySplitDataset(
                    dataset_dir=self.hparams.dataset_dir,
                    split=self.hparams.test_split or self.hparams.val_split,
                    img_size=self.img_size,
                    train=False,
                )
            else:
                test_dir = self.hparams.test_dir or self.hparams.val_dir
                self.test_dataset = MedicalBinaryFolderDataset(
                    dataset_dir=test_dir,
                    img_size=self.img_size,
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
