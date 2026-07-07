from typing import Union
from typing import Optional

from torch.utils.data import DataLoader

from datasets.acdc_dataset import ACDCBaselineTransform, ACDCSliceDataset, ACDCVolumeDataset
from datasets.lightning_data_module import LightningDataModule


class ACDCSemantic(LightningDataModule):
    def __init__(
        self,
        base_dir: Optional[str] = None,
        path: Optional[str] = None,
        root_dir: Optional[str] = None,
        list_dir: Optional[str] = None,
        img_size=(224, 224),
        num_classes: int = 4,
        batch_size: int = 12,
        num_workers: int = 8,
        color_jitter_enabled: bool = True,
        scale_range=(0.5, 2.0),
        check_empty_targets: bool = True,
        ignore_idx: int = 255,
        drop_background: bool = False,
        make_rgb: bool = True,
    ):
        resolved_base_dir = base_dir or path or root_dir
        if resolved_base_dir is None:
            raise ValueError("ACDCSemantic requires one of: base_dir, path, or root_dir.")
        if list_dir is None:
            raise ValueError("ACDCSemantic requires list_dir.")

        super().__init__(
            path=resolved_base_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            ignore_idx=ignore_idx,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.base_dir = resolved_base_dir
        self.list_dir = list_dir
        self.drop_background = drop_background
        self.make_rgb = make_rgb

        self.transforms = ACDCBaselineTransform(output_size=img_size)

    def setup(self, stage: Union[str, None] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ACDCSliceDataset(
                base_dir=self.base_dir,
                list_dir=self.list_dir,
                split="train",
                transforms=self.transforms,
                ignore_index=self.ignore_idx,
                drop_background=self.drop_background,
                make_rgb=self.make_rgb,
            )
            self.eval_dataset = ACDCVolumeDataset(
                base_dir=self.base_dir,
                list_dir=self.list_dir,
                split="test",
                ignore_index=self.ignore_idx,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ACDCVolumeDataset(
                base_dir=self.base_dir,
                list_dir=self.list_dir,
                split="test",
                ignore_index=self.ignore_idx,
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
            self.eval_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=lambda x: x[0],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=lambda x: x[0],
        )
