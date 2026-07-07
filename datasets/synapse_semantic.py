from typing import Union
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.synapse_transforms import Transforms
from datasets.synapse_npz_dataset import SynapseNPZ2DSliceDataset, SynapseH5VolumeCaseDataset


class SynapseSemantic(LightningDataModule):
    def __init__(
        self,
        train_base_dir: str,
        test_base_dir: str,
        list_dir: str,
        img_size=(384, 384),
        num_classes: int = 9,
        batch_size: int = 16,
        num_workers: int = 8,
        color_jitter_enabled: bool = True,
        scale_range=(0.5, 2.0),
        check_empty_targets: bool = True,
        ignore_idx: int = 255,
        drop_background: bool = False,
        make_rgb: bool = True,
    ):
        super().__init__(
            path="",
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            ignore_idx=ignore_idx,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.train_base_dir = train_base_dir
        self.test_base_dir = test_base_dir
        self.list_dir = list_dir
        self.drop_background = drop_background
        self.make_rgb = make_rgb

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage: Union[str, None] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SynapseNPZ2DSliceDataset(
                base_dir=self.train_base_dir,
                list_dir=self.list_dir,
                split="train",
                transforms=self.transforms,
                ignore_index=self.ignore_idx,
                drop_background=self.drop_background,
                make_rgb=self.make_rgb,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SynapseH5VolumeCaseDataset(
                base_dir=self.test_base_dir,
                list_dir=self.list_dir,
                split="test_vol",
                transforms=None,
                ignore_index=self.ignore_idx,
                make_rgb=self.make_rgb,
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
            self.train_dataset,
            shuffle=False,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
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
