import json
from pathlib import Path
from typing import Union, Callable, Optional
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import tv_tensors
from pycocotools import mask as coco_mask
import torch
import numpy as np
from PIL import Image

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms


# We map all custom categories to class 0 since classification is irrelevant.
# The base model requires background as the last class (num_classes).
CLASS_MAPPING = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}


class CoolblueDataset(TorchDataset):
    def __init__(
        self,
        img_dir: Path,
        annotation_file: Path,
        transforms: Optional[Callable] = None,
        target_parser: Optional[Callable] = None,
    ):
        self.img_dir = img_dir
        self.transforms = transforms
        self.target_parser = target_parser

        with open(annotation_file, "r") as file:
            annotation_data = json.load(file)

        self.images = annotation_data["images"]
        self.image_id_to_file_name = {
            image["id"]: Path(image["file_name"]).name 
            for image in self.images
        }
        
        # Build dictionaries for fast lookup during __getitem__
        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}

        for annotation in annotation_data.get("annotations", []):
            img_filename = self.image_id_to_file_name[annotation["image_id"]]

            if img_filename not in self.labels_by_id:
                self.labels_by_id[img_filename] = {}
                self.polygons_by_id[img_filename] = {}
                self.is_crowd_by_id[img_filename] = {}

            ann_id = annotation["id"]
            self.labels_by_id[img_filename][ann_id] = annotation["category_id"]
            self.polygons_by_id[img_filename][ann_id] = annotation["segmentation"]
            self.is_crowd_by_id[img_filename][ann_id] = bool(annotation["iscrowd"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        img_info = self.images[index]
        img_filename = Path(img_info["file_name"]).name
        img_path = self.img_dir / img_filename

        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))

        masks, labels, is_crowd = self.target_parser(
            polygons_by_id=self.polygons_by_id.get(img_filename, {}),
            labels_by_id=self.labels_by_id.get(img_filename, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(img_filename, {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )

        if not masks:
            # Handle empty targets gracefully if needed, or return empty tensors
            target = {
                "masks": tv_tensors.Mask(torch.empty((0, img.shape[-2], img.shape[-1]), dtype=torch.bool)),
                "labels": torch.empty((0,), dtype=torch.long),
                "is_crowd": torch.empty((0,), dtype=torch.bool),
            }
        else:
            target = {
                "masks": tv_tensors.Mask(torch.stack(masks)),
                "labels": torch.tensor(labels, dtype=torch.long),
                "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
            }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CoolblueInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 4,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 80,
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(
        polygons_by_id: dict[int, list[list[float]]],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        width: int,
        height: int,
        **kwargs
    ):
        masks, labels, is_crowd = [], [], []

        for label_id, cls_id in labels_by_id.items():
            if cls_id not in CLASS_MAPPING:
                continue

            segmentation = polygons_by_id[label_id]
            
            # Skip empty segmentations
            if not segmentation:
                continue
                
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles) if isinstance(rles, list) else rles

            masks.append(tv_tensors.Mask(coco_mask.decode(rle), dtype=torch.bool))
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id[label_id])

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        base_path = Path(self.path)
        img_dir = base_path / "rgb"
        splits_dir = base_path / "manual_split"
        
        train_json = splits_dir / "train_coolblue_2024_12.json"
        val_json = splits_dir / "val_coolblue_2024_12.json"

        self.train_dataset = CoolblueDataset(
            img_dir=img_dir,
            annotation_file=train_json,
            transforms=self.transforms,
            target_parser=self.target_parser,
        )
        
        self.val_dataset = CoolblueDataset(
            img_dir=img_dir,
            annotation_file=val_json,
            transforms=Transforms(img_size=self.img_size, color_jitter_enabled=False, scale_range=(1.0, 1.0)), # Eval transforms
            target_parser=self.target_parser,
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
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
