from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from PIL import Image
import torch
import numpy as np

from datasets.lightning_data_module import LightningDataModule

class SRInstance(LightningDataModule):
    """
    Simplified dataset for inference. 
    Returns raw uint8 tensors to support the model's internal PIL-based resizing.
    """
    def __init__(
        self,
        path,
        num_workers: int = 0,
        batch_size: int = 1,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 80,
        img_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        **kwargs
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=False,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.img_extensions = img_extensions

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        image_folder = Path(self.path)
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        self.image_paths = []
        for ext in self.img_extensions:
            self.image_paths.extend(list(image_folder.glob(f"*{ext}")))
            self.image_paths.extend(list(image_folder.glob(f"*{ext.upper()}")))
        
        self.image_paths = sorted(self.image_paths)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}")
        
        print(f"[INFO] Found {len(self.image_paths)} images in {image_folder}")
        
        self.val_dataset = SimpleImageDataset(image_paths=self.image_paths)
        self.train_dataset = self.val_dataset

        return self

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

class SimpleImageDataset:
    def __init__(self, image_paths: list):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load image and ensure it's RGB
        image = Image.open(img_path).convert('RGB')
        
        # Convert to uint8 tensor [C, H, W] to satisfy model.resize_and_pad_imgs_instance_panoptic
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        
        info = {"file_name": img_path.name}
        
        return img_tensor, info