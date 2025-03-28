# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------


import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torch import nn


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta=32,
        max_contrast_factor=0.5,
        saturation_factor=0.5,
        max_hue_delta=18,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()

        self.scale_jitter = T.ScaleJitter(
            target_size=img_size,
            scale_range=scale_range,
        )

        self.random_crop = T.RandomCrop(img_size)

    def random_factor(self, factor, center=1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, brightness_factor=self.random_factor(self.max_brightness_factor)
            )

        return img

    def contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(
                img,
                contrast_factor=self.random_factor(self.max_contrast_factor),
            )

        return img

    def saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img,
                saturation_factor=self.random_factor(self.max_saturation_factor),
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(
                img,
                hue_factor=self.random_factor(self.max_hue_delta, center=0.0),
            )

        return img

    def color_jitter(self, img):
        img = self.brightness(img)

        if torch.rand(()) < 0.5:
            img = self.contrast(img)
            img = self.saturation_and_hue(img)
        else:
            img = self.saturation_and_hue(img)
            img = self.contrast(img)

        return img

    def pad(self, img, target: dict):
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def forward(self, img, target: dict):
        img_ = img.clone()
        target_ = {
            "masks": target["masks"][~target["is_crowd"]].clone(),
            "labels": target["labels"][~target["is_crowd"]].clone(),
        }

        if self.color_jitter_enabled:
            img_ = self.color_jitter(img_)

        img_, target_ = self.random_horizontal_flip(img_, target)

        img_, target_ = self.scale_jitter(img_, target_)

        img_, target_ = self.pad(img_, target_)

        img_, target_ = self.random_crop(img_, target_)

        mask_sums = target_["masks"].sum(dim=[-2, -1])
        non_empty_mask = mask_sums > 0

        if non_empty_mask.sum() == 0:
            return self(img, target)

        target_["masks"] = target_["masks"][non_empty_mask]
        target_["labels"] = target_["labels"][non_empty_mask]

        return img_, target_
