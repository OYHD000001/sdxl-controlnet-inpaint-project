from __future__ import annotations

import random
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class SDXLTrainTransforms:
    """
    Shared geometric transforms for all image-like inputs in a sample.

    TODO:
    - add richer augmentation once baseline training is stable
    - keep image/mask/conditioning alignment exact at all times
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        center_crop: bool = False,
        random_flip: bool = False,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.center_crop = center_crop
        self.random_flip = random_flip

    def _resize(self, image: Image.Image, is_mask: bool) -> Image.Image:
        interpolation = Image.NEAREST if is_mask else Image.BILINEAR
        return image.resize((self.image_width, self.image_height), interpolation)

    def _crop(self, image: Image.Image, is_mask: bool) -> Image.Image:
        if self.center_crop:
            return TF.center_crop(image, [self.image_height, self.image_width])
        return image

    def _to_image_tensor(self, image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image)
        return tensor * 2.0 - 1.0

    def _to_conditioning_tensor(self, image: Image.Image) -> torch.Tensor:
        # Match the official diffusers ControlNet training path:
        # conditioning images stay in [0, 1], while VAE inputs use [-1, 1].
        return TF.to_tensor(image)

    def _to_mask_tensor(self, mask: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(mask)
        if tensor.shape[0] > 1:
            tensor = tensor[:1]
        return (tensor > 0.5).float()

    def __call__(
        self,
        target_image: Image.Image,
        source_image: Image.Image,
        mask_image: Image.Image,
        masked_source_image: Image.Image,
        conditioning_image: Image.Image,
    ) -> dict[str, Any]:
        do_flip = self.random_flip and random.random() < 0.5

        images = {
            "target_image": self._crop(self._resize(target_image, is_mask=False), is_mask=False),
            "source_image": self._crop(self._resize(source_image, is_mask=False), is_mask=False),
            "mask_image": self._crop(self._resize(mask_image, is_mask=True), is_mask=True),
            "masked_source_image": self._crop(self._resize(masked_source_image, is_mask=False), is_mask=False),
            "conditioning_image": self._crop(self._resize(conditioning_image, is_mask=False), is_mask=False),
        }

        if do_flip:
            images = {key: TF.hflip(value) for key, value in images.items()}

        return {
            "target_image": self._to_image_tensor(images["target_image"]),
            "source_image": self._to_image_tensor(images["source_image"]),
            "mask_image": self._to_mask_tensor(images["mask_image"]),
            "masked_source_image": self._to_image_tensor(images["masked_source_image"]),
            "conditioning_image": self._to_conditioning_tensor(images["conditioning_image"]),
        }


def build_train_transforms(
    image_height: int,
    image_width: int,
    center_crop: bool = False,
    random_flip: bool = False,
) -> SDXLTrainTransforms:
    return SDXLTrainTransforms(
        image_height=image_height,
        image_width=image_width,
        center_crop=center_crop,
        random_flip=random_flip,
    )
