from typing import Union

import torch
from PIL import Image

IMAGE_TYPE = Union[str, Image.Image, torch.Tensor]


def image2tensor(image: IMAGE_TYPE, vis_processor):
    if isinstance(image, str):  # is a image path
        raw_image = Image.open(image).convert('RGB')
        image = vis_processor(raw_image).unsqueeze(0)
    elif isinstance(image, Image.Image):
        raw_image = image
        image = vis_processor(raw_image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
    return image
