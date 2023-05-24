from typing import Union

import torch
from PIL import Image

IMAGE_TYPE = Union[str, Image.Image, torch.Tensor]


def image2tensor(image: IMAGE_TYPE, vis_processor):
    if isinstance(image, str):  # is a image path
        raw_image = Image.open(image).convert('RGB')
        image = vis_processor(raw_image)
    elif isinstance(image, Image.Image):
        raw_image = image.convert('RGB')
        image = vis_processor(raw_image)
    elif isinstance(image, torch.Tensor):
        image = image
    else:
        raise ValueError()
    return image
