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


def cast_type(precision):
    precision_list = ['fp16', 'bf16', 'fp32']
    if precision == 'fp16':
        return torch.float16
    elif precision == 'bf16':
        return torch.bfloat16
    elif precision == 'fp32':
        return torch.float32
    else:
        raise ValueError(
            f'the precision should in {precision_list}, but got {precision}')
