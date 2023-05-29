from collections import defaultdict
from typing import Union, List, Dict

import torch

from ModaFew.utils import IMAGE_TYPE


class BaseInterface:
    def __init__(self, task):
        self._task = task
        self._default_task_map = {
            'vqa': self.vqa_prompt,
            'caption': self.caption_prompt,
            'classification': self.classification_prompt
        }
        self.prompt_task_map = self._default_task_map

    def construct_prompt(self, *args, **kwargs):
        raise NotImplemented

    def construct_images(self, images: List[IMAGE_TYPE],
                         query_image: IMAGE_TYPE):
        images.append(query_image)
        return images

    @torch.no_grad()
    def get_model_input(self, images_list: List[List[IMAGE_TYPE]], texts_list: List[List[str]]) -> Dict:
        """
        Get the model input.
        For image, you should change it to tensor.
        For text, you should tokenize it to tensor.
        :param images: the batch images List
        :param texts: the batch texts List
        :return: the model input Dict, the key is the model_forward parameters name.
        """
        raise NotImplemented

    @torch.no_grad()
    def model_forward(self, *args, **kwargs) -> List[str]:
        raise NotImplemented

    def postprocess(self, outputs: List[str]) -> List[str]:
        return outputs

    @staticmethod
    def vqa_prompt(*args, **kwargs) -> str:
        raise NotImplemented

    @staticmethod
    def caption_prompt(*args, **kwargs) -> str:
        raise NotImplemented

    @staticmethod
    def classification_prompt(*args, **kwargs) -> str:
        raise NotImplemented

    def add_task(self, task, prompt_method):
        self.prompt_task_map[task] = prompt_method

    @torch.no_grad()
    def few_shot_generation(self,
                            context_images: Union[List[List[IMAGE_TYPE]], List[IMAGE_TYPE]],
                            context_texts: Union[List[List[dict]], List[dict]],
                            input_images: Union[List[IMAGE_TYPE], IMAGE_TYPE],
                            queries: Union[List[dict], dict],
                            **kwargs
                            ):
        if not isinstance(input_images, list):
            input_images = [input_images]
            context_images = [context_images]
            context_texts = [context_texts]
            queries = [queries]

        batch_size = len(context_images)
        prompts_list = []
        image_list = []
        for b in range(batch_size):
            prompts = self.construct_prompt(context_texts[b], queries[b])
            images = self.construct_images(context_images[b], input_images[b])
            prompts_list.append(prompts)
            image_list.append(images)
        
        model_input = self.get_model_input(image_list, prompts_list)

        outputs = self.model_forward(**model_input, **kwargs)
        outputs = self.postprocess(outputs)
        return outputs

    @property
    def get_task_map(self):
        return self.prompt_task_map.keys()
