from collections import defaultdict
from typing import Union, List, Dict

import torch

from ModaFew.utils import IMAGE_TYPE


class BaseInterface:
    def __int__(self, task):
        self._task = task
        self._default_task_map = {
            'vqa': self.vqa_prompt,
            'caption': self.caption_prompt,
            'classification': self.classfication_prompt
        }
        self.prompt_task_map = self._default_task_map

    def construct_prompt(self, *args, **kwargs):
        raise NotImplemented

    def construct_images(self, images: List[IMAGE_TYPE],
                         query_image: IMAGE_TYPE):
        images.append(query_image)
        return images

    @torch.no_grad()
    def get_model_input(self, images: List[IMAGE_TYPE], texts: List[str]) -> Dict:
        """
        Get the model input of one input.
        For image, you should change it to tensor.
        For text, you should tokenize it to tensor.
        :param images: the images List
        :param texts: the texts List
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
                            context_images: Union[List[List[IMAGE_TYPE]],
                            List[IMAGE_TYPE]],
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

        batch_model_inputs_collect = defaultdict(list)
        batch_size = len(context_images)
        for b in range(batch_size):
            prompts = self.construct_prompt(context_texts[b], queries[b])
            image_list = self.construct_images(context_images[b], input_images[b])
            model_input = self.get_model_input(image_list, prompts)
            for input_name, input_values in model_input.items():
                batch_model_inputs_collect[input_name].append(input_values)

        batch_model_inputs = {}
        for n, v in batch_model_inputs_collect.items():
            batch_model_inputs[n] = torch.stack(v)

        outputs = self.model_forward(**batch_model_inputs, **kwargs)
        outputs = self.postprocess(outputs)
        return outputs

    @property
    def get_task_map(self):
        return self.prompt_task_map.keys()
