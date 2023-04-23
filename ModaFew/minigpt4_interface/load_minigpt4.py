import torch
import numpy as np

from typing import Union, List
from PIL.Image import Image

from .minigpt4.common.config import Config
from .minigpt4.common.registry import registry
from .minigpt4.conversation.conversation import Chat, CONV_VISION


class MiniGPT4Interface:
    def __init__(self, config_path, gpu_id, **kwargs):
        print('Initializing Chat')
        gpu_id = int(gpu_id)
        self.cfg = Config(config_path, **kwargs)
        model_config = self.cfg.model_cfg
        model_config.device_8bit = gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(gpu_id))


        vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device='cuda:{}'.format(gpu_id))
        self.conversation_history = CONV_VISION
        self.image_list = []
        print('Initialization Finished')

    
    def reset(self):
        self.conversation_history = CONV_VISION.copy()

    def _chat_one_time(self, image, query, **kwargs):
        if image:
            self.chat.upload_img(image, self.conversation_history, self.image_list)
        self.chat.ask(query, self.conversation_history)
        answer = self.chat.answer(self.conversation_history, self.image_list, **kwargs)[0]
        return answer
        
    def zero_shot_generation(self, 
                             image: Union[Image, str, torch.Tensor], 
                             query: str ='', 
                             **kwargs):
        assert image, f"In zero_shot_generation function, the image should be Union[Image, str, torch.Tensor], but got {type(image)}"
        answer = self._chat_one_time(image, query, **kwargs)
        self.reset()
        return answer

    def few_shot_generation(self, 
                            example_images: List[Union[Image, str, torch.Tensor]], 
                            example_texts: List[str], 
                            input_images: Union[List[Union[Image, str, torch.Tensor]], Image, str, torch.Tensor], 
                            query: str ='', 
                            **kwargs):
        assert len(example_images) == len(example_texts), f"The few-shot image should num should be the same as the num of example_texts"
        few_shot_num = len(example_texts)
        for i in range(few_shot_num):
            self.chat.upload_img(example_images[i], self.conversation_history, self.image_list)
            self.chat.ask(query, self.conversation_history)
            self.conversation_history.append_message(self.conversation_history.roles[1], example_texts[i])
        
        if not isinstance(input_images, List):
            input_images = [input_images]

        assert len(input_images) == 1, f"Now only support one image as output"

        for input_image in input_images:
            self.chat.upload_img(input_image, self.conversation_history, self.image_list)
            self.chat.ask(query, self.conversation_history)
            output_text = self.chat.answer(self.conversation_history, self.image_list, **kwargs)[0]
        self.reset()
        return output_text
    
"""
Conversation.messages:
1) upload image
[["Human", "<Img><ImageHere></Img>"]]

2) ask
[["Human", "<Img><ImageHere></Img> Describe the image" ]]

3) answer
[["Human", "<Img><ImageHere></Img> Describe the image" ], ["Assistant", None]]

Prompt = Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> Describe the image###Assistant:

get_context_emb
Prompt = Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> Describe the image###Assistant:
prompt_segs = ["Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img>", "</Img> Describe the image###Assistant:"]

after get the answer 
[["Human", "<Img><ImageHere></Img> Describe the image" ], ["Assistant", "This is a cat!"]]

4) ask again
[["Human", "<Img><ImageHere></Img> Describe the image" ], ["Assistant", "This is a cat!"], ["Human", "Describe the image again"]]

"""

if __name__ == '__main__':
    import time
    import torch
    from PIL import Image
    import requests

    device = '0'
    time_begin = time.time()
    interface = MiniGPT4Interface(config_path='./minigpt4_interface/eval_config.yaml', device=device)


    """
    Step 1: Load images
    """
    demo_image_one = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw
    )

    demo_image_two = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            stream=True
        ).raw
    )

    query_image = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
            stream=True
        ).raw
    )
    example_images = [demo_image_one, demo_image_two]

    texts_input = [
        ["An image of two cats.", "An image of a bathroom sink."]
    ]
    query='What\'s the object in the image?'
    answer = interface.few_shot_generation(example_images, texts_input, query_image, query=query)
    print(f'The few-shot answer: {answer}')

    answer = interface.zero_shot_generation(query_image, query=query)
    print(f'The zero-shot anser: {answer}')