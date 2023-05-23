import os
import torch
import pathlib

from ModaFew.utils import image2tensor
from typing import Union, List
from PIL.Image import Image
from omegaconf import OmegaConf

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from minigpt4.models import MiniGPT4
from minigpt4.processors import Blip2ImageEvalProcessor


def read_default_config(config_path, prompt_path, minigpt4_path, vicuna_path):
    if config_path is not None:
        config = OmegaConf.load(config_path)
        if minigpt4_path:
            config.model['ckpt'] = minigpt4_path
        if vicuna_path:
            config.model['llama_model'] = vicuna_path
        return config

    file_path = pathlib.Path(__file__).resolve()
    root_dir = file_path.parents[2]
    minigpt4_repo_path = root_dir / 'requirements_repo' / 'MiniGPT-4'
    config_path = minigpt4_repo_path / 'eval_configs' / 'minigpt4_eval.yaml'

    config = OmegaConf.load(str(config_path))
    
    config.model['ckpt'] = minigpt4_path
    config.model['llama_model'] = vicuna_path
    return config


class MiniGPT4ChatInterface:
    def __init__(self, device, 
                 config_path=None, 
                 prompt_path=None,
                 minigpt4_path=None,
                 vicuna_path=None,
                 **kwargs):
        config = read_default_config(config_path, prompt_path, minigpt4_path, vicuna_path)
        user_config = OmegaConf.create(kwargs)
        config.options = user_config
        
        self.cfg = Config(config)
        model_config = self.cfg.model_cfg
        self.model = MiniGPT4.from_config(model_config).to(device)
        self.model.eval()
        
        vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = Blip2ImageEvalProcessor.from_config(vis_processor_cfg)
        self.chat = Chat(self.model, vis_processor, device=device)
        
        self.conversation_history = CONV_VISION.copy()
        self.image_list = []
        print('Initialization Finished')

    
    def reset(self):
        self.conversation_history = CONV_VISION.copy()
        self.image_list = []

    def _chat_one_time(self, image, query, **kwargs):
        if image:
            self.chat.upload_img(image, self.conversation_history, self.image_list)
        self.chat.ask(query, self.conversation_history)
        answer = self.chat.answer(self.conversation_history, self.image_list, **kwargs)[0]
        return answer
        
    @torch.no_grad()
    def zero_shot_generation(self, 
                             image: Union[Image, str, torch.Tensor], 
                             query: str ='', 
                             **kwargs):
        assert image, f"In zero_shot_generation function, the image should be Union[Image, str, torch.Tensor], but got {type(image)}"
        answer = self._chat_one_time(image, query, **kwargs)
        self.reset()
        return answer

    @torch.no_grad()
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

        assert len(input_images) == 1, f"Now only support one image as input"

        for input_image in input_images:
            self.chat.upload_img(input_image, self.conversation_history, self.image_list)
            self.chat.ask(query, self.conversation_history)
            output_text = self.chat.answxer(self.conversation_history, self.image_list, **kwargs)[0]
        self.reset()
        return output_text
    
    def vqa_prompt(self, question, answer=None) -> str:
        return f"Question:{question} Short answer:{answer if answer is not None else ''}"



class MiniGPT4Interface:
    def __init__(self, device, 
                 config_path=None, 
                 prompt_path=None,
                 minigpt4_path=None,
                 vicuna_path=None,
                 **kwargs):
        config = read_default_config(config_path, prompt_path, minigpt4_path, vicuna_path)
        user_config = OmegaConf.create(kwargs)
        config.options = user_config
        
        self.device = device
        self.cfg = Config(config)
        model_config = self.cfg.model_cfg
        self.model = MiniGPT4.from_config(model_config).to(device)
        self.model.eval()
        
        vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = Blip2ImageEvalProcessor.from_config(vis_processor_cfg)
        # self.chat = Chat(self.model, svis_processor, device=device)
        
        self.conversation_history = CONV_VISION.copy()
        self.image_list = []
        self.system_prompt = "Give the following image: <Img>ImageContent</Img>. "\
                            "You will be able to see the image once I provide it to you. Please answer my questions.###"
        print('Initialization Finished')

    
    def reset(self):
        self.conversation_history = CONV_VISION.copy()
        self.image_list = []

    def _chat_one_time(self, image, query, **kwargs):
        if image:
            self.chat.upload_img(image, self.conversation_history, self.image_list)
        self.chat.ask(query, self.conversation_history)
        answer = self.chat.answer(self.conversation_history, self.image_list, **kwargs)[0]
        return answer



    def process_context(self, context_images, context_texts, input_images, query):
        """
        对batch中一个元素生成图像的embedding
        """
        context_images = [image2tensor(img, self.vis_processor).to(self.device)  for img in context_images]
        context_images = torch.stack(context_images, dim=0)
        image_embeds, _ = self.model.encode_img(context_images)
        assert len(image_embeds.shape) == 2
        
        text_prompt = self.system_prompt
        text_prompt = ''.join(text_prompt)
        
        prompt_segs = text_prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(image_embeds) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], image_embeds) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
        
        
        
        

    @torch.no_grad()
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

        assert len(input_images) == 1, f"Now only support one image as input"

        for input_image in input_images:
            self.chat.upload_img(input_image, self.conversation_history, self.image_list)
            self.chat.ask(query, self.conversation_history)
            output_text = self.chat.answxer(self.conversation_history, self.image_list, **kwargs)[0]
        self.reset()
        return output_text
    
    def vqa_prompt(self, question, answer=None) -> str:
        return f"Question:{question} Short answer:{answer if answer is not None else ''}"


    
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
Prompt = Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###
Human: <Img><ImageHere></Img> Describe the image###Assistant:
prompt_segs = ["Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img>", "</Img> Describe the image###Assistant:"]

after get the answer 
[["Human", "<Img><ImageHere></Img> Describe the image" ], ["Assistant", "This is a cat!"]]

4) ask again
[["Human", "<Img><ImageHere></Img> Describe the image" ], ["Assistant", "This is a cat!"], ["Human", "Describe the image again"]]

"""
