from typing import List, Dict

import torch
from huggingface_hub import hf_hub_download

from ModaFew.base_interface import BaseInterface
from ModaFew.utils import IMAGE_TYPE, image2tensor
from open_flamingo import create_model_and_transforms


class FlamingoInterface(BaseInterface):
    def __init__(self,
                 clip_vision_encoder_path='ViT-L-14',
                 clip_vision_encoder_pretrained='openai',
                 lang_encoder_path='checkpoint/llama-7b',
                 tokenizer_path='checkpoint/llama-7b',
                 cross_attn_every_n_layers=4,
                 inference=True,
                 precision="fp16",
                 device='cuda',
                 checkpoint_path='checkpoint/openflamingo/',
                 task=None):
        super().__init__(task=task)

        model_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt", local_dir=checkpoint_path)
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=clip_vision_encoder_path,
            clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            inference=inference,
            precision=precision,
            device=device,
            checkpoint_path=model_path
        )

        self.precision = precision
        self.device = device
        self.tokenizer.padding_side = "left"

    def get_model_input(self, images_list: List[List[IMAGE_TYPE]], texts_list: List[List[str]]) -> Dict:
        image_tensors = self.process_batch_image(images_list).to(self.device)
        if self.precision == 'fp16':
            image_tensors = image_tensors.half()
        texts_token = self.tokenizer(texts_list, return_tensors="pt").to(self.device)
        return {
            'vision_x': image_tensors,
            'lang_x': texts_token['input_ids'],
            'attention_mask': texts_token['attention_mask']
        }

    def model_forward(self, vision_x, lang_x, attention_mask, **kwargs):
        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            **kwargs
        )
        outputs = generated_text[:, len(lang_x[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def cat_single_image(self, images: List[IMAGE_TYPE]):
        """
        @param: images: A List[Image] for processing
        return: A tensor with [1, len(images), 1, 3, 224, 224]
        """
        image_tensor = [image2tensor(i, self.image_processor).unsqueeze(0) for i in images]
        image_tensor = torch.cat(image_tensor, dim=0)
        image_tensor = image_tensor.unsqueeze(1).unsqueeze(0)
        return image_tensor


    def process_batch_image(self, batch_images: List[List[IMAGE_TYPE]]):
        """
        @param: batch_images: A List[List[Image]], every element is a few-shot images and one query image
        return A tensor with [batch_size, few-shot_num, 1, 3, 224, 224]
        """
        batch_tensor = []
        for images in batch_images:
            image_tensor = self.cat_single_image(images)
            batch_tensor.append(image_tensor)
        batch_tensor = torch.cat(batch_tensor, dim=0)
        return batch_tensor
    
    
    @staticmethod
    def vqa_prompt(question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    @staticmethod
    def caption_prompt(caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    @staticmethod
    def classification_prompt(self, class_str=None) -> str:
        return f"<image>A photo of a {class_str if class_str is not None else ''}{'<|endofchunk|>' if class_str is not None else ''}"
