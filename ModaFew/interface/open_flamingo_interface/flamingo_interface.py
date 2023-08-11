from typing import List, Dict

import torch
from torch import autocast

from huggingface_hub import hf_hub_download

from ModaFew.interface.base_interface import BaseInterface
from ModaFew.interface.utils import IMAGE_TYPE, image2tensor, cast_type
from open_flamingo import create_model_and_transforms


class FlamingoInterface(BaseInterface):
    MODELNAME_LIST = [
        'OpenFlamingo-3B-vitl-mpt1b',
        'OpenFlamingo-3B-vitl-mpt1b-langinstruct',
        'OpenFlamingo-4B-vitl-rpj3b',
        'OpenFlamingo-4B-vitl-rpj3b-langinstruct',
        'OpenFlamingo-9B-vitl-mpt7b',
        'OpenFlamingo-9B-deprecated'
    ]

    def __init__(self,
                 flamingo_checkpoint_path,
                 lang_encoder_path,
                 tokenizer_path,
                 hf_root,
                 cross_attn_every_n_layers=1,
                 precision="fp16",
                 device="cuda",
                 task=None):
        super().__init__(task=task)
        assert hf_root in self.MODELNAME_LIST, f'The hf_root should in {self.MODELNAME_LIST}, but got {hf_root}'
        hf_root = 'openflamingo/' + hf_root
        flamingo_checkpoint_path = hf_hub_download(
            hf_root, "checkpoint.pt", local_dir=flamingo_checkpoint_path)

        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers
        )

        self.data_type = cast_type(precision)
        self.device = device
        self.autocast_args = {
            'device_type': 'cuda' if 'cuda' in device else 'cpu',
            'dtype': self.data_type
        }
        # load model weight
        model_state = torch.load(flamingo_checkpoint_path)
        self.model.load_state_dict(model_state, strict=False)
        self.model = self.model.to(device, dtype=self.data_type)
        self.model.eval()
        self.tokenizer.padding_side = "left"

    def get_model_input(self, images_list: List[List[IMAGE_TYPE]], texts_list: List[List[str]]) -> Dict:
        image_tensors = self.process_batch_image(images_list).to(self.device)

        texts_token = self.tokenizer(
            texts_list, return_tensors="pt").to(self.device)
        return {
            'vision_x': image_tensors,
            'lang_x': texts_token['input_ids'],
            'attention_mask': texts_token['attention_mask']
        }

    @torch.inference_mode()
    def model_forward(self, vision_x, lang_x, attention_mask, **kwargs):
        with autocast(**self.autocast_args):
            outputs = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                **kwargs
            )
            outputs = outputs[:, len(lang_x[0]):]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def cat_single_image(self, images: List[IMAGE_TYPE]):
        """
        @param: images: A List[Image] for processing
        return: A tensor with [1, len(images), 1, 3, 224, 224]
        """
        image_tensor = [image2tensor(
            i, self.image_processor).unsqueeze(0) for i in images]
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
