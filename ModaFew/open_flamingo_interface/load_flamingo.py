import torch

from typing import List, Union

from PIL.Image import Image

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download


class FlamingoInterface:
    def __init__(self,
                 clip_vision_encoder_path='ViT-L-14',
                 clip_vision_encoder_pretrained='openai',
                 lang_encoder_path='checkpoint/llama-7b',
                 tokenizer_path='checkpoint/llama-7b',
                 cross_attn_every_n_layers=4,
                 inference=True,
                 precision= "fp16",
                 device='cuda',
                 checkpoint_path='checkpoint/openflamingo/'):
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
        print('Load model sucessfully!')

        self.image_token = '<image>' 
        self.chunk_token = '<|endofchunk|>'
        self.precision = precision
        self.device = device
 
    def cat_single_image(self, images: List[Image]):
        """
        @param: images: A List[Image] for processing
        return: A tensor with [1, len(images), 1, 3, 224, 224]
        """
        image_tensor = [self.image_processor(i).unsqueeze(0) for i in images]
        image_tensor = torch.cat(image_tensor, dim=0)
        image_tensor = image_tensor.unsqueeze(1).unsqueeze(0)
        return image_tensor

    def process_batch_image(self, batch_images: List[List[Image]]):
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

    def cat_single_text(self, texts: List[str], query_prefix: str = ''):
        """
        @param: texts: A List[str], every element is a text for few-shot image,
                       the last element is the query text.
        @param: query_prefix: A str, It's a prefix for query text
        return: the prompt
        """
        prompt = ''
        for text in texts:
            prompt += (self.image_token + text + self.chunk_token)

        prompt += self.image_token
        prompt += query_prefix
        return prompt

    def process_batch_text(self, 
                           batch_texts: List[List[str]], query_prefix: str = ''):
        """
        @param: batch_texts: A List[List[str]], every element is a few-shot text and the last one is query text
        @param: query_prefix: A str, It's a prefix for query text.
        return: the tensor with [batch, seq_lenght]
        """
        cat_prompt_list = []
        for texts in batch_texts:
            text_prompt = self.cat_single_text(texts, query_prefix)
            cat_prompt_list.append(text_prompt)
        return self.tokenizer(cat_prompt_list, return_tensors="pt")

    def generate(self, 
                 images: Union[List[List[Image]], torch.Tensor], 
                 texts: Union[List[List[str]], torch.Tensor], 
                 query_prefix: str = '', 
                 **kwargs):
        self.tokenizer.padding_side = "left"
        if isinstance(images, list):
            images = self.process_batch_image(images).to(self.device)
        if isinstance(texts, list):
            assert query_prefix, f"If the texts is List[List[str]], query_prefix must provide "
            texts = self.process_batch_text(texts, query_prefix)
        images = images.to(self.device)
        texts = texts.to(self.device)
        if self.precision == 'fp16':
            images = images.half()
        generated_text = self.model.generate(
            vision_x=images,
            lang_x=texts["input_ids"],
            attention_mask=texts["attention_mask"],
            **kwargs
        )
        return generated_text


