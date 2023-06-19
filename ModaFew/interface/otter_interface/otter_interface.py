import torch
import transformers

from typing import List, Dict
from otter.modeling_otter import OtterForConditionalGeneration
from ModaFew.interface.base_interface import BaseInterface
from ModaFew.interface.utils import IMAGE_TYPE, image2tensor


class OtterInterface(BaseInterface):
    def __init__(self, model_path, precision, device, task):
        super(OtterInterface).__init__(task)
        if precision == 'fp16':
            m_type = torch.float16
        elif precision == 'bf16':
            m_type = torch.bfloat16
        elif precision == 'fp32':
            m_type = torch.float32
        else:
            m_type = torch.float16
            print(f'precision got None value or error value: {precision}, now use fp16')
        self.model = OtterForConditionalGeneration.from_pretrained(model_path, 
                                                                   torch_dtype=m_type, 
                                                                   device_map="auto")
        self.image_processor = transformers.CLIPImageProcessor()
        self.tokenizer = self.model.tokenizer
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
        outputs = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            **kwargs
        )
        outputs = outputs[:, len(lang_x[0]) :]

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
    def classification_prompt(class_str=None) -> str:
        return f"<image>A photo of a {class_str if class_str is not None else ''}{'<|endofchunk|>' if class_str is not None else ''}"
