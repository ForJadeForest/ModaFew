import torch

from typing import List

from PIL.Image import Image

from .open_flamingo import Precision_MODE
from .open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download


class FlamingoInterface:
    def __init__(self,
                 clip_vision_encoder_path='ViT-L-14',
                 clip_vision_encoder_pretrained='openai',
                 lang_encoder_path='checkpoint/llama-7b',
                 tokenizer_path='checkpoint/llama-7b',
                 cross_attn_every_n_layers=4,
                 inference=True,
                 precision: Precision_MODE = "fp16",
                 device='cuda',
                 checkpoint_path='checkpoint/openflamingo/'):
        model_path =hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt", local_dir=checkpoint_path)
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
        image_tensor = [self.image_processor(i).unsqueeze(0) for i in images]
        image_tensor = torch.cat(image_tensor, dim=0)
        image_tensor = image_tensor.unsqueeze(1).unsqueeze(0)
        return image_tensor

    def process_batch_image(self, batch_images: List[List[Image]]):
        batch_tensor = []
        for images in batch_images:
            image_tensor = self.cat_single_image(images)
            batch_tensor.append(image_tensor)
        batch_tensor = torch.cat(batch_tensor, dim=0)
        return batch_tensor

    def cat_single_text(self, texts: List[str], query_prefix: str = ''):
        prompt = ''
        for text in texts:
            prompt += (self.image_token + text + self.chunk_token)

        prompt += self.image_token
        prompt += query_prefix
        return prompt

    def process_batch_text(self, batch_texts, query_prefix: str = ''):
        cat_prompt_list = []
        for texts in batch_texts:
            text_prompt = self.cat_single_text(texts, query_prefix)
            cat_prompt_list.append(text_prompt)
        return self.tokenizer(cat_prompt_list, return_tensors="pt")

    def generate(self, images, texts, query_prefix='', **kwargs):
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


if __name__ == '__main__':
    import time
    import torch

    from PIL import Image
    import requests


    device = 'cuda:0'
    time_begin = time.time()
    interface = FlamingoInterface(device=device)


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

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
     batch_size x num_media x num_frames x channels x height x width. 
     In this case batch_size = 1, num_media = 3, num_frames = 1 
     (this will always be one expect for video which we don't support yet), 
     channels = 3, height = 224, width = 224.
             
            For Interface, you just only use the process_batch_image function! 
            The input are List[List[Image]]
    """
    image_input = [[demo_image_one, demo_image_two, query_image]]
    image_input = interface.process_batch_image(image_input)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
     We also expect an <|endofchunk|> special token to indicate the end of the text 
     portion associated with an image.
            For Interface, you just only use the process_batch_text function!
            The Inputs are List[List[str]]. The query_prefix is the query prompt.
    """
    texts_input = [
        ["An image of two cats.", "An image of a bathroom sink."]
    ]
    texts_input = interface.process_batch_text(texts_input, query_prefix='An image of ')

    """
    Step 4: Generate text
    """
    generated_text = interface.generate(
        image_input,
        texts_input,
        max_new_tokens=20,
        num_beams=3,
    )

    print("Generated text: ", interface.tokenizer.decode(generated_text[0]), "time: ", time.time() - time_begin)
