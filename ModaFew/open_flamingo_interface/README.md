# MINIGPT4-Interface
This is Flamingo-Interface!


## example:
```python

import time
import torch
import requests

from PIL import Image



device = 'cuda:0'
time_begin = time.time()
lang_encoder_path = "path/to/llama-7b"
tokenizer_path = "path/to/llama-7b"
checkpoint_path = "path/to/openflamingo_checkpoint"

interface = FlamingoInterface(device=device,
                              lang_encoder_path=lang_encoder_path,
                              tokenizer_path=tokenizer_path,
                              checkpoint_path=checkpoint_path)


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

```


## Method 
1. `generate(self, image, texts, query_prefix, **kwargs)`
    - This function is to do few-shot inference or zero-shot inference
      - **image** `Union[List[List[Image]], torch.Tensor]`: The Image you want to use. The each element is few-shot images and query images. If you want to do zero-shot inference, just add one Image. 
      - **texts**: `Union[List[List[str]], torch.Tensor]`: A List[str], every element is a text for few-shot image, the last element is the query text. If you want to do zero-shot inference, just set empty List.
      - **query_prefix**: `str`: The query text.
      - **kwargs**: The parameters for generation

