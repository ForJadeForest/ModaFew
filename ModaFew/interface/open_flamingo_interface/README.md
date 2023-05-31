# Flamingo-Interface
This is Flamingo-Interface! 


## example:
```python
import time
import torch
import requests

from PIL import Image
from ModaFew import FlamingoInterface, MiniGPT4Interface


device = 'cuda:0'
time_begin = time.time()
lang_encoder_path = "/path/to/llama-7b"
tokenizer_path = "/path/to/llama-7b"
checkpoint_path = "/path/to/openflamingo/"

interface = FlamingoInterface(device=device,
                              lang_encoder_path=lang_encoder_path,
                              tokenizer_path=tokenizer_path,
                              checkpoint_path=checkpoint_path,
                              task='caption')

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
texts_input = ["An image of two cats.", "An image of a bathroom sink."]
texts_input = [{'caption': x} for x in texts_input]

query={'caption': None}

answer = interface.few_shot_generation(example_images, 
                                       texts_input, 
                                       query_image, 
                                       queries=query,
                                       max_new_tokens=10,
                                       num_beams=1,
                                       length_penalty=-2)
print(f'The few-shot answer: {answer}')

```


## Note
The first you use the flamingo you need login huggingface account.
please click this url to check more details: https://huggingface.co/docs/huggingface_hub/quick-start#login