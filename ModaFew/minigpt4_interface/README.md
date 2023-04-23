# MINIGPT4-Interface
This is MINIGPT4-Interface!


## example:
```python
import time
import torch
from PIL import Image
import requests
from minigpt4_interface import MiniGPT4Interface

device = '0'
time_begin = time.time()
interface = MiniGPT4Interface(config_path='/path/to/ModaFew/ModaFew/minigpt4_interface/minigpt4/prompts/alignment.txt', device=device)

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
query='What\'s the object in the image?'
answer = interface.few_shot_generation(example_images, texts_input, query_image, query=query)
print(f'The few-shot answer: {answer}')

answer = interface.zero_shot_generation(query_image, query=query)
print(f'The zero-shot anser: {answer}')
```

## Init 
For init the Interface, you should give a config_path. The important args you should set are: prompt_path, ckpt, llama_model. 

For weight download, please refer https://github.com/Vision-CAIR/MiniGPT-4.
```yaml
model:
  arch: mini_gpt4
  model_type: pretrain_vicuna
  freeze_vit: True
  freeze_qformer: True
  max_txt_len: 160
  end_sym: "###"
  low_resource: True
  prompt_template: '###Human: {} ###Assistant: '
  # This is the default prompt path
  prompt_path: "/path/to/ModaFew/ModaFew/minigpt4_interface/minigpt4/prompts/alignment.txt"
  ckpt: "/path/to/prerained_minigpt4_7b.pth"
  llama_model: "/path/to/checkpoint/vicuna-7b"



datasets:
  cc_sbu_align:
    vis_processor:
      train:
        name: "blip2_image_eval"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"

run:
  task: image_text_pretrain
```


## Method 
1. `zero_shot_generation(self, image, query, **kwargs)`
    - This function is to do zero-shot inference. 
      - **image** `Union[Image, str, torch.Tensor]`: The Image you want to use. It can be PIL.Image.Image, the path to the image, or the processed image(torch.tensor).
      - **query**: str: The query you want to ask the model.
      - **kwargs**: The parameters for generation

2. `few_shot_generation(self, example_images, example_texts, input_images, query='', **kwargs)`
   - This function is to do few-shot inference. It will make the input as: 
        ```
        Human: <example_image1> + query + Assistant:  + example_text1 +
        Human: <example_image2> + query + Assistant:  + example_text2 ... + 
        <input_images> + query + Assistant: 
        ```

     - **example_images** `List[Union[Image, str, torch.Tensor]]`: The few-shot images you want to use.
     - **example_texts** `List[str]`: The answer of the query for each few-shot images.
     - **input_images** `Union[List[Union[Image, str, torch.Tensor]], Image, str, torch.Tensor]`: The query image. Now only support one input.
     - **query** `str`: The query sentence.
     - **kwargs**: The parameters for generation


3. `reset(self)`
   - This function is to reset the conversation history.
