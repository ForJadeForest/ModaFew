# MINIGPT4-Interface
This is MINIGPT4-Interface!


## example:
```python
import requests
from ModaFew import FlamingoInterface, MiniGPT4Interface
from PIL import Image
device = 'cuda:0'

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
example_images = [demo_image_one, demo_image_two]
vicuna_path = '/data/share/pyz/checkpoint/vicuna-7b'
minigpt4_path = '/data/share/pyz/checkpoint/prerained_minigpt4_7b.pth'

interface = MiniGPT4Interface(device=device, vicuna_path=vicuna_path, minigpt4_path=minigpt4_path)
query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
        stream=True
    ).raw
)
texts_input = ["An image of two cats.", "An image of a bathroom sink."]
query='What\'s the object in the image?'
answer = interface.few_shot_generation(example_images, texts_input, query_image, query=query)
print(f'The few-shot answer: {answer}')
```

## Init 
For init the Interface, you should give a config_path. 
If you not give one config_path, it will use the default config. But you should provide the `vicuna_path` and `minigpt4_path`.

For weight download, please refer https://github.com/Vision-CAIR/MiniGPT-4.

This is the default config:
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


### Note
When use batch inference, the config of vicuna-7b should modify the pad_token_id to 2. 
Because sometimes the will generate the padding token.
details: https://github.com/huggingface/transformers/issues/22546#issuecomment-1561257076