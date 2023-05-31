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

interface = MiniGPT4Interface(device=device, vicuna_path=vicuna_path, minigpt4_path=minigpt4_path, task='caption')
query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
        stream=True
    ).raw
)
texts_input = ["An image of two cats.", "An image of a bathroom sink."]
texts_input = [{'caption': x} for x in texts_input]
query={'caption': None}
answer = interface.few_shot_generation(example_images, 
                                       texts_input, 
                                       query_image, 
                                       queries=query)
print(f'The few-shot answer: {answer}')
```

## Init 
For init the Interface, you should give a config_path. 
If you not give one config_path, it will use the default config. But you should provide the `vicuna_path` and `minigpt4_path`.

For weight download, please refer https://github.com/Vision-CAIR/MiniGPT-4.

This is the default config (modify from MiniGPT-4/eval_configs/minigpt4_eval.yaml):
```yaml
model:
  arch: mini_gpt4
  model_type: pretrain_vicuna
  freeze_vit: True
  freeze_qformer: True
  max_txt_len: 160
  end_sym: "###"
  low_resource: False
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


### Note
When use batch inference, the config of vicuna-7b should modify the pad_token_id to 2. 
Because sometimes the will generate the padding token.
details: https://github.com/huggingface/transformers/issues/22546#issuecomment-1561257076

- Now the result of batch inference has some errors. Constrain the batch size to equal to 1.