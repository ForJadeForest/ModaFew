# ModaFew

This repo provide some interfaces for load large model and large model inference. 

The main usage is to do few-shot inference now! 



## Model Zoo

- Flamingo: https://github.com/mlfoundations/open_flamingo

- MiniGPT4: https://github.com/Vision-CAIR/MiniGPT-4

## Example

```python
from ModaFew import FlamingoInterface, MiniGPT4Interface
device = 'cuda'

lang_encoder_path = "path/to/llama-7b"
tokenizer_path = "path/to/llama-7b"
checkpoint_path = "path/to/openflamingo_checkpoint"

interface = FlamingoInterface(device=device,
                              lang_encoder_path=lang_encoder_path,
                              tokenizer_path=tokenizer_path,
                              checkpoint_path=checkpoint_path)

images = [[Image]]
texts = [[Text]]
interface.generation(images, texts, max_new_tokens=20, num_beams=1)

# ==============================MiniGPT4Interface===================================
device = '0'
interface = MiniGPT4Interface(config_path='/path/to/ModaFew/ModaFew/minigpt4_interface/minigpt4/prompts/alignment.txt', device=device)

example_images = [demo_image_one, demo_image_two]

texts_input = ["An image of two cats.", "An image of a bathroom sink."]
query='What\'s the object in the image?'
answer = interface.few_shot_generation(example_images, texts_input, query_image, query=query)
print(f'The few-shot answer: {answer}')
```

## Install
```bash
git clone https://github.com/ForJadeForest/ModaFew.git
cd ModaFew
pip install -r requirements.txt
pip install -e .
```
