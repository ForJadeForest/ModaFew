# ModaFew

This repo provide some interfaces for load large model and large model inference. 

The main usage is to do few-shot inference now! 



## Model Zoo

- Flamingo: https://github.com/mlfoundations/open_flamingo

- MiniGPT4: https://github.com/Vision-CAIR/MiniGPT-4

## Example

```python
from open_flamingo.load_flamingo import FlamingoInterface
device = 'cuda'
interface = FlamingoInterface(device=device)

images = [[Image]]
texts = [[Text]]
interface.generation(images, texts, max_new_tokens=20, num_beams=1)
```

## Install
```bash
git clone https://github.com/ForJadeForest/ModaFew.git
cd ModaFew
pip install -r requirements.txt
pip install -e .
```
