import requests
from ModaFew import FlamingoInterface, MiniGPT4Interface, MiniGPT4ChatInterface
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
example_images = [[demo_image_one, demo_image_two], [demo_image_two, demo_image_one]]
vicuna_path = '/data/share/pyz/checkpoint/vicuna-7b'
minigpt4_path = '/data/share/pyz/checkpoint/prerained_minigpt4_7b.pth'

interface = MiniGPT4Interface(device=device, vicuna_path=vicuna_path, minigpt4_path=minigpt4_path, task='caption')
query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
        stream=True
    ).raw
)
query_image = [query_image, query_image]

texts_input = [["An image of two cats.", "An image of a bathroom sink."], ["An image of a bathroom sink.", "An image of two cats."]]
texts_input = [[{'caption': i} for i in x] for x in texts_input]
query=[{'caption': None}, {'caption': None}]
# query = 'What\'s the object in the image?'
answer = interface.few_shot_generation(example_images, texts_input, query_image, queries=query,)
print(f'The few-shot answer: {answer}')