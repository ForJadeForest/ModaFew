from minigpt4_interface import MiniGPT4Interface


if __name__ == '__main__':
    import sys
    sys.path.append('/data/share/pyz/')
    import time
    import torch
    from PIL import Image
    import requests

    gpu_id = '2'
    time_begin = time.time()
    interface = MiniGPT4Interface(config_path='./minigpt4_interface/eval_config.yaml', gpu_id=gpu_id)


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
    example_images = [demo_image_one, demo_image_two]

    texts_input = ["An image of two cats.", "An image of a bathroom sink."]
    
    query='What\'s the object in the image?'
    answer = interface.few_shot_generation(example_images, texts_input, query_image, query=query)
    print(f'The few-shot answer: {answer}')

    answer = interface.zero_shot_generation(query_image, query=query)
    print(f'The zero-shot anser: {answer}')