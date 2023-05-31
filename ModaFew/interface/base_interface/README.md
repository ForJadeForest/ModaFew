This is the base interface for ModaFew.



## Method
For each class, you should implement these funcions:
1. `get_model_input(self, images_list: List[List[IMAGE_TYPE]], texts_list: List[List[str]]) -> Dict:`
   - This function is to prepare the batch inputs to model. You should return one dict, where the key is the `model_forward()` parameters.
   - **images_list** `List[List[IMAGE_TYPE]]`: The batch inputs. `IMAGE_TYPE` can be path to image, PIL.Image.Image or tensor. 
   - **texts_list** `List[List[str]]`: The text for each image.

2. `model_forward(self, *args, **kwargs) -> List[str]`
   - This funcion is to do model forward and return the generated texts.

3. `postprocess(self, outputs: List[str]) -> List[str]:`
   - This funcion is to do postprocess for ouputs of `model_forward()`.


1. `few_shot_generation(self, context_images, context_texts, input_images, queries='', **kwargs)`
   - This function is to do few-shot inference. 
     - **context_images** `Union[List[List[IMAGE_TYPE]], List[IMAGE_TYPE], None]`: The few-shot images you want to use. 
     - **context_texts** `Union[List[List[dict]], List[dict], None]`: The answer of the query for each few-shot images.
     - **input_images** `Union[List[IMAGE_TYPE], IMAGE_TYPE]`: The query image. Now only support one input.
     - **query** `Union[List[dict], dict]`: The query sentence.
     - **kwargs**: The parameters for generation
   - If `context_images` and `context_texts` is None, the method will do zero-shot inference.
   