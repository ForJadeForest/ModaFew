import pathlib
from typing import Union, List

import torch
from PIL.Image import Image
from omegaconf import OmegaConf
from transformers import StoppingCriteria, StoppingCriteriaList

from ModaFew import BaseInterface
from ModaFew.utils import image2tensor
from minigpt4.common.config import Config
from minigpt4.models import MiniGPT4
from minigpt4.processors import Blip2ImageEvalProcessor


def read_default_config(config_path, minigpt4_path, vicuna_path):
    if config_path is not None:
        config = OmegaConf.load(config_path)
        if minigpt4_path:
            config.model['ckpt'] = minigpt4_path
        if vicuna_path:
            config.model['llama_model'] = vicuna_path
        return config

    file_path = pathlib.Path(__file__).resolve()
    root_dir = file_path.parents[2]
    minigpt4_repo_path = root_dir / 'requirements_repo' / 'MiniGPT-4'
    config_path = minigpt4_repo_path / 'eval_configs' / 'minigpt4_eval.yaml'

    config = OmegaConf.load(str(config_path))

    config.model['ckpt'] = minigpt4_path
    config.model['llama_model'] = vicuna_path
    return config


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=None, encounters=1):
        super().__init__()
        if stops is None:
            stops = []
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class MiniGPT4Interface(BaseInterface):
    def __init__(self, device,
                 config_path=None,
                 minigpt4_path=None,
                 vicuna_path=None,
                 task=None,
                 **kwargs):
        super().__init__(task=task)
        config = read_default_config(config_path, minigpt4_path, vicuna_path)
        user_config = OmegaConf.create(kwargs)
        config.options = user_config

        self.device = device
        self.cfg = Config(config)
        model_config = self.cfg.model_cfg
        self.model = MiniGPT4.from_config(model_config).to(device)
        self.model.eval()

        vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = Blip2ImageEvalProcessor.from_config(vis_processor_cfg)

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.system_prompt = "Give the following image: <Img>ImageContent</Img>. " \
                             "You will be able to see the image once I provide it to you. " \
                             "Please answer my questions.###"
        print('Initialization Finished')

    @torch.no_grad()
    def get_model_input(self, images, texts):
        images = [image2tensor(img, self.vis_processor).to(self.device) for img in images]
        image_embeds, _ = self.model.encode_img(images)
        text_prompt = self.system_prompt + ''.join(texts)
        prompt_segs = text_prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(image_embeds) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embeds = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embeds = [emb for pair in zip(seg_embeds[:-1], image_embeds) for emb in pair] + [seg_embeds[-1]]
        mixed_embeds = torch.cat(mixed_embeds, dim=1)
        return mixed_embeds

    @torch.no_grad()
    def model_forward(self, input_embeds, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                      repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        current_max_len = input_embeds.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        input_embeds = input_embeds[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        return outputs

    def postprocess(self, outputs):
        processed_outputs = []
        for output in outputs:
            if output[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output = output[1:]
            if output[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output = output[1:]
            output_text = self.model.llama_tokenizer.decode(output, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            processed_outputs.append(output_text)
        return processed_outputs

    @torch.no_grad()
    def few_shot_generation(self,
                            context_images: Union[List[List[Union[Image, str, torch.Tensor]]],
                            List[Union[Image, str, torch.Tensor]]],
                            context_texts: Union[List[List[dict]], List[dict]],
                            input_images: Union[List[Union[Image, str, torch.Tensor]], Image, str, torch.Tensor],
                            queries: Union[List[dict], dict],
                            **kwargs):

        if not isinstance(input_images, list):
            input_images = [input_images]
            context_images = [context_images]
            context_texts = [context_texts]
            queries = [queries]

        batch_model_inputs = []
        batch_size = len(context_images)
        for b in range(batch_size):
            prompts = self.construct_prompt(context_texts[b], queries[b])
            image_list = context_images[b] + input_images[b]
            model_input = self.get_model_input(image_list, prompts)
            batch_model_inputs.append(model_input)
        batch_model_inputs = torch.stack(batch_model_inputs)
        outputs = self.model_forward(batch_model_inputs, **kwargs)
        outputs = self.postprocess(outputs)
        return outputs

    def construct_prompt(self,
                         example_texts: List[dict],
                         query: dict):
        prompts = self.system_prompt
        prompts_method = self.prompt_task_map[self._task]
        for text_data in example_texts:
            prompts += prompts_method(**text_data)
        prompts += prompts_method(**query)

        return prompts

    def vqa_prompt(self, question, answer=None) -> str:
        return f"Human: <Img><ImageHere></Img> {question}###Assistant:{answer if answer is not None else ''}"
