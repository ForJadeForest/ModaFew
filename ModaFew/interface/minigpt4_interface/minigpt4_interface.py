import pathlib
from typing import List, Optional

import torch
from ModaFew.interface.base_interface import BaseInterface
from ModaFew.interface.utils import image2tensor
from minigpt4.common.config import Config
from minigpt4.models import MiniGPT4
from minigpt4.processors import Blip2ImageEvalProcessor
from omegaconf import OmegaConf
from transformers import StoppingCriteria, StoppingCriteriaList


def read_default_config(config_path, minigpt4_path, vicuna_path):
    if config_path is not None:
        config = OmegaConf.load(config_path)
        if minigpt4_path:
            config.model['ckpt'] = minigpt4_path
        if vicuna_path:
            config.model['llama_model'] = vicuna_path
        return config

    file_path = pathlib.Path(__file__).resolve()
    root_dir = file_path.parents[3]
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
        model_config['vit_precision'] = 'fp32'
        self.model = MiniGPT4.from_config(model_config).to(device)
        self.model.eval()

        vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = Blip2ImageEvalProcessor.from_config(
            vis_processor_cfg)

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])
        self.system_prompt = "Give the following image: <Img>ImageContent</Img>. " \
                             "You will be able to see the image once I provide it to you. " \
                             "Please answer my questions."
        print('Initialization Finished')

    @torch.inference_mode()
    def process_one_sample(self, image_embeds, prompt_segs):
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embeds = [self.model.llama_model.model.embed_tokens(
            seg_t) for seg_t in seg_tokens]
        mixed_embeds = [emb.squeeze() for pair in zip(
            seg_embeds[:-1], image_embeds) for emb in pair] + [seg_embeds[-1].squeeze()]
        mixed_embeds = torch.cat(mixed_embeds, dim=0)
        return mixed_embeds

    @torch.inference_mode()
    def get_model_input(self, images_list, texts_list):
        images_tensor_list = []
        for images in images_list:
            images = [image2tensor(img, self.vis_processor).to(
                self.device) for img in images]
            images_tensor_list.append(torch.stack(images))

        # batch, few-shot-num, 3, 224, 224
        images_tensor = torch.stack(images_tensor_list)
        images_tensor_shape = images_tensor.shape
        batch_size, shot_num = images_tensor.shape[:2]
        # assert batch_size == 1, f"the batch size should equal to 1 when use minigpt4 right now, but get {batch_size}"

        images_tensor = images_tensor.reshape(-1, *images_tensor_shape[-3:])
        images_embeds, images_attn = self.model.encode_img(images_tensor)

        images_embeds = images_embeds.reshape(
            batch_size, shot_num, *images_embeds.shape[1:])
        images_attn = images_attn.reshape(batch_size, shot_num, -1)

        prompt_segs = [t.split('<ImageHere>') for t in texts_list]

        input_embeds = []
        for i in range(batch_size):
            mixed_embeds = self.process_one_sample(
                images_embeds[i], prompt_segs[i])
            input_embeds.append(mixed_embeds)

        padding_idx = self.model.llama_tokenizer.pad_token_id
        padding_value = self.model.llama_model.model.embed_tokens(
            torch.tensor(padding_idx, device=self.device))

        input_embeds, attention_mask = self.pad_sequence_with_mask(
            input_embeds, padding_value)

        input_dict = {
            'input_embeds': input_embeds,
            'attention_mask': attention_mask
        }
        return input_dict

    @torch.inference_mode()
    def model_forward(self, input_embeds, attention_mask=None, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                      repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        current_max_len = input_embeds.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        input_embeds = input_embeds[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
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
            # some users find that there is a start token <s> at the beginning. remove it
            if output[0] == 1:
                output = output[1:]
            output_text = self.model.llama_tokenizer.decode(
                output, add_special_tokens=False)
            output_text = output_text.split(
                '###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_text = output_text.replace('#', '')
            processed_outputs.append(output_text)
        return processed_outputs

    def construct_prompt(self,
                         context_texts: Optional[List[dict]],
                         query: dict):
        prompts = self.system_prompt
        prompts += super().construct_prompt(context_texts, query)
        return prompts

    @staticmethod
    def vqa_prompt(question, answer=None) -> str:
        return f"###Human: <Img><ImageHere></Img> Please answer the question shortly: {question}###Assistant:{answer if answer is not None else ''}"

    @staticmethod
    def caption_prompt(caption=None) -> str:
        return f"###Human: <Img><ImageHere></Img> Please describe the image detailed.###Assistant:{caption if caption is not None else ''}"

    def pad_sequence_with_mask(self, sequences, padding_value):
        # 计算最长序列的长度
        max_len = max([seq.shape[0] for seq in sequences])
        embedding_dim = sequences[0].shape[1]
        # 创建一个形状为(len(sequences), max_len, 128)的零张量
        padded_sequence = torch.zeros(
            (len(sequences), max_len, embedding_dim), device=self.device)
        # 创建一个形状为(len(sequences), max_len)的零张量
        mask = torch.ones((len(sequences), max_len), device=self.device)
        # 填充序列和掩码张量
        for i, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            padding_length = max_len - seq_len
            padded_sequence[i, padding_length:, :] = seq
            padded_sequence[i, :padding_length, :] = padding_value
            mask[i, seq_len:] = 0

        return padded_sequence, mask
