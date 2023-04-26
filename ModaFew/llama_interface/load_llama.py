import torch
from typing import Tuple, Optional, List
from transformers import AutoTokenizer, LlamaForCausalLM

class llama_interface:
    def __init__(self, ckpt_dir: str, tokenizer_path: Optional[str] = None, device='cpu', precision='fp16'):
        if tokenizer_path is None:
            tokenizer_path = ckpt_dir
        self.model = LlamaForCausalLM.from_pretrained(ckpt_dir)
        self.model.eval()
        if precision == 'fp16':
            self.model = self.model.half()
            print('half done!')
        self.model = self.model.to(device)
        self._auto_cast = 'cuda' if 'cuda' in device else 'cpu'
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print('init done')

    @torch.no_grad()
    def generation(self, prompts: List[str], **kwargs):
        if not isinstance(prompts, list):
            prompts = [prompts]
        final_result = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
            with torch.autocast(self._auto_cast):
                generate_ids = self.model.generate(inputs.input_ids,  **kwargs)
            gene_texts = self.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            final_result.append(gene_texts)
        return final_result



if __name__ == '__main__':
    path = '/data/share/pyz/ModaFew/checkpoint/llama-7b'
    device = 'cuda'
    precision = 'fp16'
    interface = llama_interface(path, device=device, precision=precision)
    text = ['I love you! ']
    print(interface.generation(text))
    text = ['You are my best friend! ', 'Can you help me write a python code to read a csv file?']
    print(interface.generation(text))