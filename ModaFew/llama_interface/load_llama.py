from typing import Tuple, Optional
from transformers import AutoTokenizer, LlamaForCausalLM

class llama_interface:
    def __init__(self, ckpt_dir: str, tokenizer_path: Optional[str] = None):
        if tokenizer_path is None:
            tokenizer_path = ckpt_dir
        self.model = LlamaForCausalLM.from_pretrained(ckpt_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def few_shot_generation(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        generate_ids = self.model.generate(inputs.input_ids,  **kwargs)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        
