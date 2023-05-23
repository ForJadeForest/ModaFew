class BaseInterface:
    def __int__(self, task):
        self._task = task
        self._default_task_map = {
            'vqa': self.vqa_prompt,
            'caption': self.caption_prompt
        }
        self.prompt_task_map = self._default_task_map

    def get_model_input(self, images, texts):
        raise NotImplemented

    def model_forward(self, *args, **kwargs):
        raise NotImplemented

    def postprocess(self, outputs):
        raise NotImplemented

    def construct_prompt(self, *args, **kwargs):
        raise NotImplemented

    @staticmethod
    def vqa_prompt(*args, **kwargs):
        raise NotImplemented

    @staticmethod
    def caption_prompt(*args, **kwargs):
        raise NotImplemented

    def add_task(self, task, prompt_method):
        self.prompt_task_map[task] = prompt_method

    @property
    def get_task_map(self):
        return self.prompt_task_map.keys()

