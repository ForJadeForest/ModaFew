

class BaseInterface:
    def _prepare_model(*args, **kwargs):
        raise NotImplemented
    
    def few_shot_inference(self, *args, **kwargs):
        """
        1. 依据已有图像List和文本List进行prompt拼接
        2. 模型forward过程
        3. 生成句子后处理, 删除特殊的token等
        4. 模型后处理 
        """
        raise NotImplemented
    
    def zero_shot_inference(self, *args, **kwargs):
        
        raise NotImplemented
        