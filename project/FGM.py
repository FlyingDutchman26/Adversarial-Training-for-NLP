import torch

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {} # 用于保存模型扰动前的参数

    def attack(
        self, 
        epsilon=1., 
        emb_name='word_embeddings' # emb_name表示模型中embedding的参数名
    ):
        '''
        生成扰动和对抗样本
        '''
        for name, param in self.model.named_parameters(): # 遍历模型的所有参数 
            if param.requires_grad and emb_name in name: # 只取word embedding层的参数
                self.backup[name] = param.data.clone() # 保存参数值
                norm = torch.norm(param.grad) # 对参数梯度进行二范式归一化
                if norm != 0 and not torch.isnan(norm): # 计算扰动，并在输入参数值上添加扰动
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
        

    def restore(
        self, 
        emb_name='word_embeddings' # emb_name表示模型中embedding的参数名
    ):
        '''
        恢复添加扰动的参数
        '''
        for name, param in self.model.named_parameters(): # 遍历模型的所有参数
            if param.requires_grad and emb_name in name:  # 只取word embedding层的参数
                assert name in self.backup
                param.data = self.backup[name] # 重新加载保存的参数值
        self.backup = {}