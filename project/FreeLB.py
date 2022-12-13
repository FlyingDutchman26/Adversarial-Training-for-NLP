import torch
import math
'''
算法流程：
对于每个输入x:
  1、通过均匀分布初始化扰动r，初始化梯度g为0，设置步数为K
  对于每步t=1...K:
    2、根据x+r计算前向loss和后向梯度g1，累计梯度g=g+g1/k
    3、更新扰动r，更新方式跟PGD一样
  4、根据g更新梯度
'''
class FreeLB():
    def __init__(self,model,k = 3,epsilon=0.1,alpha=1e-2):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.r_at = None
        self.r_grad_backup = None
        self.k = k  # 每次对抗训练迭代步数
        self.epsilon = epsilon
        self.alpha = alpha

        
    def attack(self,emb_name = 'word_embeddings', is_first_attack = False):
        '''
        算法流程中：
        在runner中先进行初始的前向计算和反向传播,然后需保存一下原始梯度,然后清零一下梯度
        之后再循环执行此attack
        '''
        for name,param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] =  param.data.clone() # 对抗训练目标是： 在原始参数上 更新 添加扰动后的梯度，因此先保存原始embedding层参数
                    self.r_at = torch.zeros_like(param.data.clone()).uniform_(-self.epsilon,self.epsilon)/math.sqrt(21128*768)
                param.data.add_(self.r_at)
                param.data = self.project(name, param.data, self.epsilon)
                
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'pooler.dense' not in name:
                # pretrained_bert.pooler.dense.weight的梯度输出是None 这层是不算进去的
                self.grad_backup[name] = param.grad.clone()
                             
    def backup_r_grad(self,emb_name = 'word_embeddings'):
        for name,param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.r_grad_backup = param.grad.clone()
                         

    def restore(self,emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self,param_name,param_data,epsilon):
        r = param_data - self.emb_backup[param_name]
        n_r = torch.norm(r).data
        # print('norm(r) = :',n_r)
        if n_r > epsilon:
            # print('Projected')
            r = epsilon * r / n_r
        return self.emb_backup[param_name] + r
                
    def upgrade_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'pooler.dense' not in name:
                param.grad = param.grad.clone()/(self.k) + self.grad_backup[name]
    
        
    def upgrade_r_at(self,emb_name = 'word_embeddings'):
        norm = torch.norm(self.r_grad_backup)
        if norm != 0 and not torch.isnan(norm):
            self.r_at = self.alpha * self.r_grad_backup.data/norm

        