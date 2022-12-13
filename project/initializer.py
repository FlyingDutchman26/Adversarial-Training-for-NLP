import torch
from datasets import load_from_disk
import random
import config
from transformers import BertTokenizer
from transformers import BertModel
#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self,split = 'train'): 
        dataset = load_from_disk('./data/ChnSentiCorp')
        self.dataset = dataset[split]
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label

def setup_seed(seed=0):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


args = {
    'max_length':config.max_length,
    'batch_size':config.batch_size,
    'epoch':config.epoch,
    'init_lr':config.init_lr,
    'seed':config.seed,
    'device':config.device,
}

device =  args['device'] if torch.cuda.is_available() else 'cpu'

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)
        # BertModel.from_pretrained 返回一个 BertModel 类的对象，其本质其实也是一个继承于nn.module的类
        self.pretrained_bert = BertModel.from_pretrained('bert-base-chinese')
        self.pretrained_bert.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 全部参与训练
        out = self.pretrained_bert(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0]) # 只要CLS:[batch_size,768]

        return out

def collate_fn(data):# 接收一批来自dataset的数据 batch_size 个 sentence 和 label，统一进行处理(转化为bert可接收的编码)
    sents = [i[0] for i in data]    # sentence
    labels = [i[1] for i in data]   # labels

    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=args['max_length'],
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids 就是编码后的词
    #token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
    #attention_mask pad的位置是0,其他位置是1
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels


print('device=', device)
# 初始化随机种子
setup_seed(args['seed'])
#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')
#数据加载器
loader = torch.utils.data.DataLoader(dataset=Dataset('train'),
                                     batch_size=args['batch_size'],
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
loader_valid = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=1200,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=False)
loader_test = torch.utils.data.DataLoader(dataset=Dataset('test'),
                                              batch_size=1200,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                            drop_last=False)
