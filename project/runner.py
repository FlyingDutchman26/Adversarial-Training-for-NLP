import torch    
from FGM import FGM
from PGD import PGD

class Runner():
    def __init__(self,model,optimizer,loss_fn) -> None:
        self.model =model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def train(self,train_loader,valid_loader,num_epoch=1):
        self.model.train()
        step = 0
        best_accuracy = 0
        for epoch in range(1,num_epoch+1):
            for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(train_loader):
                self.model.train()
                out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss = self.loss_fn(out,labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%10 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')

    @torch.no_grad()
    def evaluate(self,valid_loader):
        self.model.eval()
        correct = 0
        total = 0
        for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(valid_loader):
            out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            out = torch.argmax(out,dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
        return correct/total
        
    @torch.no_grad()
    def predict(self,test_loader):
        self.load_model()
        self.model.eval()
        correct = 0
        total = 0
        for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(test_loader):
            out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            out = torch.argmax(out,dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
        score = correct/total
        # print(total)
        print(f'Score on test set:{score}')
        return score
    
    def save_model(self, save_path = './modelparams/bestmodel_parms.pth'):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path='./modelparams/bestmodel_parms.pth'):
        self.model.load_state_dict(torch.load(model_path))
        
        
class Runner_FGM(Runner):
    def __init__(self,model,optimizer,loss_fn,fgm = None) -> None:
        super(Runner_FGM,self).__init__(model,optimizer,loss_fn)
        self.fgm = fgm
        
    def train(self,train_loader,valid_loader,num_epoch=1):
        self.model.train()
        step = 0
        best_accuracy = 0
        for epoch in range(1,num_epoch+1):
            for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(train_loader):
                self.model.train()
                out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss = self.loss_fn(out,labels)
                loss.backward()
                self.fgm.attack()
                out_adv = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss_adv = self.loss_fn(out_adv,labels)
                loss_adv.backward()
                self.fgm.restore()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%10 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')
                    
class Runner_PGD(Runner):
    def __init__(self,model,optimizer,loss_fn,pgd = None) -> None:
        super(Runner_PGD,self).__init__(model,optimizer,loss_fn)
        self.pgd = pgd
        
    def train(self,train_loader,valid_loader,num_epoch=1):
        self.model.train()
        step = 0
        best_accuracy = 0
        K = self.pgd.k
        for epoch in range(1,num_epoch+1):
            for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(train_loader):
                self.model.train()
                out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss = self.loss_fn(out,labels)
                loss.backward()
                self.pgd.backup_grad()
                for t in range(K):
                    self.pgd.attack(is_first_attack=(t==0))
                    if t == K-1:
                        self.pgd.restore_grad()
                    else:
                        self.optimizer.zero_grad()
                        
                    out_adv = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                    loss_adv = self.loss_fn(out_adv,labels)
                    loss_adv.backward() # 前面就按公式正常迭代梯度，最后一次在最初梯度上累计一次
                    
                self.pgd.restore()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%10 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')

class Runner_FreeLB(Runner):
    def __init__(self,model,optimizer,loss_fn,freelb = None):
        super(Runner_FreeLB,self).__init__(model,optimizer,loss_fn)
        self.freelb = freelb
    
    def train(self,train_loader,valid_loader,num_epoch=1):
        self.model.train()
        step = 0
        best_accuracy = 0
        K = self.freelb.k
        for epoch in range(1,num_epoch+1):
            for batch_id, (input_ids, attention_mask, token_type_ids,labels) in enumerate(train_loader):
                self.model.train()
                
                out = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss = self.loss_fn(out,labels)
                loss.backward()
                self.optimizer.zero_grad()
                
                for t in range(K):
                    self.freelb.backup_grad()
                    self.optimizer.zero_grad()
                    self.freelb.attack(is_first_attack=(t==0))        
                    out_adv = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                    loss_adv = self.loss_fn(out_adv,labels)
                    loss_adv.backward()
                    self.freelb.backup_r_grad()
                    self.freelb.upgrade_grad()
                    self.freelb.upgrade_r_at()
                    
                self.freelb.restore()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out = torch.argmax(out,dim=1)
                    score = (out == labels).sum()/len(labels)
                valid_accuracy = self.evaluate(valid_loader)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    self.save_model()
                    print(f'Best performance on valid set upgraded: accuracy: {best_accuracy}')
                step += 1
                if step%10 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss},[score]:{score}')
        
                


