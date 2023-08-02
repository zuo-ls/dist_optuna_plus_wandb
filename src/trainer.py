import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn.functional as F
import torch.optim as optim
from Base.BaseTrainer import BaseTrainer
from mnist.data_mnist import get_dataloaders
from mnist.model_mnist import LSTM_mnist
from Hooks.LogHook import LogHook


class Trainer(BaseTrainer):
    def __init__(self,model,opt,train_loader,valid_loader,test_loader,device,max_epoch,valid_freq=1):
        super(Trainer,self).__init__(model,opt,train_loader,valid_loader,test_loader,device,max_epoch,valid_freq)
    
    def train_one_iter(self,batch):
        x,y = batch

        logits = self.model(x)
        loss = F.cross_entropy(logits,y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.train_out = {
            "loss":loss,
            "acc":acc
        }
        
    def valid(self):
        # whole valid process rather than one valid iter
        valid_x = self.valid_loader.dataset.data.to(self.device)
        valid_label = self.valid_loader.dataset.targets.to(self.device)

        valid_logits = self.model(valid_x)
        valid_loss = F.cross_entropy(valid_logits,valid_label)
        valid_acc = (valid_logits.argmax(dim=1) == valid_label).float().mean()

        self.valid_out = {
            "loss":valid_loss,
            "acc":valid_acc
        }

    def test(self):
        # whole test process rather than one test iter
        test_x = self.test_loader.dataset.data.to(self.device)
        test_label = self.test_loader.dataset.targets.to(self.device)

        test_logits = self.model(test_x)
        test_loss = F.cross_entropy(test_logits,test_label)
        test_acc = (test_logits.argmax(dim=1) == test_label).float().mean()

        self.test_out = {
            "loss":test_loss.item(),
            "acc":test_acc.item()
        }

        print("test loss: {:.4f}, test acc: {:.4f}".format(test_loss,test_acc))
    

if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_dataloaders(
        train_bs=128, valid_bs=128, test_bs=128,
        n_train=10000, n_valid=500, n_test=500,
    )
    model = LSTM_mnist()
    opt = optim.Adam(model.parameters(), lr=1e-2)

    trainer = Trainer(
        model=model,
        opt=opt,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_epoch=5,
        valid_freq=10,
    )

    trainer.register_hooks([
        LogHook()
    ])
    trainer.run()
    trainer.test()