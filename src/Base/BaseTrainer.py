from typing import List
import weakref
from Base.BaseHook import BaseHook
import logging
from utils.metrics_util import MetricsRecorder
import torch

class BaseTrainer:
    def __init__(
        self,model,opt,train_loader,valid_loader,test_loader,device,max_epoch,valid_freq=1):
        self.hooks = []

        self.device = device
        self.model = model.to(self.device)
        self.opt = opt
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.valid_freq = valid_freq
        
        self.current_epoch = 0
        self.current_iter = 0
        self.train_recorder = MetricsRecorder()
        self.valid_recorder = MetricsRecorder()

        self.len_loader = len(train_loader)
        self.max_epoch = max_epoch


    def register_hooks(self, hooks):
        for each_hook in hooks:
            assert isinstance(each_hook, BaseHook)
            each_hook.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)
        self.print_hook_order()
    def before_train(self):
        for each_hook in self.hooks:
            each_hook.before_train()
    def after_train(self):
        for each_hook in self.hooks:
            each_hook.after_train()
    def before_iter(self):
        for each_hook in self.hooks:
            each_hook.before_iter()
    def after_iter(self):
        for each_hook in self.hooks:
            each_hook.after_iter()
    def is_pass(self,func):
        """
        return True if the func returns a pass statement
        """
        return func.__code__.co_code == (lambda: None).__code__.co_code
    def lr(
        self,
        group: List[int] = None,
        ):
        """
        get learning rate, if group is None, return a list of all groups' lr.
        """
        if group is None:
            group = range(len(self.opt.param_groups))
        return [self.opt.param_groups[i]["lr"] for i in group] 
    def print_hook_order(self):
        """
        print the order of hooks
        """
        before_train_hooks = [each_hook.__class__.__name__ for each_hook in self.hooks if not self.is_pass(each_hook.before_train)]
        before_iter_hooks = [each_hook.__class__.__name__ for each_hook in self.hooks if not self.is_pass(each_hook.before_iter)]
        after_iter_hooks = [each_hook.__class__.__name__ for each_hook in self.hooks if not self.is_pass(each_hook.after_iter)]
        after_train_hooks = [each_hook.__class__.__name__ for each_hook in self.hooks if not self.is_pass(each_hook.after_train)]

        split_str = '\n\t\t'
        summary_str = f'''
            [+]==============>| Hook Order |<=================
            [Register hooks]: {'; '.join([each_hook.__class__.__name__ for each_hook in self.hooks])}
            
            [Before Train]:
            \t{split_str.join(before_train_hooks)}

            [Before Iter]:
            \t{split_str.join(before_iter_hooks)}

            [After Iter]:
            \t{split_str.join(after_iter_hooks)}

            [After Train]:
            \t{split_str.join(after_train_hooks)}
            [-]==============>| Hook Order |<=================
            '''
        logging.info(summary_str)
    def run(self):
        self.before_train()
        for epoch in range(self.max_epoch):
            self.current_epoch = epoch
            self._iter_train_loader = iter(self.train_loader)
            for batch in range(self.len_loader):
                self.current_iter = batch + self.current_epoch * self.len_loader
                
                self.before_iter()

                # train one iter
                self.model.train()
                batch = next(self._iter_train_loader)
                self.send_batch_to_device(batch)
                self.train_one_iter(batch)
                self.model.eval()
                
                # valid if needed
                if self.current_iter % self.valid_freq == 0:
                    with torch.no_grad():
                        self.valid()
                self.after_iter() 

        self.after_train()
    
    def send_batch_to_device(self,batch):
        if isinstance(batch, dict):
            for key in batch:
                batch[key] = batch[key].to(self.device)
        elif isinstance(batch, list):
            for i in range(len(batch)):
                batch[i] = batch[i].to(self.device)
        else:
            batch = batch.to(self.device)

    def train_one_iter(self):
        raise NotImplementedError
    def valid(self):
        raise NotImplementedError