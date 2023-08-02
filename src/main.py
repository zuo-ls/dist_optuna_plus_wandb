import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch.optim as optim
from mnist.data_mnist import get_dataloaders
from mnist.model_mnist import LSTM_mnist
from Base.OptunawandbBase import OptunawandbBase
from trainer import Trainer

class ParamTune(OptunawandbBase):

    def load_items(self,**kwargs):
        train_loader, valid_loader, test_loader = get_dataloaders(
        train_bs=128, valid_bs=128, test_bs=128,
        n_train=10000, n_valid=500, n_test=500,
    )
        model = LSTM_mnist(
            input_size = kwargs["input_size"],
            hidden_size = kwargs["hidden_size"],
            num_layers = kwargs["num_layers"],
        )
        opt = optim.Adam(model.parameters(), lr=kwargs["lr"])

        trainer = Trainer(
            model=model,
            opt=opt,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=kwargs["device"],
            max_epoch=5,
            valid_freq=10,
        )
        return trainer
    def get_suggested_params(self, trial):
        """returns a dict of suggested params"""
        trial_params = dict(
    		lr = trial.suggest_loguniform('lr', 1e-4,1e-1),
      	    hidden_size = trial.suggest_categorical('hidden_size', [32,64,128])
        )
        return trial_params   
    
    def get_other_cfg(self):
        """returns a dict of other cfg"""
        return dict(
            input_size = 28,
            num_layers = 2,
        )

paramtuner = ParamTune(
	proj_name = 'myproj',
  	exp_purpose = 'Distributed Optuna plus WandB: test the code.',
  	do_init_wandb = True, # whether to init wandb, set to False if you want to run the code locally
  	metric = 'valid_acc', # metric to be optimized, should be included in self.valid_out in the method Trainer.valid()
  	n_search_trials = 10, # number of trials to find the best hyperparams
  	report_last_n=5, 
  	do_fixed_trial=True,
  	n_fixed_trials=5,
  	trial_name=None,
  	storage='sqlite:///optuna.db',
  	n_jobs=4, # number of parallel jobs
  	n_gpus=2, # number of gpus
  	client=None,
)

trial_param_list = [
    dict(lr = 1e-2,hidden_size = 64,),
    dict(lr = 1e-3,hidden_size = 128,),
]

paramtuner.run(
    trial_param_list=trial_param_list,
)