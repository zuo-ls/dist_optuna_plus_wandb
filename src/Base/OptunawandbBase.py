import torch
import wandb
import optuna
import multiprocessing
from optuna.storages import RDBStorage
import optuna_distributed
from utils.base_util import set_seed,DataPackDict
from Hooks.LogHook import LogHook
from Hooks.OptunawandbHook import OptunawandbHook

class OptunawandbBase:
    """
    Incorporate optuna and wandb into one class. Should work with class OptunawandbHook.
    OptunawandbHook is a hook for Trainer, which will log the metric to wandb and optuna.
    
    OptunawandbBase will:
    - create a study for optuna
    - init wandb run for each trial
    - implement objective function for optuna
    - log best trial to wandb 
    - automatically init a new wandb run to log the summary and visualization

    Usage:
    1. create a class that inherits from OptunawandbBase
    2. implement load_items, get_suggested_params, load_update_cfg
    3. call run() to start searching for best params

    Args:
        proj_name: name of the project in wandb
        metric: metric to optimize
        report_last_n: report mean of last n trials to optuna for pruning
        exp_purpose: purpose of the experiment, will be logged to wandb as notes
        do_init_wandb: whether to init wandb run in objective function, set to False to disable wandb
        do_fixed_trial: set to True to run fixed number of trials (n_fixed_trials) with fixed params, 
                        the fixed params should be specified in trial_param_list.
                        pass trial_param_list to run() to specify the params.
    """

    def __init__(self,proj_name,metric,report_last_n=5,n_search_trials=1,exp_purpose=None,do_init_wandb=True,do_fixed_trial=False,trial_name=None,n_fixed_trials=1,n_gpus=-1,storage=None,client=None,n_jobs=-1,allow_prune=True):
        self.study = None
        self.metric = metric
        self.proj_name = proj_name
        self.exp_purpose = exp_purpose
        self.do_init_wandb = do_init_wandb
        self.report_last_n = report_last_n
        self.do_fixed_trial = do_fixed_trial
        self.n_fixed_trials = n_fixed_trials
        self.trial_name = trial_name
        self.client = client
        self.n_search_trials = n_search_trials
        self.allow_prune = allow_prune
        
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif n_jobs == None:
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs
        
        if storage is None:
            self.storage = storage
        else:
            self.storage = RDBStorage(storage)
        
        if n_gpus == -1:
            self.n_gpus = torch.cuda.device_count()
        else:
            self.n_gpus = n_gpus

        if self.do_fixed_trial and self.n_fixed_trials>1:
            self.allow_prune = False
            print('allow_prune is set to False to run n fixed trials.')
        
        if exp_purpose is None:
            self.exp_purpose = f'{proj_name}: purpose not specified.'

    # register fixed trails (list of dict) to enqueue_trial
    def register_trials(self, trial_param_list:list):
        for trial_param in trial_param_list:
            self.study.enqueue_trial(trial_param)

    # dask_cuda based
    def run(self,trial_param_list=None): 
        try:
            print('try to create study...')
            study = optuna.create_study(
                study_name=self.proj_name,
                direction='maximize',
                storage= self.storage,            
                )
        except:
            print('study exists, delete it and create a new one...')
            optuna.study.delete_study(study_name=self.proj_name,storage=self.storage)
            study = optuna.create_study(
                study_name=self.proj_name,
                direction='maximize',
                storage= self.storage,            
                )
        print('study created.')
        
        # dist_optuna
        # wrap study with optuna_distributed
        self.study = optuna_distributed.from_study(study=study,client=self.client)
        
        # register fixed trials
        if self.do_fixed_trial and trial_param_list is None:
            raise ValueError('trial_param_list cannot be None if do_fixed_trial is True. should specify params for this single trial.')
        if trial_param_list is not None and self.do_fixed_trial:
            print('registering fixed trials...')
            trial_param_list_to_regist = trial_param_list*self.n_fixed_trials
            self.register_trials(trial_param_list_to_regist) 

        # run optuna
        if self.do_fixed_trial:
            assert (trial_param_list is not None) and (len(trial_param_list) != 0)
            self.study.optimize(self.objective, n_trials=self.n_fixed_trials*len(trial_param_list),n_jobs=self.n_jobs)
        else:
            self.study.optimize(self.objective,n_jobs=self.n_jobs,n_trials=self.n_search_trials)

        if not self.do_fixed_trial:
            self.log_summary()
    
    def log_summary(self):
        """init a new wandb run to log the summary"""
        wandb.init(project=self.proj_name, name='summary',)
        self.log_best_wandb_summary()
        self.log_optuna_visualization_to_wandb(study_obj=self.get_study_obj())
        wandb.finish()

    def get_study_obj(self):
        """get study object from optuna_distributed or optuna"""
        if 'distributed' in str(type(self.study)):
            study_obj = self.study._study
        else:
            study_obj = self.study
        return study_obj
        
    def check_previous_trials(self,trial):
        """check if previous trials have same params, if so, return True, else False, and return the trial number of previous trial"""
        for each_previous_trial in self.study.trials[:trial.number]:
            # if previous_trial.state == TrialState.COMPLETE and trial.params == previous_trial.params:
            if trial.params == each_previous_trial.params:   
                return DataPackDict(
                    exist = True,
                    trial_number = each_previous_trial.number,
                )
        return DataPackDict(
            exist = False,
            trial_number = None,
        )

    def set_trial_seed(self,trial_number,no_change_seed, base_seed=42):
        if no_change_seed:
            set_seed(base_seed)
            wandb.summary['seed'] = base_seed
        else:
            set_seed(trial_number + base_seed)
            wandb.summary['seed'] = trial_number+base_seed


    def objective(self,trial):
        # get suggested params and load to cfg
        to_tune_params = self.get_suggested_params(trial)
        other_cfg = self.get_other_cfg()

        # change device id
        if not (self.do_fixed_trial and self.n_fixed_trials==1):
            current_gpu_id = int(trial.number) % int(self.n_gpus)
            device = f'cuda:{current_gpu_id}'

        # log params to wandb and set seed
        self.init_wandb(trial,to_tune_params)
        params_exist_flag = self.check_previous_trials(trial).exist

        self.set_trial_seed(
            trial_number=trial.number,
            no_change_seed= not params_exist_flag,
        )
        
        trainer = self.load_items(device=device,**to_tune_params,**other_cfg)

        hooks = [
            LogHook(trail_name=f'trial_{trial.number}'),
            OptunawandbHook(trial, do_watch_model=True, metric=self.metric, report_last_n = self.report_last_n, allow_prune=self.allow_prune),
        ]
        trainer.register_hooks(hooks)
        trainer.run()

        # return report value
        optunawand_obj = [each for each in trainer.hooks if 'OptunawandbHook' in str(type(each))][0]
        report_value = optunawand_obj.report_value
        wandb.finish()
        return report_value

    def init_wandb(self,trial,to_tune_params):    
        wandb_init_params = DataPackDict(
                project = self.proj_name,
                name = f'trial_{trial.number}',
                config = to_tune_params,
                reinit = True,
                notes = self.exp_purpose,
                settings=wandb.Settings(start_method='fork'),
        )
        if not self.do_init_wandb:
            wandb_init_params.update(mode="disabled")
        
        if self.do_fixed_trial and self.trial_name is not None:
            wandb_init_params.update(name=self.trial_name)
        
        wandb.init(**wandb_init_params)
    
    def log_optuna_visualization_to_wandb(self,study_obj):
        plot_optim_hist = optuna.visualization.plot_optimization_history(study_obj)
        plot_optim_hist = plot_optim_hist.to_html()
        wandb.log({'optimization_history': wandb.Html(plot_optim_hist, inject=False)})

        param_importance = optuna.visualization.plot_param_importances(study_obj)
        param_importance = param_importance.to_html()
        wandb.log({'param_importance': wandb.Html(param_importance, inject=False)})

        intermediate_value = optuna.visualization.plot_intermediate_values(study_obj)
        intermediate_value = intermediate_value.to_html()
        wandb.log({'intermediate_value': wandb.Html(intermediate_value, inject=False)})

        plot_parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study_obj)
        plot_parallel_coordinate = plot_parallel_coordinate.to_html()
        wandb.log({'plot_parallel_coordinate': wandb.Html(plot_parallel_coordinate, inject=False)})

        plot_slice = optuna.visualization.plot_slice(study_obj)
        plot_slice = plot_slice.to_html()
        wandb.log({'plot_slice': wandb.Html(plot_slice, inject=False)})

        plot_contour = optuna.visualization.plot_contour(study_obj)
        plot_contour = plot_contour.to_html()
        wandb.log({'plot_contour': wandb.Html(plot_contour, inject=False)})

        plot_edf = optuna.visualization.plot_edf(study_obj)
        plot_edf = plot_edf.to_html()
        wandb.log({'plot_edf': wandb.Html(plot_edf, inject=False)})

    
    def log_best_wandb_summary(self):
        for param_name, param_value in self.study.best_trial.params.items():
            wandb.run.summary['best_'+param_name] = param_value
        wandb.run.summary["best_"+self.metric] = self.study.best_trial.value 
    
    def load_items(self,**kwargs):
        raise NotImplementedError('should return trainer, the trainer will be trained by calling trainer.run().')
    
    def get_suggested_params(self, trial):
        """returns a dict of suggested params"""
        raise NotImplementedError
    
    def get_other_cfg(self):
        """returns a dict of other cfg"""
        return {}



