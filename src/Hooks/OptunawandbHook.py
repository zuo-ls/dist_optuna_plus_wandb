import wandb
from Base.BaseHook import BaseHook
import optuna

class OptunawandbHook(BaseHook):
    """
    Optuna wandb hook.

    This hook will:
    report value to wandb and prune if needed.
    log the model to wandb if do_watch_model is True.
    get metric values from dict saved in trainer.xxx_recorder, and log them to wandb.

    Args:
        trial (optuna.trial.Trial): The trial object.
        do_watch_model (bool): Whether to watch the model. Default: False.
        metric (str): The metric name to report to wandb. must in valid_recorder.manual_summary_metrics_dict, 
                      i.e. should be included in self.valid_out in the method Trainer.valid()
    """
    def __init__(self, trial, do_watch_model = False, metric = None, report_last_n = 1,allow_prune=True):
        self.trial = trial
        self.do_watch_model = do_watch_model
        self.report_last_n = report_last_n
        self.allow_prune = allow_prune
        if metric is None:
            raise ValueError("metric must be specified")
        self.metric = metric
        self.report_value = 0
    def before_train(self): 
        if self.do_watch_model:
            wandb.watch(self.trainer.model, log='all', log_freq=100)
    def after_train(self): pass
    def before_iter(self): pass
    def after_iter(self): 
        # prune if needed
        if self.trainer.current_iter % self.trainer.valid_freq == 0:
            self.log2wandb()
            self.report_value = self.get_report_value()
            self.trial.report(self.report_value, self.trainer.current_iter)
            if self.allow_prune and self.trial.should_prune():
                raise optuna.TrialPruned()
    def log2wandb(self):
        wandb.log(
            {"report_value":self.report_value}, step=self.trainer.current_iter
        )
        wandb.log(
            self.get_last_item(
                self.trainer.train_recorder.summary_metrics_dict
            ), step=self.trainer.current_iter
            )
        wandb.log(
            self.get_last_item(
                self.trainer.valid_recorder.manual_summary_metrics_dict
            ), step=self.trainer.current_iter
        )

    def get_last_item(self,record_dict):
        return {k:v[-1] for k,v in record_dict.items()}
    def get_history_mean(self,record_list,last_n = None):
        """
        get the mean of last n elements in a list, if the list has less than n elements, return the mean of all elements
        """
        if last_n is None:
            last_n = self.report_last_n
        return sum(record_list[-last_n:]) / last_n if len(record_list) >= last_n else sum(record_list) / len(record_list)
    def get_report_value(self,):
        report_value = self.trainer.valid_recorder.manual_summary_metrics_dict[self.metric]
        report_value = self.get_history_mean(report_value)
        return report_value