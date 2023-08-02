import torch
import numpy as np
from utils.base_util import plot_dict

class MetricsRecorder:
    """
    Metrics recorder. 

    Two kinds of recording methods:
    1) autometicly average the metrics in metrics_dict and save in summary_metrics_dict;
    2) manually record and save in manual_summary_metrics_dict;

    See required params in register() function.
    """
    def __init__(self,):
        # batch dict
        self.metrics_dict = {}
        # epoch dict
        self.summary_metrics_dict = {}
        # metrics dict for manually record (summary)
        self.manual_summary_metrics_dict = {}
        # metrics dict for manually record (batch)
        self.manual_metrics_dict = {}
    
    # Registering metrics is required before recording 
    def register(self,names=None,manual_summary_metrics=None,manual_metrics=None):
        """
        register lists are:
        - names: list of metrics names. automatically average and save in summary_metrics_dict.
        - manual_summary_metrics: list of manual summary metrics names. manually record and save in manual_summary_metrics_dict.
        - manual_metrics: list of manual metrics names. manually record each step and save in manual_metrics_dict. could be used for later analysis and later saving in manual_summary_metrics_dict.
        """
        if names is not None:
            assert isinstance(names,list)
            for each_name in names:
                self.metrics_dict[each_name] = []
                self.summary_metrics_dict[each_name] = []
        if manual_summary_metrics is not None:
            for each_manual in manual_summary_metrics:
                self.manual_summary_metrics_dict[each_manual] = []
        if manual_metrics is not None:
            for each_manual in manual_metrics:
                self.manual_metrics_dict[each_manual] = []

    @torch.no_grad()
    def log(self,k,v):
        """
        log to metrics_dict
        log metrics values of a single batch
        """
        self.metrics_dict[k]+=[v]
    
    @torch.no_grad()
    def log_manual(self,k,v):
        """
        log to manual_metrics_dict
        log manual metrics values of a single batch
        """
        self.manual_metrics_dict[k]+=[v]
    
    @torch.no_grad()
    def log_manual_summary_metric(self,k,v):
        """
        log to manual_summary_metrics_dict
        """
        self.manual_summary_metrics_dict[k]+=[v]
    
    # clear batch dict
    def _clear(self,*names,do_clear_all=True):
        if do_clear_all:
            for each_name in self.metrics_dict.keys():
                self.metrics_dict[each_name] = []
        for each_name in names:
            self.metrics_dict[each_name] = []
    
    # clear manual batch dict
    def clear_manual(self,*names,do_clear_all=False):
        if do_clear_all:
            for each_name in self.manual_metrics_dict.keys():
                self.manual_metrics_dict[each_name] = []
        elif set(names).issubset(self.manual_metrics_dict):
            for each_name in names:
                self.manual_metrics_dict[each_name] = []
        else:
            raise Exception('Specified metric(s) is not in the registered manual_metrics_dict! cannot be cleared! ')

    
    @torch.no_grad()
    def calculate_metrics(self):
        # update epoch dict based on batch dict
        for each_name in self.metrics_dict.keys():
            self.summary_metrics_dict[each_name].append(np.mean(self.metrics_dict[each_name]))
        # clear batch dict
        self._clear(do_clear_all=True)
    
    def plot_summary_metrics(
        self,
        path = None,
        caption='Plots of summary metrics.',
        subplot_size=3,
        caption_size=10,
        ):
        plot_dict(
            **self.summary_metrics_dict,
            **self.manual_summary_metrics_dict,
            path=path,
            caption=caption,
            subplot_size=subplot_size,
            caption_size=caption_size,
        )


