import random
import numpy as np
import itertools as it
import torch
import logging
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Any, Tuple
import os


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.info(f"Set global seed to {seed}.")

def plot_dict(path=None,caption='Title',subplot_size=3,caption_size=10,**metrics_dict):
    n_plots = len(metrics_dict)
    f,axs = plt.subplots(1,n_plots)
    f.set_size_inches(subplot_size*n_plots,subplot_size)
    f.suptitle(caption, fontsize=caption_size)
    axs = (axs,) if n_plots==1 else axs
    for each_ax,each_k in zip(axs,metrics_dict):
        each_ax.plot(metrics_dict[each_k])
        each_ax.set_title(each_k)
    plt.tight_layout()
    if path is not None:
        f_name = caption[:-1] if caption.endswith('.') else caption
        plt.savefig(os.path.join(path,f_name+'.pdf'))
        plt.savefig(os.path.join(path,f_name+'.png'))
    plt_clear()

class DataPackDict(OrderedDict):
    # ref: https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/base/base_utils.py
    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]
    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)
    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())

def plt_clear():
    plt.clf()
    plt.cla()
    plt.close()
    