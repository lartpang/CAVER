import os
import random

import numpy as np
import torch
from torch import nn


def worker_init_fn(worker_id, base_seed):
    set_seed_for_lib(base_seed + worker_id)


def set_seed_for_lib(seed):
    random.seed(seed)
    np.random.seed(seed)
    # 为了禁止hash随机化，使得实验可复现。
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def initialize_seed_cudnn(seed, deterministic):
    assert isinstance(deterministic, bool) and isinstance(seed, int)
    if seed >= 0:
        print(f"We will use the fixed seed: {seed} !!!")
        set_seed_for_lib(seed)
    else:
        print(f"We will not use the fixed seed !!!")
    if not deterministic:
        print("We will use `torch.backends.cudnn.benchmark`")
    else:
        print("We will not use `torch.backends.cudnn.benchmark`")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def is_on_gpu(x):
    """
    判定x是否是gpu上的实例，可以检测tensor和module
    :param x: (torch.Tensor, nn.Module)目标对象
    :return: 是否在gpu上
    """
    # https://blog.csdn.net/WYXHAHAHA123/article/details/86596981
    if isinstance(x, torch.Tensor):
        return "cuda" in x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).is_cuda
    else:
        raise NotImplementedError


def get_device(x):
    """
    返回x的设备信息，可以处理tensor和module
    :param x: (torch.Tensor, nn.Module) 目标对象
    :return: 所在设备
    """
    # https://blog.csdn.net/WYXHAHAHA123/article/details/86596981
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).device
    else:
        raise NotImplementedError
