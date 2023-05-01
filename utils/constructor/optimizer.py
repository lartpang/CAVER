# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from torch import nn
from torch.optim import Adam, AdamW, SGD


def make_optim_with_cfg(model: nn.Module, optimizer_cfg: dict):
    lr = optimizer_cfg["lr"]
    optimizer_strategy = optimizer_cfg["strategy"]
    optimizer_type = optimizer_cfg["optimizer"]
    chosen_optimizer_cfg = optimizer_cfg["optimizer_candidates"][optimizer_type]
    grouped_params = group_params(model=model, lr=lr, optimizer_strategy=optimizer_strategy, cfg=chosen_optimizer_cfg)
    optimizer = construct_optimizer(
        params=grouped_params, lr=lr, optimizer_type=optimizer_type, cfg=chosen_optimizer_cfg
    )
    return optimizer


def construct_optimizer(cfg, params, lr, optimizer_type):
    if optimizer_type == "sgd":
        optimizer = SGD(params=params, lr=lr, **cfg)
    elif optimizer_type == "adamw":
        optimizer = AdamW(params=params, lr=lr, **cfg)
    elif optimizer_type == "adam":
        optimizer = Adam(params=params, lr=lr, **cfg)
    else:
        raise NotImplementedError
    return optimizer


def group_params(cfg, lr, model, optimizer_strategy):
    if optimizer_strategy == "yolov5":
        """
        norm, weight, bias = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                bias.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                norm.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                weight.append(v.weight)  # apply decay

        if opt.adam:
            optimizer = optim.Adam(norm, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(norm, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

        optimizer.add_param_group({"params": weight, "weight_decay": hyp["weight_decay"]})  # add weight with weight_decay
        optimizer.add_param_group({"params": bias})  # add bias (biases)
        """
        norm, weight, bias = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                bias.append(v.bias)  # conv bias and bn bias
            if isinstance(v, nn.BatchNorm2d):
                norm.append(v.weight)  # bn weight
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                weight.append(v.weight)  # conv weight
        params = [
            {"params": bias, "weight_decay": 0.0},
            {"params": norm, "weight_decay": 0.0},
            {"params": weight},
        ]
    elif optimizer_strategy == "r3":
        params = [
            # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
            # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
            # 到减少模型过拟合的效果。
            {
                "params": [param for name, param in model.named_parameters() if name[-4:] == "bias"],
                "lr": 2 * lr,
                # NOTE: 覆盖默认设置，之前忘了设置
                "weight_decay": 0,
            },
            {
                "params": [param for name, param in model.named_parameters() if name[-4:] != "bias"],
            },
        ]
    elif optimizer_strategy == "all":
        params = model.parameters()
    elif optimizer_strategy == "finetune":
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "get_grouped_params"):
            params_groups = model.get_grouped_params()
            params = [
                {"params": params_groups["pretrained"], "lr": 0.1 * lr},
                {"params": params_groups["retrained"], "lr": lr},
            ]
            if params_groups.get("no_training"):
                params.append({"params": params_groups["no_training"], "lr": 0, "weight_decay": 0})

        else:
            params = [{"params": model.parameters(), "lr": 0.1 * lr}]
    else:
        raise NotImplementedError
    return params
