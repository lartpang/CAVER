# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import math
from bisect import bisect_right
from functools import partial

import numpy as np
from torch.optim import lr_scheduler


def _get_anneal_func(anneal_strategy):
    _coef_type_mapping = dict(
        cos=lambda x: (1 - math.cos(math.pi * x)) / 2.0,
        linear=lambda x: x,
    )
    _anneal_coef_func = _coef_type_mapping[anneal_strategy]

    def _anneal_coef(start, end, pct):
        """start + [0, 1] * (end - start)"""
        assert 0 <= pct <= 1, f"pct must be in [0, 1]"
        return start + (end - start) * _anneal_coef_func(pct)

    return _anneal_coef


def get_one_cycle_coef_func(total_steps, max_coef, pct_start, anneal_strategy, div_factor, final_div_factor):
    """
    使用学习率放缩系数的最值来确定整体变化曲线

    Args:
        total_steps: 总体迭代数目
        max_coef: 最大的学习率系数
        pct_start: 学习率变化趋势由升转降点处占整体迭代的百分比
        anneal_strategy: 变化方式，支持cos和linear
        div_factor: 初始学习率(initial_coef)对应的系数等于 max_coef / div_factor
        final_div_factor: 最终学习率对应的系数等于 initial_coef / final_div_factor

    Returns:
        参数仅为curr_idx的一个函数，其返回为对应的系数
    """
    assert 0 <= pct_start <= 1.0, f"{pct_start} must be in [0, 1]"
    step_size_up = float(pct_start * total_steps) - 1
    step_size_down = float(total_steps - step_size_up) - 1
    initial_coef = max_coef / div_factor
    min_lr = initial_coef / final_div_factor
    anneal_func = _get_anneal_func(anneal_strategy=anneal_strategy)

    def _get_one_cycle_coef(curr_idx):
        if curr_idx <= step_size_up:
            up_step_pct = curr_idx / step_size_up
            coefficient = anneal_func(start=initial_coef, end=max_coef, pct=up_step_pct)
        else:
            down_step_pct = (curr_idx - step_size_up) / step_size_down
            coefficient = anneal_func(start=max_coef, end=min_lr, pct=down_step_pct)
        return coefficient

    return _get_one_cycle_coef


def get_linear_one_cycle_coef_func(total_num):
    """
    使用绝对值函数公式直接定义变化曲线，该函数仅支持线性变化，仅提供了F3Net的策略

    Args:
        total_num: 总体迭代次数

    Returns:
        返回参数仅为curr_idx的函数，其返回值为对应的系数
    """

    def _get_one_cycle_coef(curr_idx):
        """F3Net的方式"""
        return 1 - np.abs((curr_idx + 1) / (total_num + 1) * 2 - 1)

    return _get_one_cycle_coef


def get_multi_step_coef_func(gamma, milestones):
    """
    lr = baselr * gamma ** 0    if curr_idx < milestones[0]
    lr = baselr * gamma ** 1   if milestones[0] <= epoch < milestones[1]
    ...
    """
    milestones = list(sorted(milestones))

    def _get_multi_step_coef(curr_idx):
        return gamma ** bisect_right(milestones, curr_idx)

    return _get_multi_step_coef


def get_cos_coef_func(total_num, min_coef, max_coef, turning_step_point):
    # \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
    # \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
    # c_t = c_{min} + 1/2 * (c_max - c_min) * (1 + cos(i_cur / i_max * pi))
    num_step_up = turning_step_point - 1
    num_step_down = total_num - num_step_up

    def _up_anneal_func(x):
        return 1 / turning_step_point * (1 + x)

    def _down_anneal_func(x):
        return min_coef + (max_coef - min_coef) * (1 + np.cos(np.pi * x / num_step_down)) / 2

    def _cos_coef_func(curr_idx):
        if curr_idx <= num_step_up:
            # 0,1,2,...,turning_epoch-1
            coefficient = _up_anneal_func(curr_idx)
        else:
            # turning_epoch,...,end_epoch
            curr_idx = curr_idx - num_step_up
            coefficient = _down_anneal_func(curr_idx)

        return coefficient

    return _cos_coef_func


def get_poly_coef_func(total_num, turning_step_point, lr_decay, min_coef=None):
    num_step_up = turning_step_point - 1
    num_step_down = total_num - num_step_up

    def _up_anneal_func(x):
        return 1 / turning_step_point * (1 + x)

    def _down_anneal_func(x):
        return pow((1 - x / num_step_down), lr_decay)

    def _get_poly_coef(curr_idx):
        # 在turning_epoch-1时达到最大系数
        if curr_idx <= num_step_up:
            # 0 -> turning_epoch-1
            coefficient = _up_anneal_func(curr_idx)
        else:
            # turning_epoch -> end_epoch
            curr_idx = curr_idx - num_step_up  # 1 -> num_step_down
            coefficient = _down_anneal_func(curr_idx)

        if min_coef is not None:
            coefficient = max(min_coef, coefficient)

        return coefficient

    return _get_poly_coef


def get_lr_coefficient(curr_epoch, total_num, lr_strategy, scheduler_cfg):
    # because the parameter `total_num` is involved in assignment,
    # so, if we don't want to pass this varible through the function
    # get_lr_coefficient's parameters, we need to use the nonlocal keyword.

    # ** curr_epoch start from 0 **
    if lr_strategy == "poly":
        turning_epoch = scheduler_cfg["warmup_length"]
        if curr_epoch < turning_epoch:
            # 0,1,2,...,turning_epoch-1
            coefficient = 1 / turning_epoch * (1 + curr_epoch)
        else:
            # turning_epoch,...,end_epoch
            curr_epoch -= turning_epoch - 1
            total_num -= turning_epoch - 1
            coefficient = np.power((1 - float(curr_epoch) / total_num), scheduler_cfg["lr_decay"])
        if min_coef := scheduler_cfg.get("min_coef"):
            coefficient = max(min_coef, coefficient)
    elif lr_strategy == "cos":
        # \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        # \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        # c_t = c_{min} + 1/2 * (c_max - c_min) * (1 + cos(i_cur / i_max * pi))
        turning_epoch = scheduler_cfg["warmup_length"]
        if curr_epoch < turning_epoch:
            # 0,1,2,...,turning_epoch-1
            coefficient = 1 / turning_epoch * (1 + curr_epoch)
        else:
            # turning_epoch,...,end_epoch
            curr_epoch -= turning_epoch - 1
            total_num -= turning_epoch - 1
            min_coef = scheduler_cfg["min_coef"]
            max_coef = scheduler_cfg["max_coef"]
            coefficient = min_coef + (max_coef - min_coef) * (1 + np.cos(np.pi * curr_epoch / total_num)) / 2
    elif lr_strategy == "linearonclr":
        coef_func = get_linear_one_cycle_coef_func(total_num=total_num)
        coefficient = coef_func(curr_idx=curr_epoch)
    elif lr_strategy == "constant":
        coefficient = 1
    elif lr_strategy == "one_cycle":
        coef_func = get_one_cycle_coef_func(
            total_steps=total_num,
            max_coef=scheduler_cfg["max_coef"],
            pct_start=scheduler_cfg["pct_start"],
            anneal_strategy=scheduler_cfg["anneal_strategy"],
            div_factor=scheduler_cfg["div_factor"],
            final_div_factor=scheduler_cfg["final_div_factor"],
        )
        coefficient = coef_func(curr_idx=curr_epoch)
    elif lr_strategy == "multi_step":
        coef_func = get_multi_step_coef_func(gamma=scheduler_cfg["gamma"], milestones=scheduler_cfg["milestones"])
        coefficient = coef_func(curr_idx=curr_epoch)
    else:
        raise Exception(f"{lr_strategy} is not implemented")

    return coefficient


def get_lr_coefficient_v1(curr_epoch, total_num, lr_strategy, scheduler_cfg):
    """
    根据给定的参数来选择使用特定的学习率调整策略
    当前支持：
        - one_cycle
        - linear_one_cycle
        - multi_step
        - cos
        - poly
        - constant
    """
    if lr_strategy == "one_cycle":
        coef_func = get_one_cycle_coef_func(
            total_steps=total_num,
            max_coef=scheduler_cfg["max_coef"],
            pct_start=scheduler_cfg["pct_start"],
            anneal_strategy=scheduler_cfg["anneal_strategy"],
            div_factor=scheduler_cfg["div_factor"],
            final_div_factor=scheduler_cfg["final_div_factor"],
        )
    elif lr_strategy == "linear_one_cycle":
        coef_func = get_linear_one_cycle_coef_func(total_num=total_num)
    elif lr_strategy == "multi_step":
        coef_func = get_multi_step_coef_func(gamma=scheduler_cfg["gamma"], milestones=scheduler_cfg["milestones"])
    elif lr_strategy == "cos":
        coef_func = get_cos_coef_func(
            total_num=total_num,
            min_coef=scheduler_cfg["min_coef"],
            max_coef=scheduler_cfg["max_coef"],
            turning_step_point=scheduler_cfg["warmup_length"],
        )
    elif lr_strategy == "poly":
        coef_func = get_poly_coef_func(
            total_num=total_num,
            turning_step_point=scheduler_cfg["warmup_length"],
            lr_decay=scheduler_cfg["lr_decay"],
            min_coef=scheduler_cfg.get("min_coef", None),
        )
    elif lr_strategy == "constant":
        coef_func = lambda x: scheduler_cfg["constant_coef"]
    else:
        raise KeyError(f"{lr_strategy} is not implemented. Has been supported: "
                       f"one_cycle, linear_one_cycle, multi_step, cos, poly, constant"
                       )

    # because the parameter `total_num` is involved in assignment,
    # so, if we don't want to pass this varible through the function
    # get_lr_coefficient's parameters, we need to use the nonlocal keyword.

    # ** curr_epoch start from 0 **
    coefficient = coef_func(curr_idx=curr_epoch)
    return coefficient


def make_scheduler_with_cfg(optimizer, total_num, scheduler_cfg: dict):
    lr_strategy = scheduler_cfg["strategy"]
    chosen_scheduler_cfg = scheduler_cfg["scheduler_candidates"][lr_strategy]
    if lr_strategy == "clr":
        scheduler = lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=chosen_scheduler_cfg["min_lr"],
            max_lr=chosen_scheduler_cfg["max_lr"],
            step_size_up=chosen_scheduler_cfg["step_size"],
            scale_mode=chosen_scheduler_cfg["mode"],
        )
    elif lr_strategy == "step":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=chosen_scheduler_cfg["milestones"],
            gamma=chosen_scheduler_cfg["gamma"],
        )
    else:
        lr_func = partial(
            get_lr_coefficient,
            total_num=total_num,
            lr_strategy=lr_strategy,
            scheduler_cfg=chosen_scheduler_cfg,
        )
        scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)
    return scheduler


# simpler and more direct.
def make_scheduler_with_cfg_v2(total_num, optimizer, scheduler_cfg: dict):
    lr_strategy = scheduler_cfg["strategy"]
    chosen_scheduler_cfg = scheduler_cfg["scheduler_candidates"][lr_strategy]
    initial_lr_groups = [group["lr"] for group in optimizer.param_groups]

    def _get_adjusted_lr(curr_epoch):
        coefficient = get_lr_coefficient(
            curr_epoch=curr_epoch,
            total_num=total_num,
            lr_strategy=lr_strategy,
            scheduler_cfg=chosen_scheduler_cfg,
        )

        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = initial_lr_groups[i] * coefficient

    return _get_adjusted_lr


# 添加对于基于batch迭代和基于epoch迭代的识别和对应的处理
class LRAdjustor(object):
    """
    ```
    lr_adjustor = LRAdjustor(
        optimizer=optimizer,
        total_num=num_iter if config.step_by_batch else config.args.epoch_num,
        scheduler_cfg=config.schedulers,
    )
    ...some code...
        model.train()
        for curr_iter_in_epoch, data in enumerate(tr_loader):
            curr_iter = curr_epoch * num_iter_per_epoch + curr_iter_in_epoch
            lr_adjustor(curr_iter)
            ...main code for training...
    ```
    """

    def __init__(self, total_num, initial_lr_groups, num_iters_per_epoch, scheduler_cfg):
        self.total_num = total_num
        self.lr_strategy = scheduler_cfg["strategy"]
        self.scheduler_cfg = scheduler_cfg["scheduler_candidates"][self.lr_strategy]
        self.step_by_batch = scheduler_cfg["sche_usebatch"]
        self.num_iters_per_epoch = num_iters_per_epoch
        self.initial_lr_groups = initial_lr_groups
        self.prev_coefficient = 1

    def __call__(self, optimizer, curr_idx):
        if not self.step_by_batch:
            curr_idx = curr_idx // self.num_iters_per_epoch

        # by epoch: curr_idx, 应该[0, num_iters_per_epoch-1] 满足相同的学习率
        # by batch: curr_idx, 应该在curr_idx == warmup_length-1时达到最大，并且curr_idx=0时为小的学习率
        coefficient = get_lr_coefficient_v1(
            curr_epoch=curr_idx,
            total_num=self.total_num,
            lr_strategy=self.lr_strategy,
            scheduler_cfg=self.scheduler_cfg,
        )

        if curr_idx >= self.total_num:
            coefficient = self.prev_coefficient

        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = self.initial_lr_groups[i] * coefficient
        self.prev_coefficient = coefficient
