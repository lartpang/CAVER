# -*- coding: utf-8 -*-
# @Time    : 2020/11/7
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

_base_ = ["base.py"]

args = dict(
    base_seed=112358,
    batch_size=8,
    print_freq=20,
    epoch_num=100,
    use_amp=True,
    iter_num=21840,
    epoch_based=True,
)

optimizers = dict(
    lr=0.005,
    strategy="all",
    optimizer="sgd",
    optimizer_candidates=dict(
        sgd=dict(
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=False,
        ),
    ),
)

schedulers = dict(
    sche_usebatch=True,
    strategy="cos",
    scheduler_candidates=dict(
        cos=dict(
            warmup_length=1,
            min_coef=0.001,
            max_coef=1,
        ),
    ),
)

data = dict(
    train=dict(
        name=[
            "VT5000TR",
        ],
        shape=dict(h=256, w=256),
    ),
    test=dict(
        name=[
            "VT821",
            "VT1000",
            "VT5000TE",
        ],
        shape=dict(h=256, w=256),
    ),
)
