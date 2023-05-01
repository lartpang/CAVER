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
            "NLPR_TR",
            "NJUD_TR",
            "DUTRGBD_TR",
        ],
        shape=dict(h=256, w=256),
    ),
    test=dict(
        name=[
            "NJUD_TE",
            "NLPR_TE",
            "LFSD",
            "RGBD135",
            "SIP",
            "SSD",
            "STEREO1000",
            "DUTRGBD_TE",
            "REDWEBS_TE",
            "COME_TE_E",
            "COME_TE_H",
        ],
        shape=dict(h=256, w=256),
    ),
)
