args = dict(
    use_amp=True,
    base_seed=0,
    deterministic=True,
    epoch_num=40,  # 训练周期, 0: directly test model
    batch_size=8,
    num_workers=4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    print_freq=100,  # >0, 保存迭代过程中的信息
)

optimizers = dict(
    lr=0.005,
    strategy="all",
    optimizer="sgd",
    optimizer_candidates=dict(
        sgd=dict(momentum=0.9, weight_decay=5e-4, nesterov=False),
        adamw=dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=False),
        adam=dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4, amsgrad=False),
    ),
)

schedulers = dict(
    sche_usebatch=True,
    strategy="poly",
    scheduler_candidates=dict(
        linear_one_cycle=dict(),
        one_cycle=dict(max_coef=50, pct_start=0.3, anneal_strategy="cos", div_factor=5e2, final_div_factor=1e2),
        cos=dict(warmup_length=2000, min_coef=0.001, max_coef=1),
        poly=dict(warmup_length=1, lr_decay=0.9, min_coef=None),
        multi_step=dict(milestones=[30, 45, 55], gamma=0.1),
        constant=dict(constant_coef=1),
    ),
)
