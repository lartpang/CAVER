import argparse
import csv
import inspect
import os
import shutil
from datetime import datetime
from functools import partial

import albumentations as A
import cv2
import numpy as np
import torch
from mmengine import Config
from tqdm import tqdm

import datasets
import method as model_lib
from method.mssim import ssim
from utils import constructor, ops, pt_utils, py_utils
from utils.data import get_data_from_txt, get_datasets_info_with_keys, read_binary_array, read_color_array
from utils.recorder import AvgMeter, CalTotalMetric, MsgLogger, TimeRecoder


def iou(prob, gt):
    inter = torch.sum(gt * prob, dim=(1, 2, 3))
    union = gt.sum(dim=(1, 2, 3)) + prob.sum(dim=(1, 2, 3)) - inter
    iou = inter / union
    return iou.mean()


class TrDataset(torch.utils.data.Dataset):
    def __init__(self, root, shape, extra_scales=None):
        super().__init__()
        if extra_scales is not None:
            self.scales = (1,) + tuple(extra_scales)

        self.total_paths = []
        for dataset_name, dataset_info in root.items():
            image_root = dataset_info["image"]["path"]
            image_suffix = dataset_info["image"]["suffix"]
            mask_root = dataset_info["mask"]["path"]
            mask_suffix = dataset_info["mask"]["suffix"]
            depth_root = dataset_info["depth"]["path"]
            depth_suffix = dataset_info["depth"]["suffix"]
            if "index_file" in dataset_info:
                valid_names = get_data_from_txt(dataset_info["index_file"])
            else:
                image_names = [x[: -len(image_suffix)] for x in os.listdir(image_root)]
                mask_names = [x[: -len(mask_suffix)] for x in os.listdir(mask_root)]
                depth_names = [x[: -len(depth_suffix)] for x in os.listdir(depth_root)]
                valid_names = list(set(image_names).intersection(mask_names).intersection(depth_names))

            for valid_name in sorted(valid_names):
                s = (
                    os.path.join(image_root, valid_name + image_suffix),
                    os.path.join(mask_root, valid_name + mask_suffix),
                    os.path.join(depth_root, valid_name + depth_suffix),
                )
                self.total_paths.append(s)
            print(f"Loading data from {dataset_name} with {len(valid_names)} samples.")

        self.joint_trans = A.Compose(
            [
                A.Resize(height=shape["h"], width=shape["w"]),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.75),
                A.Normalize(),
            ],
            additional_targets=dict(depth="mask"),  # For RGBD dataset
        )

    def __len__(self):
        return len(self.total_paths)

    def __getitem__(self, index):
        image_path, mask_path, depth_path = self.total_paths[index]

        image = read_color_array(image_path)
        mask = read_binary_array(mask_path, to_normalize=True, thr=0.5)
        depth = read_binary_array(depth_path, to_normalize=True, thr=-1)

        transformed = self.joint_trans(image=image, mask=mask, depth=depth)
        image = transformed["image"]
        mask = transformed["mask"]
        depth = transformed["depth"]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        return dict(image=image_tensor, mask=mask_tensor, depth=depth_tensor)


class TeDataset(torch.utils.data.Dataset):
    def __init__(self, root, shape):
        super().__init__()
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask", "depth"])
        self.image_paths = self.datasets["image"]
        self.mask_paths = self.datasets["mask"]
        self.depth_paths = self.datasets["depth"]

        self.joint_trans = A.Compose([A.Resize(height=shape["h"], width=shape["w"]), A.Normalize()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        depth_path = self.depth_paths[index]

        image = read_color_array(image_path)
        depth = read_binary_array(depth_path, to_normalize=True, thr=-1)

        transformed = self.joint_trans(image=image, mask=depth)
        image = transformed["image"]
        depth = transformed["mask"]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        return dict(
            image=image_tensor,
            depth=depth_tensor,
            image_info=dict(mask_path=mask_path, mask_name=os.path.basename(mask_path)),
        )


@torch.no_grad()
def eval_once(model, data_loader, save_path="", show_bar=True):
    model.eval()
    cal_total_seg_metrics = CalTotalMetric()

    bar_iter = enumerate(data_loader)
    if show_bar:
        bar_iter = tqdm(bar_iter, total=len(data_loader), leave=False, ncols=79)
    for batch_id, batch in bar_iter:
        images = batch["image"].cuda(non_blocking=True)
        depths = batch["depth"].cuda(non_blocking=True)
        logits = model(data=dict(image=images, depth=depths))
        probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()

        for i, pred in enumerate(probs):
            mask_path = batch["image_info"]["mask_path"][i]
            mask_array = read_binary_array(mask_path, dtype=np.uint8)
            mask_h, mask_w = mask_array.shape

            pred = cv2.resize(pred, dsize=(mask_w, mask_h), interpolation=cv2.INTER_LINEAR)  # 0~1

            if save_path:  # 这里的save_path包含了数据集名字
                pred_name = os.path.splitext(batch["image_info"]["mask_name"][i])[0] + ".png"
                ops.save_array_as_image(data_array=pred, save_name=pred_name, save_dir=save_path)

            pred = (pred * 255).astype(np.uint8)
            cal_total_seg_metrics.step(pred, mask_array, mask_path)
    return cal_total_seg_metrics.get_results()


def testing(model, msg_logger, cfg):
    msg_logger(name="log", msg="\n", show=False)

    csv_row = [cfg.exp_name]
    for te_data_name in cfg.data.test.name:
        te_data_path = datasets.__dict__[te_data_name]
        te_dataset = TeDataset(root=(te_data_name, te_data_path), shape=cfg.data.test.shape)
        te_loader = torch.utils.data.DataLoader(
            dataset=te_dataset,
            batch_size=cfg.args.batch_size,
            num_workers=cfg.args.num_workers,
            pin_memory=True,
        )
        print(f"Testing on {te_data_name} with {len(te_dataset)} samples")
        pred_save_path = os.path.join(cfg.path.save, te_data_name)
        seg_results = eval_once(model=model, save_path=pred_save_path, data_loader=te_loader, show_bar=cfg.show_bar)
        msg_logger(name="log", msg=f"Results on {te_data_path}:\n{seg_results}")

        csv_row.extend(list(seg_results.values()))

    # write the results into the csv file
    with open(cfg.path.csv, encoding="utf-8", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_row)


def loss_func(logits, seg_gts):
    losses = []
    loss_str = []
    # for main
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=seg_gts, reduction="mean")
    losses.append(bce_loss)
    loss_str.append(f"bce:{bce_loss.item():.5f}")

    prob = logits.sigmoid()
    ssim_loss = 1 - ssim(prob, seg_gts)
    losses.append(ssim_loss)
    loss_str.append(f"ssim:{ssim_loss.item():.5f}")

    ssim_loss = 1 - iou(prob, seg_gts)
    losses.append(ssim_loss)
    loss_str.append(f"iou:{ssim_loss.item():.5f}")
    return sum(losses), " ".join(loss_str)


def training(model, msg_logger, cfg):
    tr_data_paths = {n: datasets.__dict__[n] for n in cfg.data.train.name}
    tr_dataset = TrDataset(root=tr_data_paths, shape=cfg.data.train.shape)
    tr_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=cfg.args.batch_size,
        num_workers=cfg.args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=None
        if cfg.args.base_seed < 0
        else partial(pt_utils.worker_init_fn, base_seed=cfg.args.base_seed),
    )
    print(f"Training on {tuple(tr_data_paths.keys())} with {len(tr_dataset)} samples")

    num_iter_per_epoch = len(tr_loader)
    num_iter = cfg.args.epoch_num * num_iter_per_epoch

    optimizer = constructor.make_optim_with_cfg(model=model, optimizer_cfg=cfg.optimizers)
    print(f"optimizer:\n{optimizer}")
    lr_adjustor = constructor.LRAdjustor(
        initial_lr_groups=[group["lr"] for group in optimizer.param_groups],
        total_num=num_iter if cfg.schedulers.sche_usebatch else cfg.args.epoch_num,
        num_iters_per_epoch=num_iter_per_epoch,
        scheduler_cfg=cfg.schedulers,
    )
    num_iter_in_cooldown = cfg.cooldown_epoch_num * num_iter_per_epoch
    py_utils.plot_lr_curve_for_scheduler(
        optimizer=optimizer,
        scheduler=lr_adjustor,
        num_steps=num_iter + num_iter_in_cooldown,
        save_path=os.path.join(cfg.path.exp, "lr.png"),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.args.use_amp)

    loss_recorder = AvgMeter()
    time_logger = TimeRecoder()
    for epoch_idx in range(cfg.args.epoch_num + cfg.cooldown_epoch_num):
        time_logger.start(msg=cfg.exp_name)
        loss_recorder.reset()
        model.train()

        for batch_idx, batch in enumerate(tr_loader):
            curr_iter = epoch_idx * num_iter_per_epoch + batch_idx
            lr_adjustor(optimizer=optimizer, curr_idx=curr_iter)

            images = batch["image"].cuda(non_blocking=True)
            depths = batch["depth"].cuda(non_blocking=True)
            masks = batch["mask"].cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=cfg.args.use_amp):
                logits = model(data=dict(image=images, depth=depths))

            losses, losses_str = loss_func(logits=logits, seg_gts=masks)
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            item_loss = losses.item()
            loss_recorder.update(value=item_loss, num=images.size(0))

            fixed_step = batch_idx == 0 or (batch_idx + 1) == num_iter_per_epoch
            interval_step = cfg.args.print_freq > 0 and (
                curr_iter % cfg.args.print_freq == 0 or curr_iter == num_iter - 1
            )
            if fixed_step or interval_step:
                lr_string = ",".join([f"{x:10.3e}" for x in [group["lr"] for group in optimizer.param_groups]])
                msg = (
                    f"[{batch_idx}/{num_iter_per_epoch} {curr_iter}/{num_iter + num_iter_in_cooldown} {epoch_idx}/{cfg.args.epoch_num + cfg.cooldown_epoch_num}] "
                    f"{list(images.shape)} Lr:{lr_string} M:{loss_recorder.avg:.5f}/C:{item_loss:.5f} "
                    f"{losses_str}"
                )
                msg_logger(name="log", msg=msg, show=True)

            if curr_iter < 3:
                py_utils.cvplot_results(
                    dict(smap=logits.sigmoid(), img=batch["image"], dep=batch["depth"], msk=batch["mask"]),
                    save_path=os.path.join(cfg.vis_path, f"iter-{curr_iter}.png"),
                )

        # 记录每个epoch最后一个batch的图像
        py_utils.cvplot_results(
            dict(smap=logits.sigmoid(), img=batch["image"], dep=batch["depth"], msk=batch["mask"]),
            save_path=os.path.join(cfg.vis_path, f"epoch-{epoch_idx}.png"),
        )
        torch.save(model.state_dict(), cfg.path.state)
        time_logger.now(pre_msg="An Epoch End...")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--pretrained", type=str, help="Pretrained params of the backbone of your model.")
    parser.add_argument("--info", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--show-bar", action="store_true")
    parser.add_argument("--cooldown-epoch-num", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config, use_predefined_variables=False)
    cfg.merge_from_dict(vars(args))

    cfg.exp_name = py_utils.construct_exp_name(config=cfg)
    if cfg.cooldown_epoch_num > 0:
        cfg.exp_name += f"_CD{cfg.cooldown_epoch_num}"

    cfg.path = py_utils.construct_path(output_root=cfg.output_root, exp_name=cfg.exp_name)
    cfg.vis_path = os.path.join(cfg.path.exp, "imgs")

    os.makedirs(cfg.path.exp, exist_ok=True)
    os.makedirs(cfg.path.save, exist_ok=True)
    os.makedirs(cfg.path.pth, exist_ok=True)

    with open(cfg.path.log, encoding="utf-8", mode="w") as f:
        f.write(f"=== {datetime.now()} ===\n")
    with open(cfg.path.cfg, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer)

    if os.path.exists(cfg.vis_path):
        shutil.rmtree(cfg.vis_path)
    os.makedirs(cfg.vis_path)

    metric_names = ["Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm", "meanFm", "maxFm"]
    with open(cfg.path.csv, encoding="utf-8", mode="w", newline="") as f:
        writer = csv.writer(f)

        first_row = ["model_name"]
        for dataset_name in cfg.data.test.name:
            first_row.extend([dataset_name] + [" "] * (len(metric_names) - 1))
        writer.writerow(first_row)

        second_row = [" "] + metric_names * len(cfg.data.test.name)
        writer.writerow(second_row)
    return cfg


def main():
    cfg = parse_config()
    pt_utils.initialize_seed_cudnn(seed=cfg.args.base_seed, deterministic=cfg.args.deterministic)
    print(f"[{datetime.now()}] {cfg.path.exp} with base_seed {cfg.args.base_seed}")

    msg_logger = MsgLogger(log=cfg.path.log)

    if hasattr(model_lib, cfg.model_name):
        ModuleClass = getattr(model_lib, cfg.model_name)
        model = ModuleClass(pretrained=cfg.pretrained)
        msg_logger(name="log", msg=inspect.getsource(ModuleClass))
    else:
        raise ModuleNotFoundError(f"Please add <{cfg.model_name}> into the __init__.py.")

    if cfg.load_from:
        model.load_state_dict(torch.load(cfg.load_from, map_location="cpu"))
        print(f"Loaded from {cfg.load_from}")

    model.cuda()
    if not cfg.evaluate:
        training(model=model, msg_logger=msg_logger, cfg=cfg)

    testing(model=model, msg_logger=msg_logger, cfg=cfg)
    print(f"{datetime.now()}: End training...")


if __name__ == "__main__":
    main()
