# -*- coding: utf-8 -*-
import contextlib
import os
from collections import OrderedDict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import functional as tv_tf
from torchvision.utils import make_grid

from utils.constructor.scheduler import LRAdjustor


def pltplot_results(data_container, save_path=None):
    """Plot the results conresponding to the batched images based on the `make_grid` method from `torchvision`.

    Args:
        data_container (dict): Dict containing data you want to plot.
        save_path (str): Path of the exported image.
    """
    axes = plt.subplots(nrows=len(data_container), ncols=1)[1].ravel()
    plt.subplots_adjust(hspace=0.03, left=0.05, bottom=0.01, right=0.99, top=0.99)

    for subplot_id, (name, data) in enumerate(data_container.items()):
        grid = make_grid(data, nrow=data.shape[0], padding=2, normalize=False)
        grid_image = np.asarray(tv_tf.to_pil_image(grid))
        axes[subplot_id].imshow(grid_image)
        axes[subplot_id].set_ylabel(name)
        axes[subplot_id].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close('all')


def cvplot_results(data_container, save_path=None, base_size=256):
    """Plot the results conresponding to the batched images based on the `make_grid` method from `torchvision`.

    Args:
        data_container (dict): Dict containing data you want to plot.
        save_path (str): Path of the exported image.
    """
    font_cfg = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)

    grids = []
    for subplot_id, (name, data) in enumerate(data_container.items()):
        if data.ndim == 3:
            data = data.unsqueeze(1)

        grid = make_grid(data, nrow=data.shape[0], padding=2, normalize=False)
        grid = np.array(tv_tf.to_pil_image(grid.float()))
        h, w = grid.shape[:2]
        ratio = base_size / h
        grid = cv2.resize(grid, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

        (text_w, text_h), baseline = cv2.getTextSize(text=name, **font_cfg)
        text_xy = 20, 20 + text_h // 2 + baseline
        cv2.putText(grid, text=name, org=text_xy, color=(255, 255, 255), **font_cfg)

        grids.append(grid)
    grids = np.concatenate(grids, axis=0)  # H,W,C

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, grids)


def plot_lr_curve_for_scheduler(optimizer, scheduler: LRAdjustor, num_steps, save_path=None):
    plt.rc("xtick", labelsize="small")
    plt.rc("ytick", labelsize="small")

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 4), dpi=600)
    # give plot a title
    ax.set_title("Learning Rate Curve")
    # make axis labels
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate")

    x_data = np.arange(num_steps)
    ys = []
    for x in x_data:
        optimizer.zero_grad()
        scheduler(optimizer=optimizer, curr_idx=x)
        ys.append(max([group["lr"] for group in optimizer.param_groups]))
    y_data = np.array(ys)

    # set lim
    x_min, x_max = 0, num_steps - 1
    dx = num_steps * 0.1
    ax.set_xlim(x_min - dx, x_max + 2 * dx)

    y_min, y_max = y_data.min(), y_data.max()
    dy = (y_data.max() - y_data.min()) * 0.1
    ax.set_ylim((y_min - dy, y_max + dy))

    marker_on = [0, -1]
    key_point_xs = [0, num_steps - 1]
    for idx in range(1, len(y_data) - 1):
        prev_y = y_data[idx - 1]
        curr_y = y_data[idx]
        next_y = y_data[idx + 1]
        if ((curr_y > prev_y and curr_y >= next_y) or (curr_y >= prev_y and curr_y > next_y)
                or (curr_y <= prev_y and curr_y < next_y) or (curr_y < prev_y and curr_y <= next_y)):
            marker_on.append(idx)
            key_point_xs.append(idx)

    marker_on = sorted(set(marker_on))
    key_point_xs = sorted(set(key_point_xs))
    key_point_ys = []
    for x in key_point_xs:
        y = y_data[x]
        key_point_ys.append(y)
        ax.annotate(
            text=f"({x:d},{y:.3e})",
            xy=(x, y),
            xytext=(-10, +10),
            size="small",
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"),
            rotation="vertical",
        )

    # set ticks
    ax.set_xticks(key_point_xs)
    # ax.set_yticks(key_point_ys)

    ax.plot(x_data, y_data, marker="o", markevery=marker_on)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def construct_print(out_str: str, total_length: int = 80):
    if len(out_str) >= total_length:
        out_str = "[ ==>>\n" + out_str + "\n <<== ]"
    else:
        space_str = " " * ((total_length - len(out_str)) // 2 - 6)
        out_str = "[ ->> " + space_str + out_str + space_str + " <<- ]"
    print(out_str)


def construct_path(output_root: str, exp_name: str) -> dict:
    proj_path = os.path.join(output_root, exp_name)

    exp_idx = 0
    exp_path = os.path.join(proj_path, f"exp_{exp_idx}")
    while os.path.exists(exp_path):
        exp_idx += 1
        exp_path = os.path.join(proj_path, f"exp_{exp_idx}")

    tb_path = os.path.join(exp_path, "tb")
    save_path = os.path.join(exp_path, "pre")

    pth_path = os.path.join(exp_path, "pth")
    state_file = os.path.join(pth_path, "state.pth")

    log_file = os.path.join(exp_path, "log.txt")
    cfg_file = os.path.join(exp_path, "cfg.py")
    trainer_file = os.path.join(exp_path, "trainer.txt")
    excel_file = os.path.join(exp_path, "results.xlsx")
    csv_file = os.path.join(exp_path, "results.csv")

    path_config = {
        # "pth_log": proj_path,
        "exp": exp_path,
        "tb": tb_path,
        "save": save_path,
        "pth": pth_path,
        "state": state_file,
        "log": log_file,
        "cfg": cfg_file,
        "trainer": trainer_file,
        "excel": excel_file,
        "csv": csv_file,
    }

    return path_config


def construct_exp_name(config: dict):
    # bs_16_lr_0.05_e30_noamp_2gpu_noms_352
    focus_item = OrderedDict(
        args=dict(batch_size="bs", epoch_num="e", iter_num="i", use_amp="amp"),
        optimizers=dict(lr="lr", strategy="ot", optimizer="op"),
        schedulers=dict(strategy="lt"),
    )

    def _format_item(_i):
        if isinstance(_i, bool):
            _i = "y" if _i else "n"
        elif isinstance(_i, (list, tuple)):
            _i = "y" if _i else "n"  # 只是判断是否非空
        elif isinstance(_i, str):
            if "_" in _i:
                _i = _i.replace("_", "").lower()
        elif _i is None:
            _i = "n"
        return _i

    if (epoch_based := config.args.get("epoch_based", None)) is not None and (not epoch_based):
        focus_item["args"].pop("epoch_num")
    else:
        # 默认基于epoch
        focus_item["args"].pop("iter_num")

    exp_name = f"{config['model_name']}_{config.data.train.shape.h}x{config.data.train.shape.w}"
    for parent_key, info_dict in focus_item.items():
        for name, alias in info_dict.items():
            if (item := config[parent_key].get(name, None)) is not None:  # if需要兼容item=false
                exp_name += f"_{alias.upper()}{_format_item(item)}"
            else:
                raise KeyError(f"{name} must be contained in {list(config[parent_key].keys())}")
    if "info" in config and config["info"]:
        exp_name += f"_INFO{config['info'].lower()}"
    return exp_name


def get_data_collection(dataset_names, all_data: dict) -> list:
    if not isinstance(dataset_names, (tuple, list)):
        dataset_names = [dataset_names]

    data_collection = []
    for name in dataset_names:
        data_path = all_data.get(name, None)
        if data_path is None:
            raise KeyError(f"{name} must be contained in {list(all_data.keys())}")
        data_collection.append((name, data_path))
    return data_collection


@contextlib.contextmanager
def open_file(file_path, mode, encoding=None):
    """
    提供了打开关闭的显式提示
    """
    print(f"打开文件{file_path}")
    f = open(file_path, encoding=encoding, mode=mode)
    yield f
    print(f"关闭文件{file_path}")
    f.close()


def are_the_same(file_path_1, file_path_2, buffer_size=8 * 1024):
    """
    通过逐块比较两个文件的二进制数据是否一致来确定两个文件是否是相同内容

    REF: https://zhuanlan.zhihu.com/p/142453128

    Args:
        file_path_1: 文件路径
        file_path_2: 文件路径
        buffer_size: 读取的数据片段大小，默认值8*1024

    Returns: dict(state=True/False, msg=message)
    """
    st1 = os.stat(file_path_1)
    st2 = os.stat(file_path_2)

    # 比较文件大小
    if st1.st_size != st2.st_size:
        return dict(state=False, msg="文件大小不一致")

    with open(file_path_1, mode="rb") as f1, open(file_path_2, mode="rb") as f2:
        while True:
            b1 = f1.read(buffer_size)  # 读取指定大小的数据进行比较
            b2 = f2.read(buffer_size)
            if b1 != b2:
                msg = (f"存在差异:"
                       f"\n{file_path_1}\n==>\n{b1.decode('utf-8')}\n<=="
                       f"\n{file_path_2}\n==>\n{b2.decode('utf-8')}\n<==")
                return dict(state=False, msg=msg)
            # b1 == b2
            if not b1:
                # b1 == b2 == False (b'')
                return dict(state=True, msg="完全一样")


def all_items_in_string(items, target_str):
    """判断items中是否全部都是属于target_str一部分的项"""
    for i in items:
        if i not in target_str:
            return False
    return True


def any_item_in_string(items, target_str):
    """判断items中是否存在属于target_str一部分的项"""
    for i in items:
        if i in target_str:
            return True
    return False


def slide_win_select(items, win_size=1, win_stride=1, drop_last=False):
    num_items = len(items)
    i = 0
    while i + win_size <= num_items:
        yield items[i:i + win_size]
        i += win_stride

    if not drop_last:
        # 对于最后不满一个win_size的切片，保留
        yield items[i:i + win_size]


def iterate_nested_sequence(nested_sequence):
    """
    当前支持list/tuple/int/float/range()的多层嵌套，注意不要嵌套的太深，小心超出python默认的最大递归深度

    例子
    ::

        for x in iterate_nested_sequence([[1, (2, 3)], range(3, 10), 0]):
            print(x)

        1
        2
        3
        3
        4
        5
        6
        7
        8
        9
        0

    :param nested_sequence: 多层嵌套的序列
    :return: generator
    """
    for item in nested_sequence:
        if isinstance(item, (int, float)):
            yield item
        elif isinstance(item, (list, tuple, range)):
            yield from iterate_nested_sequence(item)
        else:
            raise NotImplementedError


def color_image(image, mask, bgr_color, ratio=0.5):
    """使用mask按照ratio比例为image上色

    Args:
        image (np.uint8): 原始图像
        mask (np.bool): 目标区域掩码
        bgr_color (list, 3): bgr颜色列表
        ratio (float, optional): 比例. Defaults to 0.5.

    Returns:
        np.uint8: 上色后的图像
    """
    image = image.copy()
    image[mask] = ((1 - ratio) * image[mask] + ratio * np.array(bgr_color)).astype(np.uint8)
    return image
