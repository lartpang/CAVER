# -*- coding: utf-8 -*-
# @Time    : 2020
# @Author  : Lart Pang
# @FileName: BaseOps.py
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn.functional as F


def cus_sample(feat: torch.Tensor, align_corners=False, mode="bilinear", **kwargs) -> torch.Tensor:
    """
    Args:
        feat: 输入特征
        mode: 插值模式
        align_corners: 具体差异可见https://www.yuque.com/lart/idh721/ugwn46
        kwargs: size/scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]

    if size := kwargs.get("size", False):
        assert isinstance(size, (tuple, list))
        if isinstance(size, list):
            size = tuple(size)
        if size == tuple(feat.shape[2:]):
            return feat
    elif scale_factor := kwargs.get("scale_factor", False):
        assert isinstance(size, (int, float))
        if scale_factor == 1:
            return feat
        # if isinstance(scale_factor, float):
        kwargs["recompute_scale_factor"] = False
    else:
        print("size or scale_factor is not be assigned, the feat will not be resized...")
        return feat
    if mode == "nearest":
        if align_corners is False:
            align_corners = None
        assert align_corners is None, (
            "align_corners option can only be set with the interpolating modes: "
            "linear | bilinear | bicubic | trilinear, so we will set it to None"
        )
    return F.interpolate(feat, mode=mode, align_corners=align_corners, **kwargs)


def upsample_add(*xs: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    resize xs[:-1] to the size of xs[-1] and add them together.

    Args:
        xs:
        kwargs: config for cus_sample
    """
    y = xs[-1]
    for x in xs[:-1]:
        y = y + cus_sample(x, size=y.size()[2:], **kwargs)
    return y


def upsample_cat(*xs: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    resize xs[:-1] to the size of xs[-1] and concat them together.

    Args:
        xs:
        kwargs: config for cus_sample
    """
    y = xs[-1]
    out = []
    for x in xs[:-1]:
        out.append(cus_sample(x, size=y.size()[2:], **kwargs))
    return torch.cat([*out, y], dim=1)


def upsample_reduce(b, a, **kwargs):
    """
    上采样所有特征到最后一个特征的尺度以及前一个特征的通道数
    """
    _, C, _, _ = b.size()
    N, _, H, W = a.size()

    b = cus_sample(b, size=(H, W), **kwargs)
    a = a.reshape(N, -1, C, H, W).mean(1)
    return b + a


def shuffle_channels(x, groups):
    """
    Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]
    一共C个channel要分成g组混合的channel，先把C reshape成(g, C/g)的形状，
    然后转置成(C/g, g)最后平坦成C组channel
    """
    N, C, H, W = x.size()
    x = x.reshape(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4)
    return x.reshape(N, C, H, W)


def clip_normalize_scale(array, clip_min=0, clip_max=250, new_min=0, new_max=255):
    array = np.clip(array, a_min=clip_min, a_max=clip_max)
    array = (array - array.min()) / (array.max() - array.min())
    array = array * (new_max - new_min) + new_min
    return array


if __name__ == "__main__":
    a = torch.rand(3, 4, 10, 10)
    b = torch.rand(3, 2, 5, 5)
    print(upsample_reduce(b, a).size())
