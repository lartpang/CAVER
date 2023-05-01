# -*- coding: utf-8 -*-
# @Time    : 2021/2/8
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import timm
import torch
import torch.nn as nn
from einops import rearrange

from utils.ops.tensor_ops import cus_sample


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
    ):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name, inplace=False))


class StackedCBRBlock(nn.Sequential):
    def __init__(self, in_c, out_c, num_blocks=1, kernel_size=3):
        assert num_blocks >= 1
        super().__init__()

        if kernel_size == 3:
            kernel_setting = dict(kernel_size=3, stride=1, padding=1)
        elif kernel_size == 1:
            kernel_setting = dict(kernel_size=1)
        else:
            raise NotImplementedError

        cs = [in_c] + [out_c] * num_blocks
        self.channel_pairs = self.slide_win_select(cs, win_size=2, win_stride=1, drop_last=True)
        self.kernel_setting = kernel_setting

        for i, (i_c, o_c) in enumerate(self.channel_pairs):
            self.add_module(name=f"cbr_{i}", module=ConvBNReLU(i_c, o_c, **self.kernel_setting))

    @staticmethod
    def slide_win_select(items, win_size=1, win_stride=1, drop_last=False):
        num_items = len(items)
        i = 0
        while i + win_size <= num_items:
            yield items[i : i + win_size]
            i += win_stride

        if not drop_last:
            # 对于最后不满一个win_size的切片，保留
            yield items[i : i + win_size]


class ConvFFN(nn.Module):
    def __init__(self, dim, out_dim=None, ffn_expand=4):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            StackedCBRBlock(dim, dim * ffn_expand, num_blocks=2, kernel_size=3),
            nn.Conv2d(dim * ffn_expand, out_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PatchwiseTokenReEmbedding:
    @staticmethod
    def encode(x, nh, ph, pw):
        return rearrange(x, "b (nh hd) (nhp ph) (nwp pw) -> b nh (hd ph pw) (nhp nwp)", nh=nh, ph=ph, pw=pw)

    @staticmethod
    def decode(x, nhp, ph, pw):
        return rearrange(x, "b nh (hd ph pw) (nhp nwp) -> b (nh hd) (nhp ph) (nwp pw)", nhp=nhp, ph=ph, pw=pw)


class SpatialViewAttn(nn.Module):
    def __init__(self, dim, p, nh=2):
        super().__init__()
        self.p = p
        self.nh = nh
        self.scale = (dim // nh * self.p ** 2) ** -0.5

        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None, need_weights: bool = False):
        if kv is None:
            kv = q
        N, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)

        # multi-head patch-wise token re-embedding (PTRE)
        q = PatchwiseTokenReEmbedding.encode(q, nh=self.nh, ph=self.p, pw=self.p)
        k = PatchwiseTokenReEmbedding.encode(k, nh=self.nh, ph=self.p, pw=self.p)
        v = PatchwiseTokenReEmbedding.encode(v, nh=self.nh, ph=self.p, pw=self.p)

        qk = torch.einsum("bndx, bndy -> bnxy", q, k) * self.scale
        qk = qk.softmax(-1)
        qkv = torch.einsum("bnxy, bndy -> bndx", qk, v)

        qkv = PatchwiseTokenReEmbedding.decode(qkv, nhp=H // self.p, ph=self.p, pw=self.p)

        x = self.proj(qkv)
        if not need_weights:
            return x
        else:
            # average attention weights over heads
            return x, qk.mean(dim=1)


class ChannelViewAttn(nn.Module):
    def __init__(self, dim, nh):
        super().__init__()
        self.nh = nh
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None):
        if kv is None:
            kv = q
        B, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)
        q = q.reshape(B, self.nh, C // self.nh, H * W)
        k = k.reshape(B, self.nh, C // self.nh, H * W)
        v = v.reshape(B, self.nh, C // self.nh, H * W)

        q = q * (q.shape[-1] ** (-0.5))
        qk = q @ k.transpose(-2, -1)
        qk = qk.softmax(dim=-1)
        qkv = qk @ v

        qkv = qkv.reshape(B, C, H, W)
        x = self.proj(qkv)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, p, nh, ffn_expand):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.sa = SpatialViewAttn(dim, p=p, nh=nh)
        self.ca = ChannelViewAttn(dim, nh=nh)
        self.alpha = nn.Parameter(data=torch.zeros(1))
        self.beta = nn.Parameter(data=torch.zeros(1))

        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = ConvFFN(dim=dim, ffn_expand=ffn_expand, out_dim=dim)

    def forward(self, x):
        normed_x = self.norm1(x)
        x = x + self.alpha.sigmoid() * self.sa(normed_x) + self.beta.sigmoid() * self.ca(normed_x)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, p, nh=4, ffn_expand=1):
        super().__init__()
        self.rgb_norm2 = nn.BatchNorm2d(dim)
        self.depth_norm2 = nn.BatchNorm2d(dim)

        self.depth_to_rgb_sa = SpatialViewAttn(dim, p=p, nh=nh)
        self.depth_to_rgb_ca = ChannelViewAttn(dim, nh=nh)
        self.rgb_alpha = nn.Parameter(data=torch.zeros(1))
        self.rgb_beta = nn.Parameter(data=torch.zeros(1))

        self.rgb_to_depth_sa = SpatialViewAttn(dim, p=p, nh=nh)
        self.rgb_to_depth_ca = ChannelViewAttn(dim, nh=nh)
        self.depth_alpha = nn.Parameter(data=torch.zeros(1))
        self.depth_beta = nn.Parameter(data=torch.zeros(1))

        self.norm3 = nn.BatchNorm2d(2 * dim)
        self.ffn = ConvFFN(dim=2 * dim, ffn_expand=ffn_expand, out_dim=2 * dim)

    def forward(self, rgb, depth):
        normed_rgb = self.rgb_norm2(rgb)
        normed_depth = self.depth_norm2(depth)
        transd_rgb = self.rgb_alpha.sigmoid() * self.depth_to_rgb_sa(
            normed_rgb, normed_depth
        ) + self.rgb_beta.sigmoid() * self.depth_to_rgb_ca(normed_rgb, normed_depth)
        rgb_rgbd = rgb + transd_rgb
        transd_depth = self.depth_alpha.sigmoid() * self.rgb_to_depth_sa(
            normed_depth, normed_rgb
        ) + self.depth_beta.sigmoid() * self.rgb_to_depth_ca(normed_depth, normed_rgb)
        depth_rgbd = depth + transd_depth

        rgbd = torch.cat([rgb_rgbd, depth_rgbd], dim=1)
        rgbd = rgbd + self.ffn(self.norm3(rgbd))
        return rgbd


class CMIU(nn.Module):
    def __init__(self, in_dim, embed_dim, p, nh, ffn_expand):
        super().__init__()
        self.p = p
        self.rgb_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.depth_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )

        self.rgb_imsa = SelfAttention(embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)
        self.depth_imsa = SelfAttention(embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)
        self.imca = CrossAttention(embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)
        self.cssa = SelfAttention(2 * embed_dim, nh=nh, p=p, ffn_expand=ffn_expand)

    def forward(self, rgb, depth, top_rgbd=None):
        """输入均为NCHW"""
        rgb = self.rgb_cnn_proj(rgb)
        depth = self.depth_cnn_proj(depth)

        rgb = self.rgb_imsa(rgb)
        depth = self.depth_imsa(depth)

        rgbd = self.imca(rgb, depth)
        if top_rgbd is not None:
            rgbd = rgbd + top_rgbd

        rgbd = self.cssa(rgbd)
        return rgbd


class CAVER_R50D(nn.Module):
    def __init__(self, ps=(8, 8, 8, 8), embed_dim=64, pretrained=None):
        super().__init__()
        self.rgb_encoder: nn.Module = timm.create_model(
            model_name="resnet50d", features_only=True, out_indices=range(1, 5)
        )
        self.depth_encoder: nn.Module = timm.create_model(
            model_name="resnet50d", features_only=True, out_indices=range(1, 5)
        )
        if pretrained:
            self.rgb_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)
            self.depth_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)

        self.cmius = nn.ModuleList(
            [
                CMIU(in_dim=c, embed_dim=embed_dim, p=p, nh=2, ffn_expand=1)
                for i, (p, c) in enumerate(zip(ps, (2048, 1024, 512, 256)))
            ]
        )

        # predictor
        self.predictor = nn.ModuleList()
        self.predictor.append(StackedCBRBlock(embed_dim * 2, embed_dim))
        self.predictor.append(StackedCBRBlock(embed_dim, 32))
        self.predictor.append(nn.Conv2d(32, 1, 1))

    def forward(self, data):
        rgb_feats = self.rgb_encoder(data["image"])
        depth_feats = self.depth_encoder(data["depth"].repeat(1, 3, 1, 1))

        # to cnn decoder for fusion
        x = self.cmius[0](rgb=rgb_feats[3], depth=depth_feats[3])
        x = self.cmius[1](rgb=rgb_feats[2], depth=depth_feats[2], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.cmius[2](rgb=rgb_feats[1], depth=depth_feats[1], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.cmius[3](rgb=rgb_feats[0], depth=depth_feats[0], top_rgbd=cus_sample(x, scale_factor=2))

        # predictor
        x = self.predictor[0](cus_sample(x, scale_factor=2))
        x = self.predictor[1](cus_sample(x, scale_factor=2))
        x = self.predictor[2](x)
        return x


class CAVER_R101D(nn.Module):
    def __init__(self, ps=(8, 8, 8, 8), embed_dim=64, pretrained=None):
        super().__init__()
        self.rgb_encoder: nn.Module = timm.create_model(
            model_name="resnet101d", features_only=True, out_indices=range(1, 5)
        )
        self.depth_encoder: nn.Module = timm.create_model(
            model_name="resnet101d", features_only=True, out_indices=range(1, 5)
        )
        if pretrained:
            self.rgb_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)
            self.depth_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)

        self.cmius = nn.ModuleList(
            [
                CMIU(in_dim=c, embed_dim=embed_dim, p=p, nh=2, ffn_expand=1)
                for i, (p, c) in enumerate(zip(ps, (2048, 1024, 512, 256)))
            ]
        )

        # predictor
        self.predictor = nn.ModuleList()
        self.predictor.append(StackedCBRBlock(embed_dim * 2, embed_dim))
        self.predictor.append(StackedCBRBlock(embed_dim, 32))
        self.predictor.append(nn.Conv2d(32, 1, 1))

    def forward(self, data):
        rgb_feats = self.rgb_encoder(data["image"])
        depth_feats = self.depth_encoder(data["depth"].repeat(1, 3, 1, 1))

        # to cnn decoder for fusion
        x = self.cmius[0](rgb=rgb_feats[3], depth=depth_feats[3])
        x = self.cmius[1](rgb=rgb_feats[2], depth=depth_feats[2], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.cmius[2](rgb=rgb_feats[1], depth=depth_feats[1], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.cmius[3](rgb=rgb_feats[0], depth=depth_feats[0], top_rgbd=cus_sample(x, scale_factor=2))

        # predictor
        x = self.predictor[0](cus_sample(x, scale_factor=2))
        x = self.predictor[1](cus_sample(x, scale_factor=2))
        x = self.predictor[2](x)
        return x
