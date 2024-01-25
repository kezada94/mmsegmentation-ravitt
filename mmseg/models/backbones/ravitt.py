from timm.models import VisionTransformer
from timm.models import create_model
from torch import Tensor
from einops.layers.torch import Rearrange
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import _load_checkpoint
from scipy import interpolate
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.registry import MODELS
from ..utils import PatchEmbed
from .vit import TransformerEncoderLayer as VisionTransformerEncoderLayer
import math


class RaViTTPatchEmbedding(nn.Module):
    def __init__(self, proj: nn.Module, norm: nn.Module, patch_size: int = 16, img_size: int = 224, isFull: bool = False, npatches: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.proj = proj
        self.norm = norm
        self.isFull = isFull
        self.npatches = npatches
        self.rearr = Rearrange('b (lh lw) c ph pw -> b c (lh ph) (lw pw)', lh=(img_size //
                               patch_size), lw=(img_size // patch_size), ph=patch_size, pw=patch_size)
        if isFull:
            self.rearr = Rearrange('b (lh lw) c ph pw -> b c (lh ph) (lw pw)',
                                   lh=1, lw=npatches, ph=patch_size, pw=patch_size)
        # base_grid0 = torch.arange(0, crop_size, device=input_batch.device).view(1, 1, -1).repeat(npatches, crop_size, 1)
        # base_grid1 = torch.arange(0, crop_size, device=input_batch.device).view(1, -1, 1).repeat(npatches, 1, crop_size)

    def overlap_positional_encoding(self, batch_size, patches, embedded_dim, positions):

        if embedded_dim % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(embedded_dim))
        sqrt_p = patches**0.5
        dev = positions.device
        if (sqrt_p).is_integer():
            npatches_h = int(sqrt_p)
            npatches_w = npatches_h
        else:
            npatches_w = (patches)
            npatches_h = 1
        pe = torch.zeros(batch_size, npatches_h, npatches_w,
                         embedded_dim, device=dev)

        # Each dimension use half of embedded_dim
        d_embed = embedded_dim
        embedded_dim = d_embed // 2

        den = (10000 ** (4 * torch.arange(0, embedded_dim // 2,
               device=dev) / d_embed)).repeat(batch_size, 1, 1, 1)

        pos_h = positions[:, :, 0].reshape(
            batch_size, npatches_h, npatches_w).unsqueeze(-1).repeat(1, 1, 1, embedded_dim // 2)
        pos_w = positions[:, :, 1].reshape(
            batch_size, npatches_h, npatches_w).unsqueeze(-1).repeat(1, 1, 1, embedded_dim // 2)

        pe[:, :, :, 0:embedded_dim:2] = torch.sin(pos_h / den)
        pe[:, :, :, 1:embedded_dim:2] = torch.cos(pos_h / den)
        pe[:, :, :, embedded_dim::2] = torch.sin(pos_w / den)
        pe[:, :, :, embedded_dim + 1::2] = torch.cos(pos_w / den)
        pe = (pe - pe.mean()) / pe.std() * 0.02

        return pe.reshape(batch_size, patches, d_embed)

    def random_patch_extraction(self, input_batch, crop_size: int):
        batch_size, npatches, _, height, width = input_batch.shape
        n = height // crop_size

        # Generate random crop coordinates for each image in the batch
        crop_top = torch.randint(
            0, height - crop_size + 1, (npatches,), device=input_batch.device)
        crop_left = torch.randint(
            0, width - crop_size + 1, (npatches,), device=input_batch.device)
        # crop_top = torch.arange(n, device=input_batch.device).unsqueeze(1).repeat(1,n).view(-1)*crop_size
        # crop_left = torch.arange(n, device=input_batch.device).repeat(n)*crop_size
        # Create a grid of coordinates for each image in the batch
        grid = torch.zeros((npatches, crop_size, crop_size,
                           2), device=input_batch.device)
        grid[:, :, :, 0] = torch.arange(0, crop_size, device=input_batch.device).view(1,
                                                                                      1, -1).repeat(npatches, crop_size, 1)
        grid[:, :, :, 1] = torch.arange(0, crop_size, device=input_batch.device).view(
            1, -1, 1).repeat(npatches, 1, crop_size)

        # Add the crop coordinates to the grid
        grid[:, :, :, 0] += crop_left.view(-1, 1, 1)
        grid[:, :, :, 1] += crop_top.view(-1, 1, 1)

        grid[:, :, :, 0] = 2 * (grid[:, :, :, 0]) / width - 1
        grid[:, :, :, 1] = 2 * (grid[:, :, :, 1]) / height - 1

        # create an empty tensor to accumulate the crops
        crops_positions = torch.stack(
            (crop_top / crop_size, crop_left / crop_size), dim=1)
        output_batch = torch.zeros(
            (batch_size, npatches, 3, crop_size, crop_size), device=input_batch.device)
        for i in range(batch_size):
            output_batch[i] = (torch.nn.functional.grid_sample(
                input_batch[i], grid, align_corners=True))
        return output_batch, crops_positions.expand(batch_size, npatches, 2)

    def ramdomizedPatchExtraction(self, batch, patch_size: int = 16, npatches: int = 192):
        x = batch.unsqueeze(1).expand(-1, npatches, -1, -1, -1)
        x, pos = self.random_patch_extraction(x, crop_size=patch_size)
        return self.rearr(x), pos

    def forward(self, x):
        if self.isFull:
            x, pos = self.ramdomizedPatchExtraction(
                x, patch_size=self.patch_size, npatches=self.npatches)
        else:
            x, pos = self.ramdomizedPatchExtraction(
                x, patch_size=self.patch_size, npatches=(self.img_size // self.patch_size)**2)
        x = self.proj(x.to())
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)

        return x, pos


def interlaced(x, x_rv, t):
    mask = torch.rand((x.shape[1],), device=x.device) < t
    mask = mask.to(x.dtype).unsqueeze(
        0).unsqueeze(2).expand(x.shape[0], -1, -1)
    return x_rv * mask + x * (1 - mask)


def new_forward(self, x):
    B, C, H, W = x.shape

    x_og = self.patch_embed(x)  # embed vit
    x_og = self._pos_embed(x_og)  # add positional encoding vit

    if self.training or self.isFull:
        x_rv, pos = self.ravitt(x)  # embed ravitt
        # add positional encoding ravitt
        x_rv = x_rv + self.ravitt.overlap_positional_encoding(
            x_rv.shape[0], x_rv.shape[1], x_rv.shape[2], pos)
        slice_to_append = x_og[:, 0, :].unsqueeze(
            1)  # Shape will be (256, 1, 128)

        x_rv = torch.cat((slice_to_append, x_rv), dim=1)
        x = self.ravitt_func(x_og, x_rv)
    else:
        x = x_og

    x = self.norm_pre(x)
    features = []
    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = self.checkpoint_seq(self.blocks, x)
    else:
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                xp = x[:, 1:, :]
                xp = self.reshaper(xp)
                features.append(xp.contiguous())

    return features


@MODELS.register_module()
class RaViTT(BaseModule):
    def __init__(self, model_path='deit_tiny_patch16_224',
                 ravitt_t=1.0,
                 ravitt_mode='full',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=80,
                 use_checkpoint=False,
                 embed_dims=192,
                 depth=12,
                 num_layers=12,
                 num_heads=4,
                 mlp_ratio=4,
                 qv_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 num_fcs=2,
                 norm_eval=False,
                 pretrained=None,
                 init_values=0.1,
                 init_cfg=None,
                 out_indices=[3, 5, 7, 11]):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')


        npatch = round(((img_size // patch_size)**2) * ravitt_t)
        t = ravitt_t
        self.model = create_model(
            model_path, img_size=img_size, patch_size=patch_size, in_chans=in_channels)

        if ravitt_mode == 'full':
            print(f'Using {npatch} patches')
            self.model.ravitt = RaViTTPatchEmbedding(
                self.model.patch_embed.proj, self.model.patch_embed.norm, patch_size=patch_size, img_size=img_size, isFull=True, npatches=npatch)
            self.model.isFull = True
        else:
            self.model.ravitt = RaViTTPatchEmbedding(
                self.model.patch_embed.proj, self.model.patch_embed.norm, patch_size=patch_size, img_size=img_size)
            self.model.isFull = False

        if ravitt_mode == 'interlaced':
            self.model.ravitt_func = lambda x, x_rv: interlaced(x, x_rv, t)
        elif ravitt_mode == 'avg':
            self.model.ravitt_func = lambda x, x_rv: x*(1-t) + x_rv*t
        elif ravitt_mode == 'choice':
            self.model.ravitt_func = lambda x, x_rv: x_rv if torch.rand(
                1) < t else x
        elif ravitt_mode == 'full':
            self.model.ravitt_func = lambda x, x_rv: x_rv
        else:
            self.model.ravitt_func = lambda x, x_rv: x

        self.model.reshaper = Rearrange(
            'b (th tw) e -> b e th tw', th=(img_size // patch_size), tw=(img_size // patch_size))
        self.num_classes = num_classes
        self.model.num_classes = num_classes
        self.out_indices = out_indices
        self.patch_size = patch_size
        self.model.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        self.model.t = ravitt_t
        self.model.forward = new_forward.__get__(self.model, VisionTransformer)
        self.model.out_indices = out_indices

        self.blocks = self.model.blocks

        self.model.head = None
        self.model.head_drop = None
        self.model.norm = None
        self.model.fc_norm = None


    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if (isinstance(self.init_cfg, dict) and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint = torch.load(self.init_cfg['checkpoint'], map_location='cpu')

            checkpoint_model = checkpoint['model']
            state_dict = self.state_dict()
            print("LOADING A PRETRAINEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEED")
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.model.patch_embed.num_patches
            num_extra_tokens = self.model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            
            self.load_state_dict(checkpoint_model, strict=False)
        elif self.init_cfg is not None:
            super().init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            # Copyright 2019 Ross Wightman
            # Licensed under the Apache License, Version 2.0 (the "License")
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def forward(self, x):
        features = self.model.forward(x)
        return tuple(features)
