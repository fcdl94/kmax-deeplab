"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: https://github.com/google-research/deeplab2/blob/main/model/pixel_decoder/kmax.py
"""

from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_tf_ as trunc_normal_

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from torch.cuda.amp import autocast

from ..backbone.convnext import LayerNorm

import math


def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()


def get_norm(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=None, act=None,
                 conv_type='2d', conv_init='he_normal', norm_init=1.0):
        super().__init__()
        
        if conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


MAX_SPAN = 255
def _compute_relative_distance_matrix(query_length, key_length):
    if (key_length - query_length) % 2:
        raise ValueError('Key_length should be query_length + 2 * memory_flange.')
    key_index = torch.arange(key_length)
    query_index = torch.arange(query_length) + (key_length - query_length) // 2
    distance_matrix = key_index[None, :] - query_index[:, None]
    # Shift the distance_matrix so that it is >= 0. Each entry of the
    # distance_matrix distance will index a relative positional embedding.
    distance_matrix = distance_matrix + MAX_SPAN - 1
    return distance_matrix

class RelativePositionalEncoding(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self._embeddings = nn.Embedding(MAX_SPAN * 2 - 1, depth)
        trunc_normal_(self._embeddings.weight, std=1.0)
        self.depth = depth

<<<<<<< HEAD
    def forward(self, query_length, key_length):
        _relative_distance_matrix = _compute_relative_distance_matrix(query_length, key_length)
        return self._embeddings.weight[_relative_distance_matrix.reshape(-1)].reshape(query_length, key_length, self.depth)
=======
    def forward(self, query_lenght, key_length):
        _relative_distance_matrix = _compute_relative_distance_matrix(query_length, key_length)
        return self._embeddings.weight[self._relative_distance_matrix.reshape(-1)].reshape(query_length, key_length, self.depth)
>>>>>>> 9110d89ad9cd07840ebf4a1bc814b4a13da1ebb8


# https://github.com/google-research/deeplab2/blob/main/model/layers/axial_layers.py#L36
class AxialAttention(nn.Module):
    def __init__(self, in_planes, total_key_depth=512, total_value_depth=1024, num_heads=8):
        assert (total_key_depth % num_heads == 0) and (total_value_depth % num_heads == 0)
        super().__init__()
        self._in_planes = in_planes
        self._total_key_depth = total_key_depth
        self._total_value_depth = total_value_depth
        self._num_heads = num_heads
        self._key_depth_per_head = total_key_depth // num_heads

        self.qkv_transform = ConvBN(in_planes, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, stride=1,
                                       padding=0, bias=False, norm=None, act=None, conv_type='1d')
        trunc_normal_(self.qkv_transform.conv.weight, std=in_planes ** -0.5)

        self._query_rpe = RelativePositionalEncoding(self._key_depth_per_head)
        self._key_rpe = RelativePositionalEncoding(self._key_depth_per_head)
        self._value_rpe = RelativePositionalEncoding(total_value_depth // num_heads)

        self._batch_norm_qkv = get_norm('syncbn', self._total_key_depth * 2 + self._total_value_depth)
        self._batch_norm_similarity = get_norm('syncbn', num_heads * 3)
        self._batch_norm_retrieved_output = get_norm('syncbn', self._total_value_depth * 2)


    def forward(self, x):
        N, C, L = x.shape
        qkv = self._batch_norm_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv, [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        q = q.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        k = k.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        v = v.reshape(N, self._num_heads, self._total_value_depth // self._num_heads, L)

        similarity_logits = []
        content_similarity = torch.einsum('bhdl,bhdm->bhlm', q, k)
        query_rpe = self._query_rpe(L, L)
        query_rpe_similarity = torch.einsum('bhdl,lmd->bhlm', q, query_rpe)
        key_rpe = self._key_rpe(L, L)
        key_rpe_similarity = torch.einsum('bhdm,lmd->bhlm', k, key_rpe)
        similarity_logits = torch.cat([content_similarity, query_rpe_similarity, key_rpe_similarity], dim=1)
        similarity_logits = self._batch_norm_similarity(similarity_logits).reshape(N, 3, self._num_heads, L, L).sum(dim=1)

        with autocast(enabled=False):
            weights = F.softmax(similarity_logits.float(), dim=-1)

        retrieved_content = torch.einsum('bhlm,bhdm->bhdl', weights, v)
        value_rpe = self._value_rpe(L, L)
        retrieved_rpe = torch.einsum('bhlm,lmd->bhdl', weights, value_rpe)

        retrieved_output = torch.cat([retrieved_content, retrieved_rpe], dim=1).reshape(N, 2*self._total_value_depth, L)
        retrieved_output = self._batch_norm_retrieved_output(retrieved_output).reshape(N, 2, self._total_value_depth, L).sum(1)

        return retrieved_output


# https://github.com/google-research/deeplab2/blob/main/model/layers/axial_layers.py#L316
class AxialAttention2D(nn.Module):
    def __init__(self, in_planes, filters=512, key_expansion=1, value_expansion=2, num_heads=8):
        super().__init__()
        total_key_depth = int(round(filters * key_expansion)) # 1
        total_value_depth = int(round(filters * value_expansion)) # 2
        self._total_key_depth = total_key_depth 
        self._total_value_depth = total_value_depth
        self._height_axis = AxialAttention(
            in_planes=in_planes,
            total_key_depth=total_key_depth,
            total_value_depth=total_value_depth,
            num_heads=num_heads)
        self._width_axis = AxialAttention(
            in_planes=total_value_depth,
            total_key_depth=total_key_depth,
            total_value_depth=total_value_depth,
            num_heads=num_heads)

    def forward(self, x):
        # N C H W -> N W C H
        N, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # channel last
        x = x.reshape(N*W, C, H)  # merge batch size and widht 
        x = self._height_axis(x)  # compute axial height attention
        # N W C H -> N H C W
        x = x.reshape(N, W, self._total_value_depth, H).permute(0, 3, 2, 1).contiguous() # prepare for width
        x = x.reshape(N*H, self._total_value_depth, W)
        x = self._width_axis(x)  # compute width
        x = x.reshape(N, H, self._total_value_depth, W).permute(0, 2, 1, 3).contiguous() 
        x = x.reshape(N, self._total_value_depth, H, W) # back to original shape
        return x


# https://github.com/google-research/deeplab2/blob/main/model/layers/axial_blocks.py#L36
class SingleBlock(nn.Module):

    def __init__(self, inplanes, filter_list, block_type, key_expansion=1, value_expansion=2, num_heads=8, drop_path_prob=0.0):
        super(SingleBlock, self).__init__()
        self._block_type = block_type.lower()
        self._filter_list = filter_list # dec_channel[i] x 2, x 1, x 4
        self._conv1_bn_act = ConvBN(inplanes, self._filter_list[0], kernel_size=1, bias=False, norm='syncbn', act='gelu')
        if self._block_type == 'axial':
            # Here there is query shape, a constant parameter indicating the shape of the activations
            self._attention = AxialAttention2D(in_planes=self._filter_list[0], filters=self._filter_list[1],
                                                key_expansion=key_expansion, value_expansion=value_expansion, num_heads=num_heads)
            output_channel = filter_list[1] * value_expansion
        elif self._block_type == 'bottleneck':
            self._conv2_bn_act = ConvBN(self._filter_list[0], self._filter_list[1], kernel_size=3, padding=1, bias=False, norm='syncbn', act='gelu')
            output_channel = filter_list[1]
        self._conv3_bn = ConvBN(output_channel, self._filter_list[2], kernel_size=1, bias=False, norm='syncbn', act=None, norm_init=0.0)

        self._shortcut = None
        if inplanes != self._filter_list[-1]:
            self._shortcut = ConvBN(inplanes, self._filter_list[-1], kernel_size=1, bias=False, norm='syncbn', act=None)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity() 

    def forward(self, x):
        x = F.gelu(x)

        shortcut = x
        if self._shortcut is not None:
            shortcut = self._shortcut(shortcut)

        x = self._conv1_bn_act(x)
        if self._block_type == 'axial':
            x = self._attention(x)
            x = F.gelu(x)
        elif self._block_type == 'bottleneck':
            x = self._conv2_bn_act(x)
        x = self._conv3_bn(x)

        x = self.drop_path(x) + shortcut

        return x


# https://github.com/google-research/deeplab2/blob/main/model/layers/axial_block_groups.py#L42
class BlockGroup(nn.Module):
    def __init__(self, inplanes, base_filter, num_blocks, block_type, **kwargs):
        super().__init__()
        self._num_blocks = num_blocks
        block_type = block_type.lower()
        if block_type == 'axial':
            # https://github.com/google-research/deeplab2/blob/main/model/layers/axial_block_groups.py#L247
            filter_list = [base_filter * 2, base_filter, base_filter * 4]
        elif block_type == 'bottleneck':
            # https://github.com/google-research/deeplab2/blob/main/model/layers/axial_block_groups.py#L250
            filter_list = [base_filter, base_filter, base_filter * 4]

        self._blocks = nn.ModuleList()
        for i in range(num_blocks):
            self._blocks.append(SingleBlock(inplanes=inplanes, filter_list=filter_list, block_type=block_type, **kwargs))
            inplanes = filter_list[-1]

    def forward(self, x):
        for i in range(self._num_blocks):
            x = self._blocks[i](x)
        return x


# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/resized_fuse.py#L31
class ResizedFuse(nn.Module):
    def __init__(self, low_in_channels, high_in_channels, out_channels):
        super().__init__()
        self.low_in_channels = low_in_channels
        self.high_in_channels = high_in_channels
        self.out_channels = out_channels
        if low_in_channels != out_channels:
            self._conv_bn_low = ConvBN(low_in_channels, out_channels, kernel_size=1, bias=False, norm='syncbn', act=None)
        if high_in_channels != out_channels:
            self._conv_bn_high = ConvBN(high_in_channels, out_channels, kernel_size=1, bias=False, norm='syncbn', act=None)

    def forward(self, lowres_x, highres_x):

        align_corners = False
        if self.low_in_channels != self.out_channels:
            lowres_x = F.gelu(lowres_x)
            lowres_x = self._conv_bn_low(lowres_x)
            lowres_x = F.interpolate(lowres_x, size=highres_x.shape[2:], mode='bilinear', align_corners=align_corners)
        else:
            lowres_x = F.interpolate(lowres_x, size=highres_x.shape[2:], mode='bilinear', align_corners=align_corners)

        if self.high_in_channels != self.out_channels:
            highres_x = F.gelu(highres_x)
            highres_x = self._conv_bn_high(highres_x)

        return lowres_x + highres_x


@SEM_SEG_HEADS_REGISTRY.register()
class kMaXPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        dec_layers: List[int],
        dec_channels: List[int],
        layer_types: List[str],
        drop_path_prob: float,
    ):
        """
        NOTE: this interface is experimental.
        Args:
        """
        super().__init__()
        self.num_stages = len(input_shape) # stem, r2, r3, r4, r5 -> 5
        assert self.num_stages == len(dec_layers) and self.num_stages == len(dec_channels) and self.num_stages == len(layer_types)  # 1 axial, 5 axial, 1 bottleneck, 1 bottleneck, 1 bottleneck
        input_shape = sorted(input_shape.items(), key=lambda x: -x[1].stride)  # sor by -stride, i.e. starting from "res5" to "res2"/"stem"
        self.in_features = [k for k, v in input_shape] # starting from "res5" to "res2"/"stem"
        in_channels = [v.channels for k, v in input_shape]

        self._in_norms = nn.ModuleList()
        self._stages = nn.ModuleList()
        self._resized_fuses = nn.ModuleList()

        for i in range(self.num_stages):
            self._in_norms.append(LayerNorm(in_channels[i], data_format="channels_first"))
            inplanes = in_channels[i] if i == 0 else dec_channels[i]
            self._stages.append(BlockGroup(inplanes=inplanes,
                base_filter=dec_channels[i], num_blocks=dec_layers[i], block_type=layer_types[i],
                key_expansion=1, value_expansion=2, num_heads=8, drop_path_prob=drop_path_prob))

            if i > 0:
                self._resized_fuses.append(ResizedFuse(
                    low_in_channels=dec_channels[i-1] * 4,  # Why times 4? The blocks outputs something 4x? see stages
                    high_in_channels=in_channels[i],
                    out_channels=dec_channels[i]))


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.IN_FEATURES
        }
        ret["dec_layers"] = cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.DEC_LAYERS
        ret["dec_channels"] = cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.DEC_CHANNELS
        ret["layer_types"] = cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.LAYER_TYPES
        ret["drop_path_prob"] = cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.DROP_PATH_PROB 
        return ret


    def forward_features(self, features):
        out = []
        multi_scale_features = []

        x = self._in_norms[0](features[self.in_features[0]])

        for idx in range(self.num_stages - 1):
            x = self._stages[idx](x)
            out.append(x)
            x = self._resized_fuses[idx](
                lowres_x=x,
                highres_x=self._in_norms[idx+1](features[self.in_features[idx+1]]))

        x = self._stages[-1](x)
        out.append(x)
        multi_scale_features = out[:3] # OS32, 16, 8, they are used for kmax_transformer_decoder.
        panoptic_features = out[-1] # OS4/OS2, it is used for final mask prediction.
        # OS 32, 8, 4
        semantic_features = [features[self.in_features[0]], features[self.in_features[2]], features[self.in_features[3]]]
        return panoptic_features, semantic_features, multi_scale_features 

