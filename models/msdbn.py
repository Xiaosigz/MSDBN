#第一阶段根据阈值粗粒度筛选，第二阶段根据fc细粒度筛选，进行多尺度融合,此外还输出了全局信息，全局以及局部信息拼接完成分类，然后损失使用KL散度。
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import random
import resnet1
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]#4
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)#768/4 = 192
        self.all_head_size = self.num_attention_heads * self.attention_head_size #

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
def channel_shuffle(x):
    batchsize, groups,num_channels, height = x.data.size()

#     channels_per_group = num_channels // groups

#     # reshape
#     x = x.view(batchsize, groups, 
#         channels_per_group, height)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height)

    return x    
class FF_Attention(nn.Module):
    def __init__(self, config):
        super(FF_Attention, self).__init__()
        self.num_attention_heads = 2#4
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)#768/4 = 192
        self.all_head_size = self.num_attention_heads * self.attention_head_size #

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    def transpose_for_scores1(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states):
#         shuffle_input = self.transpose_for_scores1(hidden_states)
#         hidden_states = channel_shuffle(shuffle_input).permute(0,2,1)
#         print(hidden_states.shape)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)#4,3,785,256
#         context_layer = context_layer.permute(0,1,3,2)
#         context_layer = channel_shuffle(context_layer, 3)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         context_layer = context_layer.permute(0, 3, 1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
# class FF_Attention(nn.Module):
#     def __init__(self, config):
#         super(FF_Attention, self).__init__()
#         self.num_attention_heads = 3#4
#         self.attention_head_size = int(config.hidden_size*3 / self.num_attention_heads)#768/4 = 192
#         self.all_head_size = self.num_attention_heads * self.attention_head_size #

#         self.query = Linear(config.hidden_size*3, self.all_head_size)
#         self.key = Linear(config.hidden_size*3, self.all_head_size)
#         self.value = Linear(config.hidden_size*3, self.all_head_size)

#         self.out = Linear(config.hidden_size*3, config.hidden_size*3)
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

#         self.softmax = Softmax(dim=-1)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
#     def transpose_for_scores1(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 3, 1)

#     def forward(self, hidden_states):
# #         shuffle_input = self.transpose_for_scores1(hidden_states)
# #         hidden_states = channel_shuffle(shuffle_input).permute(0,2,1)
# #         print(hidden_states.shape)
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_probs = self.softmax(attention_scores)
#         weights = attention_probs
#         attention_probs = self.attn_dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)#4,3,785,256
# #         context_layer = context_layer.permute(0,1,3,2)
# #         context_layer = channel_shuffle(context_layer, 3)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
# #         context_layer = context_layer.permute(0, 3, 1, 2).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         attention_output = self.out(context_layer)
#         attention_output = self.proj_dropout(attention_output)
#         return attention_output, weights
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class FF_Mlp(nn.Module):
    def __init__(self, config):
        super(FF_Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size*3, config.transformer["mlp_dim"]*3)
        self.fc2 = Linear(config.transformer["mlp_dim"]*3, config.hidden_size*3)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class FF_Block(nn.Module):
    def __init__(self, config):
        super(FF_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = FF_Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

# class Part_Attention(nn.Module):
#     def __init__(self):
#         super(Part_Attention, self).__init__()

#     def forward(self, x):
#         length = len(x)
#         last_map = x[0]
#         for i in range(1, length):
#             last_map = torch.matmul(x[i], last_map)
#         last_map = last_map[:,:,0,1:]

#         _, max_inx = last_map.max(2)
#         return _, max_inx

# class Encoder(nn.Module):
#     def __init__(self, config):
#         super(Encoder, self).__init__()
#         self.layer = nn.ModuleList()
#         for _ in range(config.transformer["num_layers"] - 1):
#             layer = Block(config)
#             self.layer.append(copy.deepcopy(layer))
#         self.part_select = Part_Attention()
#         self.part_layer = Block(config)
#         self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)

#     def forward(self, hidden_states):
#         attn_weights = []
#         for layer in self.layer:
#             hidden_states, weights = layer(hidden_states)
#             attn_weights.append(weights)            
#         part_num, part_inx = self.part_select(attn_weights)
#         part_inx = part_inx + 1
#         parts = []
#         B, num = part_inx.shape
#         for i in range(B):
#             parts.append(hidden_states[i, part_inx[i,:]])
#         parts = torch.stack(parts).squeeze(1)
#         concat = torch.cat((hidden_states[:,0].unsqueeze(1), parts), dim=1)
#         part_states, part_weights = self.part_layer(concat)
#         part_encoded = self.part_norm(part_states)   

#         return part_encoded
# class SE_Mlp(nn.Module):
#     def __init__(self, config):
#         super(SE_Mlp, self).__init__()
#         self.fc0 = nn.Linear(768*3, 768)
#         self.relu0 = nn.ReLU()
#         self.fc1 = nn.Linear(768, 768*3)  # 从 c -> c/r
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(768*3, 768) # 从 c/r -> c
#         self.relu2 = nn.ReLU()
#         self.dropout = nn.Dropout(config.transformer["dropout_rate"])
#         self.sigmoid = nn.Sigmoid()
        
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.fc0.weight)
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
# #         nn.init.xavier_uniform_(self.fc3.weight)
#         nn.init.normal_(self.fc0.bias, std=1e-6)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#         nn.init.normal_(self.fc2.bias, std=1e-6)
# #         nn.init.normal_(self.fc3.bias, std=1e-6)
#     def forward(self, x):
#         x_orig = self.fc0(x)
#         x = self.fc1(x_orig)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
# #         x = self.fc3(x)
# #         x = self.relu3(x)
#         x = self.dropout(x)
#         sig = self.sigmoid(x)
#         return sig*x_orig

# def cosine_similarity(x, y):
#     x = x.detach().cpu().numpy()
#     y = y.detach().cpu().numpy()
#     score = np.dot(x, y.T)
#     score = np.diag(score)
#     # print(score)
#     score_under = []
#     for i in range(len(score)):
#         result_face = sum([c * c for c in x[i][:]])
#         result_background = sum([d * d for d in y[i][:]])
#         score_under.append((result_face * result_background)**0.5)
#     out = score/score_under
#     return out

class ViT_Branch(nn.Module):#方案二
    def __init__(self,
                 in_channel: int,
                 num_classes: int,
                 num_selects: int):
        super().__init__()
        self.fc=nn.Linear(in_channel, num_classes)
        self.num_selects=num_selects
        self.softmax=Softmax(dim=-1)
        self.avepool=nn.AvgPool2d(2, stride=2)
        self.unfold1=nn.Unfold(kernel_size=(8, 8), stride=3,padding=2)
        self.fold1=nn.Fold(output_size=(8, 8), kernel_size=(8, 8), stride=8)
        self.unfold2=nn.Unfold(kernel_size=(6, 6), stride=3,padding=1)
        self.fold2=nn.Fold(output_size=(6, 6), kernel_size=(6, 6), stride=6)
        self.unfold3=nn.Unfold(kernel_size=(4, 4), stride=3)
        self.fold3=nn.Fold(output_size=(4, 4), kernel_size=(4, 4), stride=4)
        self.imageunfold = nn.Unfold(kernel_size=(128, 128), stride=48, padding=32)
        self.imagefold = nn.Fold(output_size=(128, 128),kernel_size=(128, 128), stride=128)
        self.fc_l = nn.Linear(768, 192)
        self.fc_m = nn.Linear(768, 192)
        self.fc_s = nn.Linear(768, 384)

    def forward(self, x,mask,img):
        bs = x.shape[0]
        image_patches = self.imageunfold(img)
        image_patches = image_patches.permute(0,2,1)
        patches_l = self.unfold1(x.permute(0,2,1).reshape(bs,768,28,28))
        patches_l = patches_l.permute(0,2,1)
        patches_m = self.unfold2(x.permute(0,2,1).reshape(bs,768,28,28))
        patches_m = patches_m.permute(0,2,1)
        patches_s = self.unfold3(x.permute(0,2,1).reshape(bs,768,28,28))
        patches_s = patches_s.permute(0,2,1)
        prob_class =self.softmax(self.fc(x)) 
        max_logits,max_ids=prob_class.max(dim=-1)#4,784
        max_logits = max_logits.reshape(bs,28,28).unsqueeze(1)#4,1,28,28
        part_logits = self.unfold1(max_logits)#4,49,64
#         print(part_logits.shape)#3,4,4
        part_logits = part_logits.mean(dim=1)#4,64
#         print(part_logits.shape)
        part_logits = torch.mul(mask,part_logits)
        _, part_ids=part_logits.sort(-1, descending=True)
        selection=part_ids[:, :1]
        parts_l = []
        parts_m = []
        parts_s = []
        image_parts = []
        
        for i in range(bs):
#             print(patches_l[i, selection[i, :]].shape)
            parts_l.append(patches_l[i, selection[i, :]])  # 获取这些更好表达的注意力向量
            parts_m.append(patches_m[i, selection[i, :]])
            parts_s.append(patches_s[i, selection[i, :]])
            image_parts.append(image_patches[i, selection[i,:]])
        image_parts=torch.stack(image_parts).permute(0,2,1)
        parts_l=torch.stack(parts_l).permute(0,2,1) #4,768*64,1
        parts_m=torch.stack(parts_m).permute(0,2,1) #4,768*36,1
#         print(parts_m.shape)
        parts_s=torch.stack(parts_s).permute(0,2,1) #4,768*16,1
        outl = self.fold1(parts_l).permute(0,2,3,1).reshape(bs,64,768)
        outm = self.fold2(parts_m)# bs,768,6,6
        outm = torch.nn.functional.interpolate(outm,size=[8,8],mode='bilinear',align_corners=False)#4,768,8,8
        outm = outm.permute(0,2,3,1).reshape(bs,64,768)
        outs = self.fold3(parts_s)
        outs = torch.nn.functional.interpolate(outs,size=[8,8],mode='bilinear',align_corners=False)
        outs = outs.permute(0,2,3,1).reshape(bs,64,768)
        part1 = self.fc_l(outl)
        part2 = self.fc_m(outm)
        part3 = self.fc_s(outs)
        cat1 = torch.cat((part1,part2),dim=2)
        ff_cat = torch.cat((cat1,part3),dim=2)
        out_image = self.imagefold(image_parts)
        image_part = torch.nn.functional.interpolate(out_image,size=[224,224],mode='bilinear',align_corners=False)
        return ff_cat,image_part


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
#         self.part_norm = LayerNorm(config.hidden_size*3, eps=1e-6)
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.encoder_norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.tokenunfold=nn.Unfold(kernel_size=(8, 8), stride=3,padding=2)
#         self.tokenfold=nn.Fold(output_size=(8, 8*64), kernel_size=(7, 7), stride=7)
        self.part_extractor =ViT_Branch(768,14,1)
        self.part_layer = Block(config)
    def forward(self, hidden_states,img):
        attn_weights = []
        hidden_list = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
            hidden_list.append(hidden_states)
        B = hidden_states.shape[0]
#         print(hidden_states[:,1:].shape)
        input_token = hidden_states[:,1:].permute(0,2,1)
        unfoldtoken = self.tokenunfold(input_token.reshape(B,768,28,28)).permute(0,2,1)#4,784,768 - 4,49,768*16
        whead_mean = attn_weights[11].mean(dim=1)#12个head取平均 bs，785，785
        B = whead_mean.shape[0]
        whead_mean = whead_mean[:,0,1:].unsqueeze(1).reshape(B,1,28,28)  #bs,784,784
        bigtoken_w = self.tokenunfold(whead_mean).mean(dim=1) #bs,16,49- bs,49
        threshold = 100/785
        threshold_m = torch.full_like(bigtoken_w , threshold)
        big_m = torch.ge(bigtoken_w,threshold_m)# 4,64
        
#         big_m  = big_m.expand(unfoldtoken.shape[0],unfoldtoken.shape[1],unfoldtoken.shape[-1])
# #         print(big_m.shape)
# #         print(unfoldtoken.shape)
#         pick_tokens = torch.mul(big_m,unfoldtoken)
#         print(pick_tokens.shape)
#         foldpick_tokens = self.tokenfold(pick_tokens.permute(0,2,1)).reshape(B,768,3136).permute(B,3136,768)
#         reshape(B,768,56,56)
#         print(foldpick_tokens.shape)
#         .reshape(B,676,768)#4,784,768
        part1,img_part = self.part_extractor(hidden_states[:,1:],big_m,img)
        last_input = torch.cat((hidden_states[:,0].unsqueeze(1),part1),dim=1)
        last_output,_ = self.part_layer(last_input)
        vit_output = self.encoder_norm(last_output)
#         encoded_gobal = self.encoder_norm2(hidden_states)
        return vit_output,img_part

# class Transformer(nn.Module):
#     def __init__(self, config, img_size):
#         super(Transformer, self).__init__()
#         self.embeddings = Embeddings(config, img_size=img_size)
#         self.encoder = Encoder(config)

#     def forward(self, input_ids):
#         embedding_output = self.embeddings(input_ids)
#         part_encoded = self.encoder(embedding_output)
#         return part_encoded


class HLF(nn.Module):
    def __init__(self):
        super(HLF, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * 4, 14)
    def forward(self, x_list):
        x1 = x_list[0]# 4,2048,7,7
        x2 = x_list[1]
        x3 = x_list[2]
        x1 = x1.reshape(x1.shape[0],2048,49)
        x2 = x2.reshape(x2.shape[0],2048,49)
        x3 = x3.reshape(x3.shape[0],2048,49)
        x31 = torch.matmul(x3, x1.transpose(-1, -2))# 4,2048,2048
        x32 = torch.matmul(x3, x2.transpose(-1, -2))# 4,2048,2048
        x31_d = torch.diagonal(x31, dim1=-2, dim2=-1).unsqueeze(2) # 4,2048
        x32_d = torch.diagonal(x32, dim1=-2, dim2=-1).unsqueeze(2) # 4,2048
        x31_d = self.softmax(x31_d)
        x32_d = self.softmax(x32_d)
        h_x1 = x1 + x1 * x31_d.expand_as(x1)+x2 * x32_d.expand_as(x2)# 4,2028,49
#         print(h_x1.shape)
        h_x1 = h_x1.reshape(h_x1.shape[0],2048,7,7)
        h_x1  = self.avgpool(h_x1)# 4,2048,1,1
#         print(h_x1.shape)
        h_x1 = h_x1.view(h_x1.size(0), -1)
        h_x1 = nn.Dropout(p=0.5)(h_x1)
        ff_features = h_x1
        h_x1= self.fc(h_x1)
        return h_x1,ff_features
class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded_vit,img_part = self.encoder(embedding_output,input_ids)
        return encoded_vit, img_part

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=22, smoothing_value=0, zero_head=False):
        super(VisionTransformer, self).__init__()
#         self.hlf = HLF()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        self.head1 = Linear(config.hidden_size, num_classes)
#         self.head2 = Linear(config.hidden_size, num_classes)
        self.head3 = Linear(config.hidden_size+2048, num_classes)
        self.pretrained_model =  resnet1.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 14)
#         self.head = Linear(768*768, num_classes)
#         part_tokens = self.transformer(x)
#         part_logits = self.part_head(part_tokens[:, 0])torch
#         input_tokens = self.tokenizer(x) 

    def forward(self, x, labels=None):
        vit_output,img_part  = self.transformer(x)
        logits2, cnn_output = self.pretrained_model (img_part)
#         _,_,_,x_list2 = self.pretrained_model2(img_part2)
#         logits4, ff_feature1 = self.hlf(x_list1)
#         logits2, ff_feature = self.hlf(x_list)
        ff_output = torch.cat((vit_output[:,0],cnn_output),dim=1)
#         x_patch = all_tokens[:,1:]#除去class_token的其余token
#         b = all_tokens.size(0) 
# #         x_bilinear=(torch.bmm(x_patch, torch.transpose(x_patch, 1, 2))/768).view(b,-1)
#         x_bilinear=(torch.bmm(torch.transpose(x_patch, 1, 2), x_patch)/196).view(b,-1)
#         x_bilinear = torch.nn.functional.normalize(torch.sign(x_bilinear) * torch.sqrt(torch.abs(x_bilinear) + 1e-10))
#         # 进行均值池化
#         logits = self.head(x_bilinear)
        logits1 = self.head1(vit_output[:,0])#局部
#         logits2 = self.head2(all_tokens_global[:,0])#全局
        logits3 = self.head3( ff_output)
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            ce_loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            contrast_loss = con_loss(all_tokens[:, 0], labels.view(-1))
            loss = ce_loss + contrast_loss
            return loss, logits
        else:
            return logits1,logits2,logits3

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
#             self.transformer.encoder.encoder_norm2.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
#             self.transformer.encoder.encoder_norm2.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname) 
def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
