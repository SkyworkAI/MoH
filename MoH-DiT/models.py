# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn
from torch.jit import Final
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class MoHAttention(nn.Module):
    fused_attn: Final[bool]
    LOAD_BALANCING_LOSSES = []

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            shared_head=0,
            routed_head=0,
            head_dim=None,
    ):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        
        if head_dim is None:
            self.head_dim = dim // num_heads
        else:
            self.head_dim = head_dim
        
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, (self.head_dim * self.num_heads) * 3, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        
        self.proj_drop = nn.Dropout(proj_drop)

        self.shared_head = shared_head
        self.routed_head = routed_head
        
        if self.routed_head > 0:
            self.wg = torch.nn.Linear(dim, num_heads - shared_head, bias=False)
            if self.shared_head > 0:
                self.wg_0 = torch.nn.Linear(dim, 2, bias=False)

        if self.shared_head > 1:
            self.wg_1 = torch.nn.Linear(dim, shared_head, bias=False)

    def forward(self, x):
        B, N, C = x.shape

        _x = x.reshape(B * N, C)
        
        if self.routed_head > 0:
            logits = self.wg(_x)
            gates = F.softmax(logits, dim=1)

            num_tokens, num_experts = gates.shape
            _, indices = torch.topk(gates, k=self.routed_head, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)

            if self.training:
                me = gates.mean(dim=0)
                ce = mask.float().mean(dim=0)
                l_aux = torch.mean(me * ce) * num_experts * num_experts

                MoHAttention.LOAD_BALANCING_LOSSES.append(l_aux)

            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates /= denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        if self.routed_head > 0:
            x = x.transpose(1, 2)

            if self.shared_head > 0:
                shared_head_weight = self.wg_1(_x)
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head

                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
        
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,1], routed_head_gates)
                
                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
            else:
                masked_gates = routed_head_gates

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)
        else:
            shared_head_weight = self.wg_1(_x)
            masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
            x = x.transpose(1, 2)

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 moh=False, shared_head=0, routed_head=0, head_dim=None,**block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            
        if moh:
            self.attn = MoHAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                     shared_head=shared_head, routed_head=routed_head, head_dim=head_dim, **block_kwargs)
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        moh=False,
        shared_head=[],
        routed_head=[],
        head_dim=[None],
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        if type(num_heads) != list:
            num_heads = [num_heads] * depth
            head_dim = [None] * depth
        
        if type(head_dim) != list:
            head_dim = [head_dim] * depth
            
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads[i], mlp_ratio=mlp_ratio,
                     moh=moh, shared_head=shared_head[i], routed_head=routed_head[i],
                     head_dim=head_dim[i]) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
def MoH_DiT_XL_2_90(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=[8,  8,  8,  8,  8,  8,  8,  8,  8,
                                                                    16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                                                                    24, 24, 24, 24, 24, 24, 24, 24, 24],
               moh=True, shared_head=[8,  8,  8,  8,  8,  8,  8,  8,  8,
                                      16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                                      16, 16, 16, 16, 16, 16, 16, 16, 16],
                         routed_head=[0,  0,  0,  0,  0,  0,  0,  0,  0,
                                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                      3,  3,  3,  3,  3,  3,  3,  3,  3],
                         head_dim=72,
                         **kwargs)


def MoH_DiT_XL_2_75(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=[8,  8,  8,  8,  8,  8,  8,  8,  8,
                                                                    16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                                                                    24, 24, 24, 24, 24, 24, 24, 24, 24],
               moh=True, shared_head=[8,  8,  8,  8,  8,  8,  8,  8,  8,
                                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                      14, 14, 14, 14, 14, 14, 14, 14, 14],
                         routed_head=[0,  0,  0,  0,  0,  0,  0,  0,  0,
                                      2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
                                      2,  2,  2,  2,  2,  2,  2,  2,  2],
                         head_dim=72,
                         **kwargs)


def MoH_DiT_L_2_90(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=[8,  8,  8,  8,  8,  8,  8,  8,
                                                                    16, 16, 16, 16, 16, 16, 16, 16,
                                                                    24, 24, 24, 24, 24, 24, 24, 24],
               moh=True, shared_head=[8,  8,  8,  8,  8,  8,  8,  8,
                                      16,  16,  16,  16,  16,  16,  16, 16,
                                      16, 16, 16, 16, 16, 16, 16, 16],
                         routed_head=[0,  0,  0,  0,  0,  0,  0,  0,
                                      0,  0,  0,  0,  0,  0,  0,  0,
                                      3,  3,  3,  3,  3,  3,  3,  3],
                         head_dim=64,
                         **kwargs)


def MoH_DiT_L_2_75(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=[8,  8,  8,  8,  8,  8,  8,  8,
                                                                    16, 16, 16, 16, 16, 16, 16, 16,
                                                                    24, 24, 24, 24, 24, 24, 24, 24],
               moh=True, shared_head=[8,  8,  8,  8,  8,  8,  8,  8,
                                      8,  8,  8,  8,  8,  8,  8,  8,
                                      12, 12, 12, 12, 12, 12, 12, 12],
                         routed_head=[0,  0,  0,  0,  0,  0,  0,  0,
                                      4,  4,  4,  4,  4,  4,  4,  4,
                                      4,  4,  4,  4,  4,  4,  4,  4],
                         head_dim=64,
                         **kwargs)


def MoH_DiT_B_2_90(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12,
               moh=True, shared_head=[6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 12, 12],
                         routed_head=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0],
                         **kwargs)


def MoH_DiT_B_2_75(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12,
               moh=True, shared_head=[4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 12, 12],
                         routed_head=[2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
                         **kwargs)


def MoH_DiT_S_2_90(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6,
               moh=True, shared_head=[3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6],
                         routed_head=[2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
                         **kwargs)


def MoH_DiT_S_2_75(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6,
               moh=True, shared_head=[2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6],
                         routed_head=[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
                         **kwargs)


DiT_models = {
    'MoH-DiT-XL/2-90': MoH_DiT_XL_2_90,  'MoH-DiT-XL/2-75': MoH_DiT_XL_2_75,
    'MoH-DiT-L/2-90':  MoH_DiT_L_2_90,   'MoH-DiT-L/2-75':  MoH_DiT_L_2_75,
    'MoH-DiT-B/2-90':  MoH_DiT_B_2_90,   'MoH-DiT-B/2-75':  MoH_DiT_B_2_75,
    'MoH-DiT-S/2-90':  MoH_DiT_S_2_90,   'MoH-DiT-S/2-75':  MoH_DiT_S_2_75,
}
