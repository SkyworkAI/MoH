import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    LOAD_BALANCING_LOSSES = []
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., shared_head=0, routed_head=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.
        # Generate sequnce length scale
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(input_resolution[0] * input_resolution[1])),
                             persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        self.shared_head = shared_head
        self.routed_head = routed_head
        if self.routed_head > 0:
            self.wg = torch.nn.Linear(dim, num_heads - shared_head, bias=False)
            if self.shared_head > 0:
                self.wg_0 = torch.nn.Linear(dim, 2, bias=False)

        if self.shared_head > 1:
            self.wg_1 = torch.nn.Linear(dim, shared_head, bias=False)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
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

                Attention.LOAD_BALANCING_LOSSES.append(l_aux)

            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates /= denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head

        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        # Use MLP to generate continuous relative positional bias
        rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                   relative_pos_index.view(-1)].view(-1, N, N)

        # Calculate attention map using sequence length scaled cosine attention and query embedding
        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * self.seq_length_scale) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.routed_head > 0:
            x = (attn @ v).transpose(1, 2)  # B, N, head, dim

            if self.shared_head > 1:
                shared_head_weight = self.wg_1(_x)
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
            else:
                shared_head_gates = torch.ones((B, N, self.shared_head)).to(_x.device).to(_x.dtype) * self.shared_head
            
            if self.shared_head == 0:
                masked_gates = routed_head_gates
            else:
                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
                
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,1], routed_head_gates)

                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, C)
        else:
            shared_head_weight = self.wg_1(_x)
            masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
                
            x = (attn @ v).transpose(1, 2)  # B, N, head, dim
            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x