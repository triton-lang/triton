import numpy as np
import torch
import torch.nn as nn
import triton

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # linear layers
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        # initialize weights
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.xavier_normal_(self.fc.weight)
        # layer normalization
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, q, k, v, mask=None):
        # dimensions
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        # linear transformations
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # scaled dot-product attention        
        attn = triton.ops.einsum('blhk,bthk->hblt', q, k, [n_head, sz_b, len_q, len_k])
        attn = attn / np.sqrt(d_k)
        if mask is not None:
            attn = attn.masked_fill(mask[None], -np.inf)
        attn = torch.softmax(attn, dim=3)
        output = triton.ops.einsum('hblt,bthv->blhv', attn, v, [sz_b, len_q, n_head, d_v])
        output = output.view(sz_b, len_q, -1)
        output = self.fc(output)
        # epilogue
        output = self.layer_norm(output + residual)
        return output, attn