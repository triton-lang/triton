# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
from utils.rotary_embedding import DeepseekScalingRotaryEmbedding
from utils.sglang_ref import extend_attention_fwd as extend_attention_fwd_ref
import argparse
import sys

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


is_hip_ = is_hip()


def input_helper(B, H, S_prefix, S_extend, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, dtype, device):
    q = torch.randn(B * S_extend, H, qk_nope_head_dim + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.randn(B * S_extend, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)

    k_buffer = torch.randn(B * S_prefix, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype)
    v_buffer = torch.randn(B * S_prefix, 1, kv_lora_rank, dtype=dtype)

    # interlancing [batch_start_off, batch_seq_len, batch_start_off, batch_seq_len, ...,]
    qo_indptr = torch.arange(B + 1, device=device) * S_extend
    kv_indptr = torch.arange(B + 1, device=device) * S_prefix  # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(B * (S_prefix), device=device)

    # o_extend = torch.empty(B * S_extend, H, kv_lora_rank, dtype=dtype, device=device)
    w_kc = torch.randn(H, kv_lora_rank, qk_nope_head_dim, dtype=dtype, device=device)
    w_vc = torch.randn(H, kv_lora_rank, v_head_dim, dtype=dtype, device=device)

    rotary_emb = DeepseekScalingRotaryEmbedding(
        qk_rope_head_dim,
        rotary_dim=qk_rope_head_dim,
        max_position_embeddings=16324,
        base=10,
        is_neox_style=True,
        scaling_factor=1.0,
        dtype=q.dtype,
        device=device,
    )

    positions = torch.tensor([S_extend], device=device).unsqueeze(0).repeat(B, 1)  # k positions and q position as last

    return q, kv_cache, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, rotary_emb, positions


def kv_b_proj(kv_a, w_kc, w_vc):
    kv_lora_rank = kv_a.shape[-1]
    qk_nope_head_dim = w_kc.shape[-1]
    v_head_dim = w_vc.shape[-1]
    num_heads = w_kc.shape[0]
    w = torch.cat((w_kc, w_vc), dim=-1).transpose(0, 1).reshape(kv_lora_rank,
                                                                num_heads * (qk_nope_head_dim + v_head_dim))
    return torch.matmul(kv_a, w).type_as(kv_a)


"""
kv_a.shape: torch.Size([2048, 512])
kv.shape: torch.Size([2048, 4096])
"""


def forward_normal_ref(q, latent_cache, k_buffer, v_buffer, o, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, H,
                       kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, rotary_emb, positions):
    _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    kv_a, _ = latent_cache.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    # projection
    # kv_a = self.kv_a_layernorm(kv_a.contiguous())
    kv = kv_b_proj(kv_a, w_kc, w_vc)
    kv = kv.view(-1, H, qk_nope_head_dim + v_head_dim)
    k_nope = kv[..., :qk_nope_head_dim]
    v = kv[..., qk_nope_head_dim:]
    k_pe = latent_cache[:, :, kv_lora_rank:]
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
    q[..., qk_nope_head_dim:] = q_pe
    k = torch.empty_like(q)
    k[..., :qk_nope_head_dim] = k_nope
    k[..., qk_nope_head_dim:] = k_pe

    latent_cache[:, :, :kv_lora_rank] = kv_a
    latent_cache[:, :, kv_lora_rank:] = k_pe

    extend_attention_fwd_ref(q, k, v, o, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, custom_mask=None,
                             mask_indptr=None, max_len_extend=qo_indptr[1])
    attn_output = o
    attn_output = attn_output.reshape(-1, H * v_head_dim)
    return attn_output


# forward_batch.extend_prefix_lens.sum() == 0 => forward_normal
def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    configs = []

    # prefill
    x_vals_list = [(args.B, 16, 0, 2048, 512, 128, 128, 64)]
    x_names = ["B", "H", "S_prefix", "S_extend", "kv_lora_rank", "qk_nope_head_dim", "v_head_dim", "qk_rope_head_dim"]
    line_vals = ["ref"]
    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, S_prefix, S_extend, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, sm_scale,
                  logit_cap, device, provider):
        warmup = 25
        rep = 100

        q, kv_cache, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, rotary_emb, positions = input_helper(
            B, H, S_prefix, S_extend, kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, dtype, device)

        o = torch.empty(qo_indptr[-1], H, v_head_dim, dtype=q.dtype, device=q.device)

        if "ref" in provider:
            fn = lambda: {
                forward_normal_ref(q, kv_cache, k_buffer, v_buffer, o, qo_indptr, kv_indptr, kv_indices, w_kc, w_vc, H,
                                   kv_lora_rank, qk_nope_head_dim, v_head_dim, qk_rope_head_dim, rotary_emb, positions)
            }

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=".", print_data=True, show_plots=False)


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-dtype", default='bf16', help="data type")
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-B", type=int, default=1)
    return parser.parse_args()


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def main():
    torch.manual_seed(0)
    args = parse_args()
    torch.set_default_device(args.device)
    benchmark(args)


if __name__ == '__main__':
    sys.exit(main())
