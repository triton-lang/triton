import torch
import triton


def linear(x, w, bias = None):
    print(x.size(), w.size())
    m, k = x.size()
    k, n = w.size()
    out = torch.empty([m, n], device=x.device)
    triton.ops.einsum('mk,nk->mn', x, w, bias)
    if bias is not None:
      out += bias
    return out