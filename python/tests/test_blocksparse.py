import itertools
import torch
import triton as tt
import pytest

def sparsify_tensor(x, mask, block):
  ret = torch.empty((x.size(0), mask.sum(), block, block), dtype=x.dtype, device=x.device)
  for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
    ret[:, idx, :, :] = x[:, h, i*block: (i+1)*block, j*block: (j+1)*block]
  return ret

def mask_tensor(x, mask, block, value = 0):
  ret = x.clone()
  for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
    ret[:, h, i*block: (i+1)*block, j*block: (j+1)*block] = value
  return ret

@pytest.mark.parametrize("MODE, TRANS_A, TRANS_B, BLOCK", 
    [
    (mode, at, bt, block) for mode in ['sdd', 'dsd', 'dds']\
                          for at   in [False, True]\
                          for bt   in [False, True]\
                          for block in [16, 32, 64]
    ]
)
def test_op(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE = torch.float16, Z = 3, H = 2, M = 128, N = 256, K = 384):
  # set seed
  torch.random.manual_seed(0)
  # create inputs
  a = torch.randn((Z, H, K, M) if TRANS_A else (Z, H, M, K), dtype=DTYPE, device='cuda')
  b = torch.randn((Z, H, N, K) if TRANS_B else (Z, H, K, N), dtype=DTYPE, device='cuda')
  shape = {'sdd': (M, N), 'dsd': (a.shape[2], a.shape[3]), 'dds': (b.shape[2], b.shape[3])}[MODE]
  layout = torch.randint(2, (H, shape[0]//BLOCK, shape[1]//BLOCK))
  # triton result
  op = tt.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B)
  ra = sparsify_tensor(a, layout, BLOCK) if MODE == 'dsd' else a
  rb = sparsify_tensor(b, layout, BLOCK) if MODE == 'dds' else b
  rc  = op(ra, rb)
  # torch result
  ta = mask_tensor(a, layout, BLOCK) if MODE == 'dsd' else a
  tb = mask_tensor(b, layout, BLOCK) if MODE == 'dds' else b
  ta = ta.transpose(2, 3) if TRANS_A else ta
  tb = tb.transpose(2, 3) if TRANS_B else tb
  tc = torch.matmul(ta, tb)
  tc = mask_tensor(tc, layout, BLOCK) if MODE == 'sdd' else tc
  tc = sparsify_tensor(tc, layout, BLOCK) if MODE == 'sdd' else tc
  # compare
  rtol, atol = {torch.float32: (1e-4, 1e-5),
                torch.float16: (1e-2, 1e-3)}[DTYPE]
  assert torch.allclose(rc, tc, rtol=rtol, atol=atol)
