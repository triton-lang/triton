import triton
import triton.language as tl
import torch

@triton.jit
def atomic(lock):
  while tl.atomic_cas(lock, 0, 1) == 1:
    pass

@triton.jit
def generic_while(lb, value):
  c = -1
  while c <= 0:
    c += 1

locks = torch.zeros(32, dtype=torch.int32, device='cuda')
mod_atomic, ctx_atomic = atomic.compile_to_ttir(locks, grid=(1,))
mod_atomic.dump()

mod_generic_while, ctx_generic_while = generic_while.compile_to_ttir(8, 9, grid=(1,))
mod_generic_while.dump()
