import os
import pytest
import torch
import triton.language as tl
import triton

@pytest.mark.parametrize('cond, opt_flag, env_var', [
    (cond, opt_flag, env_var) for cond in [True, False] \
                              for opt_flag in [True, False] \
                              for env_var in [True, False]\
])
@pytest.mark.forked
def test_assert(cond, opt_flag, env_var, device="cuda"):
    os.environ['TRITON_DEBUG'] = str(int(env_var))
    torch.zeros([1], dtype=torch.int32, device=device)

    @triton.jit(debug=opt_flag)
    def _kernel(COND: tl.constexpr):
        tl.device_assert(COND, 'test')

    if not cond and (opt_flag or env_var):
        with pytest.raises(RuntimeError):
            _kernel[(1, )](cond)
            torch.cuda.synchronize()
        return

    _kernel[(1, )](cond)
    torch.cuda.synchronize()
