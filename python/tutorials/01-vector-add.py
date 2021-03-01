import torch
import triton

# source-code for Triton compute kernel
# here we just copy-paste the above code without the extensive comments.
# you may prefer to store it in a .c file and load it from there instead.
_src = """
__global__ void add(float* z, float* x, float* y, int N){
    // program id
    int pid = get_program_id(0);
    // create arrays of pointers
    int offset[BLOCK] = pid * BLOCK + 0 ... BLOCK;
    float* pz[BLOCK] = z + offset;
    float* px[BLOCK] = x + offset;
    float* py[BLOCK] = y + offset;
    // bounds checking
    bool check[BLOCK] = offset < N;
    // write-back
    *?(check)pz = *?(check)px + *?(check)py;
}
    """
# This function returns a callable `triton.kernel` object
# created from the above source code.
# For portability, we maintain a cache of kernels for different `torch.device`
# We compile the kernel with -DBLOCK=1024
_kernels = dict()

def make_add_kernel(device):
    if device not in _kernels:
        defines = {'BLOCK': 1024}
        autotune_vals = [({'BLOCK': '1024'}, 4), ({'BLOCK': '2048'}, 4)]
        autotune_key = ["N"]
        _kernels[device] = triton.kernel(_src, device=device, defines=defines, autotune_vals=autotune_vals,
                                         autotune_key=autotune_key)
    return _kernels[device]

# This is a standard torch custom autograd Function
# The only difference is that we can now use the above kernel
# in the `forward` and `backward` functions.`
class _add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        # constraints of the op
        assert x.dtype == torch.float32
        # *allocate output*
        z = torch.empty_like(x)
        # *create launch grid*:
        # this is a function which takes compilation parameters `opt`
        # as input and returns a tuple of int (i.e., launch grid) for the kernel.
        # triton.cdiv is a shortcut for ceil division:
        # triton.cdiv(a, b) = (a + b - 1) // b
        N = z.shape[0]
        grid = lambda opt: (triton.cdiv(N, opt.BLOCK), )
        # *launch kernel*:
        # pointer to the data of torch tensors can be retrieved with
        # the `.data_ptr()` method
        kernel = make_add_kernel(z.device)
        kernel(z.data_ptr(), x.data_ptr(), y.data_ptr(), N, grid=grid)
        return z

# Just like we standard PyTorch ops
# We use the `.apply` method to create a
# callable object for our function
add = _add.apply

torch.manual_seed(0)
x = torch.rand(98432, device='cuda')
y = torch.rand(98432, device='cuda')
za = x + y
zb = add(x, y)
print(za)
print(zb)
print(f'The maximum difference between torch and triton is ' f'{torch.max(torch.abs(za - zb))}')
