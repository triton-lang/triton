import torch
import triton

class _add(torch.autograd.Function):
    src = """
__global__ void add(float* z, float* x, float* y, int N) {

    int pid = get_program_id(0);

    int offset[TILE] = pid * TILE + 0 ... TILE;
    float* pz[TILE]  = z + offset;
    float* px[TILE]  = x + offset;
    float* py[TILE]  = y + offset;

    bool check[TILE] = offset < N;

    *?(check)pz = *?(check)px + *?(check)py;
}
    """

    kernel = triton.kernel(src, defines={'TILE': 1024}, num_warps=[4])

    @staticmethod
    def forward(ctx, x, y):
       z = torch.empty_like(x).cuda()

       N = x.numel()
       grid = lambda opt: (triton.cdiv(N, opt.d('TILE')),)

       _add.kernel(z,x,y, N, grid=grid)

       return z

add = _add.apply

# test
torch.manual_seed(0)
x = torch.rand(98432).cuda()
y = torch.rand(98432).cuda()
za = x + y
zb = add(x, y)

print(torch.allclose(za,zb))
