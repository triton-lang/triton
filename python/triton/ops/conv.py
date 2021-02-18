import torch
import triton
import os

class _conv(torch.autograd.Function):
    src = triton.read(os.path.join(os.path.dirname(__file__), 'conv.c'))
    kernel = dict()

    @staticmethod
    def unpack(IDX, CI, R, S):
        s = IDX % S
        cr = IDX // S
        r = cr % R
        ci = cr // R
        return ci, r, s

    @staticmethod
    def forward(ctx, a, b, pad, stride):
        # create kernel if necessary
        dtype = a.dtype
        device = a.device
        # shapes
        Z, CI, H, W = a.shape
        _, R, S, CO = b.shape
        P = (H + 2 * pad[0] - R) // stride[0] + 1
        Q = (W + 2 * pad[1] - S) // stride[1] + 1
        # compile kernel
        if (dtype, device) not in _conv.kernel:
            TK = 16
            defines = {
                'TYPE': dtype,
                'TM': 64,
                'TN': 64,
                'TK': TK,
                'TZ': 1,
                'HH': H,
                'WW': W,
                'PP': P,
                'QQ': Q,
                'SS': S,
                'RR': R,
            }
            idx = torch.arange(CI * R * S)
            ci, r, s = _conv.unpack(idx, CI, R, S)
            nci, nr, ns = _conv.unpack(idx + TK, CI, R, S)
            delta = (nci - ci) * a.stride(1) + (nr - r) * a.stride(2) + (ns - s) * a.stride(3)
            delta = delta.type(torch.int32).cuda()
            _conv.kernel[dtype] = (delta, triton.kernel(_conv.src, device=device, defines=defines))
        delta, kernel = _conv.kernel[dtype]
        # allocate output
        c = torch.empty([Z, CO, P, Q], dtype=dtype, device=device)
        # enqueue
        kernel(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            1.,
            Z * P * Q,
            CO,
            CI * R * S,
            pad[0],
            pad[1],
            stride[0],
            stride[1],
            delta.data_ptr(),
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(3),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(3),
            c.stride(0),
            c.stride(1),
            c.stride(2),
            c.stride(3),
            grid=lambda opt: [triton.cdiv(Z * P * Q, opt.TM), triton.cdiv(CO, opt.TN)])
        return c

conv = _conv.apply