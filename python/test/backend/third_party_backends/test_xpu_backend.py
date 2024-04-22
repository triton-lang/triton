import torch


def test_xpu_backend(cmdopt):
    if cmdopt == "xpu":
        has_ipex = False
        try:
            # Import IPEX to provide Intel GPU runtime
            import intel_extension_for_pytorch  # type: ignore # noqa: F401
            has_ipex = True if hasattr(torch, "xpu") else False
        except Exception:
            has_ipex = False

        import triton
        import triton.language as tl

        @triton.jit()
        def kernel(x_ptr, y_ptr, out_ptr):
            pid = tl.program_id(axis=0)
            x = tl.load(x_ptr + pid)
            y = tl.load(y_ptr + pid)
            out = x + y
            tl.store(out_ptr + pid, out)

        if has_ipex:
            for _ in range(1000):
                x = torch.randn((65536, ), device="xpu", dtype=torch.float32)
                y = torch.randn((65536, ), device="xpu", dtype=torch.float32)
                z = torch.zeros((65536, ), device="xpu", dtype=torch.float32)
                kernel[(65536, )](x, y, z, num_warps=32)
                assert torch.all(x + y == z)
    else:
        return
