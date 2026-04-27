import triton


def test_fixed_n_warmup_and_n_repeat(device):
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1

    triton.testing.do_bench(fn, n_warmup=3, n_repeat=7)
    # 1 initial call (JIT warmup) + 3 warmup + 7 repeat
    assert call_count == 1 + 3 + 7
