import torch
from softmax_xent_new import triton_softmax_new
from softmax_xent_in_place import triton_softmax_in_place

from belt.torch_utils import CudaTimer

# WHICH SOFTMAX SHOULD WE TEST?
current_softmax = triton_softmax_new


def test_softmax(
    num_seq=16, n_vocab=32 * 1024, dtypes=[torch.float32, torch.float16], verbose=True
):
    did_it_work = True
    for dtype in dtypes:
        x = torch.randn(num_seq, n_vocab).to(dtype)
        indices = 4 + torch.ones(num_seq).long()

        triton_input = x.cuda()
        triton_indices = indices.cuda()
        triton_result = current_softmax(triton_input, triton_indices)
        if verbose:
            print("Triton:", triton_result)

        torch_input = x.cuda()
        torch_indices = indices.cuda()
        torch_xent_fn = torch.nn.CrossEntropyLoss(reduction="none")
        torch_result = torch_xent_fn(torch_input, torch_indices)
        if verbose:
            print("Torch:", torch_result)
            torch.testing.assert_allclose(torch_result, triton_result)
        did_it_work = did_it_work and torch.allclose(torch_result, triton_result)
    return did_it_work


def test_grad(
    num_seq=4, n_vocab=512, verbose=True, dtypes=[torch.float32, torch.float16]
):
    it_worked = True
    if verbose:
        print(num_seq, n_vocab)
    for dtype in dtypes:
        logit = torch.randn(num_seq, n_vocab, requires_grad=True, device="cuda").to(
            dtype
        )
        indices = torch.ones(num_seq).long()

        triton_logit = torch.nn.Parameter(
            logit.clone().detach().cuda(), requires_grad=True
        )
        triton_indices = indices.clone().detach().cuda()
        triton_result = current_softmax(triton_logit, triton_indices)
        triton_result.mean().backward()
        if verbose:
            print("Triton grad:\n", (triton_logit.grad[:, :6]))

        torch_logit = torch.nn.Parameter(
            logit.clone().detach().cuda(), requires_grad=True
        )
        torch_indices = indices.clone().detach().cuda()
        torch_xent_fn = torch.nn.CrossEntropyLoss(reduction="none")
        torch_result = torch_xent_fn(torch_logit, torch_indices)
        torch_result.mean().backward()

        print(
            f"Did the logits stay the same? {torch.all(torch.eq(logit, triton_logit))}"
        )
        if verbose:
            print("Torch grad:\n", (torch_logit.grad[:, :6]))
            torch.testing.assert_allclose(torch_logit.grad, triton_logit.grad)
        else:
            it_worked = it_worked and torch.allclose(
                torch_logit.grad, triton_logit.grad
            )
            if not it_worked:
                print((torch_logit.grad - triton_logit.grad)[:3, :9])
    return it_worked


def test_many_settings(
    seq_settings=[8, 7, 64, 256, 512, 16 * 1024],  # FAIL sometimes 2048, 16 * 1024],
    vocab_settings=[128, 512, 1024, 2048, 8192, 100 * 512],  # 3*512, 99*512 FAILS
    verbose=True,
    test_backwards=False,
    dtypes=[torch.float32],  # torch.float16],
):
    if test_backwards:
        print("Testing gradients...")
        test_fn = test_grad
    else:
        print("Testing forward pass results...")
        test_fn = test_softmax
    print(f"Testing logit datatypes {dtypes}.")
    results = {}
    for n_vocab in vocab_settings:
        for num_seq in seq_settings:
            for dtype in dtypes:
                try:
                    did_it_work = test_fn(
                        num_seq=num_seq, n_vocab=n_vocab, verbose=False, dtypes=[dtype]
                    )
                except Exception as e:
                    did_it_work = False
                    print(e)
                results[(num_seq, n_vocab, dtype)] = did_it_work
                if verbose:
                    print(
                        f"Working with batch size {num_seq}, n_vocab {n_vocab}, dtype {dtype}? {did_it_work}"
                    )
    print(results)
    return results


def time_softmax(repeat=3, num_seq=16 * 512):
    dtype = torch.float32
    x = torch.randn(num_seq, 51200, requires_grad=True).to(dtype)
    indices = torch.arange(num_seq).long()
    triton_input = x.cuda()
    triton_indices = indices.cuda()
    triton_result = current_softmax(triton_input, triton_indices)
    triton_result.mean().backward()

    with CudaTimer() as timer:
        for _ in range(repeat):
            triton_result = current_softmax(triton_input, triton_indices)
            triton_result.mean().backward()
    print("Triton", timer.elapsed_seconds() / repeat)

    torch_input = x.cuda()
    torch_indices = indices.cuda()
    torch_xent_fn = torch.nn.CrossEntropyLoss(reduction="none")
    torch_result = torch_xent_fn(torch_input, torch_indices)
    torch_result.mean().backward()

    with CudaTimer() as timer:
        for _ in range(repeat):
            torch_result = torch_xent_fn(torch_input, torch_indices)
            torch_result.mean().backward()
    print("Torch", timer.elapsed_seconds() / repeat)


def repr_weird_repeat(num_seq=32 * 512, n_vocab=1024):
    dtype = torch.float32
    x = torch.randn(num_seq, n_vocab, requires_grad=True).to(dtype)
    indices = torch.ones(num_seq).long()
    triton_input = x.cuda()
    triton_indices = indices.cuda()
    triton_result = current_softmax(triton_input, triton_indices)
    print(triton_result.mean())
    print("Zeros or Negatives?", len([z for z in triton_result if z < 0.1]))
    triton_result.mean().backward()

    y = torch.randn(num_seq, n_vocab, requires_grad=True).to(dtype).cuda()
    triton_result2 = current_softmax(y, triton_indices)
    print(triton_result2.mean())
    print("Zeros or Negatives?", len([z for z in triton_result2 if z < 0.1]))

    z = torch.randn(num_seq, n_vocab, requires_grad=True).to(dtype).cuda()
    triton_result3 = current_softmax(z, triton_indices)
    print(triton_result3.mean())
    print("Zeros or Negatives?", len([z for z in triton_result3 if z < 0.1]))


if __name__ == "__main__":
    test_many_settings()
