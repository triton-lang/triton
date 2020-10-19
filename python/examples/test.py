import triton
import numpy
import torch
import itertools

torch.manual_seed(0)
numpy.random.seed(0)

def to_sparse(expr, data, layout, shape, block):
    # shape of result
    sparse = None
    shape_ret = []
    for i, d in enumerate(expr):
        if d.isupper() and sparse is None:
            sparse = i
            shape_ret.append(int(layout.sum()))
        if d.isupper():
            shape_ret.append(block[d])
        else:
            shape_ret.append(shape[i])
    # iterator
    steps = [block[d] if d.isupper() else 1 for d in expr]
    it = [range(0, shape[i], steps[i]) for i in range(len(expr))]
    # create result
    ret = torch.empty(*shape_ret, dtype=data.dtype, device=data.device)
    blockid = 0
    for curr in itertools.product(*it):
        if all([curr[i] == it[i][0] for i in range(len(curr)) if expr[i].isupper()]):
            blockid = 0
        data_slice = [slice(curr[i], curr[i] + steps[i], 1) for i in range(len(curr))]
        ret_slice = [slice(0, block[expr[i]], 1) if expr[i].isupper() else slice(curr[i], curr[i] + 1) for i in range(len(curr))]
        ret_slice.insert(sparse, blockid)
        blockid += 1
        ret[ret_slice] = data[data_slice]
    return ret

def test_expr(expr, shape, blocks):
    # decompose expr
    expr_a, expr_bc = expr.split(",")
    expr_b, expr_c  = expr_bc.split("->")
    # check with argument is sparse
    sparse_a = any(x.isupper() for x in expr_a)
    sparse_b = any(x.isupper() for x in expr_b)
    sparse_c = any(x.isupper() for x in expr_c)
    # allocate data
    shape_a = [shape[d.lower()] for d in expr_a]
    shape_b = [shape[d.lower()] for d in expr_b]
    shape_c = [shape[d.lower()] for d in expr_c]
    ref_a = torch.rand(*shape_a, device='cuda')
    ref_b = torch.rand(*shape_b, device='cuda')
    ref_c = torch.zeros(*shape_c, device='cuda')
    #ref_a[:] = 1
    #ref_b[:] = 1
    # layouts
    layout_a = [shape[d.lower()]//blocks[d] for d in expr_a if d.isupper()]
    layout_b = [shape[d.lower()]//blocks[d] for d in expr_b if d.isupper()]
    layout_c = [shape[d.lower()]//blocks[d] for d in expr_c if d.isupper()]
    layout_a = torch.randint(1, 2, layout_a, device='cuda')
    layout_b = torch.randint(1, 2, layout_b, device='cuda')
    layout_c = torch.randint(1, 2, layout_c, device='cuda')
    # triton computation
    triton_a = to_sparse(expr_a, ref_a, layout_a, shape_a, blocks) if sparse_a else ref_a
    triton_b = to_sparse(expr_b, ref_b, layout_b, shape_b, blocks) if sparse_b else ref_b
    triton_c = to_sparse(expr_c, ref_c, layout_c, shape_c, blocks) if sparse_c else ref_c
    triton.ops.einsum(expr, triton_a, triton_b, triton_c, layout_a, layout_b, layout_c, blocks)
    torch.cuda.synchronize()
    # reference computation
    ref_c = torch.einsum(expr.lower(), ref_a, ref_b)
    if sparse_c:
        ref_c = to_sparse(expr_c, ref_c, layout_c, shape_c, blocks)
    torch.cuda.synchronize()
    #print(ref_c)
    #print(triton_c)
    print((ref_c - triton_c).abs().max())




# shape characteristics
test_expr('bHMK,bhkn->bhmn', {'b': 2, 'h': 2, 'm': 64, 'k': 64, 'n': 64}, {'H': 1, 'M': 32, 'K': 32})
test_expr('bhmk,bHKN->bhmn', {'b': 2, 'h': 2, 'm': 64, 'k': 64, 'n': 64}, {'H': 1, 'K': 32, 'N': 32})
test_expr('bhmk,bhkn->bHMN', {'b': 2, 'h': 2, 'm': 64, 'k': 64, 'n': 64}, {'H': 1, 'M': 32, 'N': 32})

