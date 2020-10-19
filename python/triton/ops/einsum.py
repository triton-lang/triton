from math import ceil, log2
from enum import IntEnum
from functools import reduce
from operator import mul
from collections import OrderedDict
from collections import namedtuple
import re
import string
import triton
# torch
import torch
# numpy -- ideally removed in a future release
import numpy as np
# sympy -- ideally removed in a future release
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.ccode import C89CodePrinter


class _einsum(torch.autograd.Function):


    #############################
    ## Triton-C code generation
    #############################
    def print_cc(expr, axes_0, axes_1, axes_2, prefix):
        if expr in axes_0:
            return f'{prefix}r{expr}[:, newaxis, newaxis]'
        if expr in axes_1:
            return f'{prefix}r{expr}[newaxis, :, newaxis]'
        if expr in axes_2:
            return f'{prefix}r{expr}[newaxis, newaxis, :]'
        return expr

    def unpack_cc(tile, axes, prefix, remat):
        ret = ''
        axes = list(map(str, axes))
        for i, d in enumerate(reversed(axes)):
            if i == len(axes) - 1:
                break
            currs = ''.join(axes[: len(axes) - i])
            nexts = ''.join(axes[: len(axes) - (i + 1)])
            ty = '' if remat else 'int '
            sz = '' if remat or tile is None else f'[{tile}]'
            ret += f'    {ty}{prefix}{nexts}{sz} = r{currs} / dim_{d};\n'
            ret += f'    {ty}{prefix}{d}{sz} = r{currs} % dim_{d};\n'
        return ret

    def strides_cc(name, expr):
        ret = [f'stride_{name}_{d}' for d in expr[:-1]] + ['1']
        ret = dict(zip(expr, ret))
        return ret

    def make_kernel(name, dtype, 
                    expr_a, expr_b, expr_c,
                    sparse_a, sparse_b, sparse_c,
                    axes_m, axes_n, axes_k, axes_b,
                    multipleof_a, multipleof_b, multipleof_c,
                    stride_a_last, stride_b_last, stride_c_last,
                    lut_mode_a, lut_mode_b,
                    delta_a, delta_b,
                    blocks):

        use_lut_a = True
        use_lut_b = True

        outer_sparse_a = [x for x in expr_a if x in sparse_a and x not in axes_k] 
        outer_dense_a = [x for x in expr_a if x not in sparse_a and x not in axes_k] 
        outer_sparse_b = [x for x in expr_b if x in sparse_b and x not in axes_k]
        outer_dense_b = [x for x in expr_b if x not in sparse_b and x not in axes_k] 
        outer_dense_c = [x for x in expr_c if x not in sparse_c and x not in axes_k] 


        src = ""

        if use_lut_a and lut_mode_a == _einsum.LUT_MODE.CONSTANT:
            src += f"""
char __constant__* AD = calloc({4*len(delta_a)});"""
        if use_lut_b and lut_mode_b == _einsum.LUT_MODE.CONSTANT:
            src += f"""
char __constant__* BD = calloc({4*len(delta_b)});"""


        src += f"""
__global__ void {name}(
              TYPE * A __noalias __readonly __aligned(16)
            , TYPE * B __noalias __readonly __aligned(16)
            , TYPE * C
            , int * locks
            , float alpha
            , int matmul_m, int matmul_n, int matmul_k __multipleof(16)
            , int div_m
            """
        for dim in [axes_m, axes_n, axes_k, axes_b]:
            for d in dim:
                src += f", int dim_{d}"
        src += "\n            "
        for dim, name, mult, sparse in zip([expr_a, expr_b, expr_c],
                                         ['a', 'b', 'c'],
                                         [multipleof_a, multipleof_b, multipleof_c],
                                         [sparse_a, sparse_b, sparse_c]):
            for d in range(len(dim) - 1):
                if sparse and dim[d] == sparse[0]:
                  src += f', int stride_{name}_block __multipleof({mult})'
                src += f", int stride_{name}_{d} __multipleof({mult})"
            src += "\n            "
        if lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += f", int stride_a_inner __multipleof({multipleof_a})"
            src += f", int rem_delta_a __multipleof({multipleof_a})"
        elif sparse_a or lut_mode_a == _einsum.LUT_MODE.DRAM:
            src += ", int* AD __noalias __readonly __aligned(16)"
        src += "\n            "
        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += f", int stride_b_inner __multipleof({multipleof_b})"
            src += f", int rem_delta_b __multipleof({multipleof_b})"
        elif sparse_b or lut_mode_b == _einsum.LUT_MODE.DRAM:
            src += ", int* BD"
        src += "\n            "
        if sparse_c:
            src += ", int* CD"
        if sparse_a or sparse_b:
            src += ", int width"
        src += """) {


    // program identifiers
    int pid_0 = get_program_id(0);
    int pid_1 = get_program_id(1);

"""
        if sparse_a:
            src += f"""
    int off_n = pid_0 / width;
    int off_header = pid_0 % width;
    int* header = AD + off_header * {2 + len(outer_sparse_a)};
    int* pdelta = AD + *(header + 0);
    matmul_k  = *(header + 1);"""
            for i, d in enumerate(outer_sparse_a):
                src += f"""
    int off_{d} = *(header + {2 + i});"""
            src += f"""
    int inca  = *(pdelta + 0);
    int incb  = *(pdelta + 1);
    int off_{''.join(map(str, outer_dense_a))} = pid_1;
"""
            _einsum.unpack_cc(None, outer_dense_a, "off_", False)
        elif sparse_b:
            src += f"""
    int off_m = pid_0 / width;
    int off_header = pid_0 % width;
    int* header = BD + off_header * {2 + len(outer_sparse_b)};
    int* pdelta = BD + *(header + 0);
    matmul_k  = *(header + 1);"""
            for i, d in enumerate(outer_sparse_b):
                src += f"""
    int off_{d} = *(header + {2 + i});"""
            src += f"""
    int incb  = *(pdelta + 0);
    int inca  = *(pdelta + 1);
    int off_{''.join(map(str, outer_dense_b))} = pid_1;
"""
            _einsum.unpack_cc(None, outer_dense_b, "off_", False)
        elif sparse_c:
            src += f"""
    // load LUT header
    int *header  = CD + pid_0 * {len(sparse_c)};"""
            for i, d in enumerate(sparse_c):
                src += f"""
    int off_{d}  = *(header + {i});"""
            src += f"""
    int off_{''.join(map(str, outer_dense_c))} = pid_1;"""
        else:
            src += """
    // re-order outer program ids
    int grid_m = (matmul_m + TM - 1) / TM;
    int grid_n = (matmul_n + TN - 1) / TN;
    int off_mn = pid_0 / div_m;
    int off_n = off_mn % grid_n;
    int off_m = (off_mn / grid_n)*div_m + (pid_0 % div_m);
    int off_b = get_program_id(1);"""

        src += """
#if TZ == 1
    int off_k = 0;
#else
    // get reduction sub-group program id
    int pid_z = get_program_id(2);
    int grid_z = get_num_programs(2);
    int div_z = matmul_k / TZ;
    int rem_z = matmul_k % TZ;
    int off_k = pid_z * div_z;
    matmul_k = select(pid_z < rem_z, div_z, div_z + rem_z);
#endif
    int rem_k = matmul_k % TK;
    
    // create ranges
"""

        sparse = sparse_a + sparse_b + sparse_c
        for axes, tile, off, prefixes in zip([axes_m, axes_n, axes_b, axes_k],
                                             ['TM', 'TN', 'TB', 'TK'],
                                             ['off_m*TM', 'off_n*TN', 'off_b*TB', 'off_k'],
                                             [['a', 'c'], ['b', 'c'], ['a', 'b', 'c'], ['a', 'b']]):
            if not axes:
                continue
            currs = ''.join(map(str,axes))
            has_sparse_component = set(axes) & set(sparse)
            if has_sparse_component:
                src += f"    int r{currs}[{tile}] = 0 ... {tile};\n"
                src += _einsum.unpack_cc(tile, axes, f'r', False) 
            else:
                src += f"    int r{currs}[{tile}] = {off} + 0 ... {tile};\n"
                src += _einsum.unpack_cc(tile, axes, f'r', False) 
            for pfx in prefixes:
                for d in axes:
                    is_dense_dim = d not in sparse
                    is_dense_storage = (pfx == 'a' and not sparse_a) or\
                                       (pfx == 'b' and not sparse_b) or\
                                       (pfx == 'c' and not sparse_c)
                    if not is_dense_dim and is_dense_storage:
                        src += f"    int {pfx}r{d}[{tile}] = off_{d} * BLOCK{d.upper()} + r{d};\n"
                    elif is_dense_dim and has_sparse_component:
                        src += f"    int {pfx}r{d}[{tile}] = off_{d};\n"
                    else:
                        src += f"    int {pfx}r{d}[{tile}] = r{d};\n"

        src += f"""    
    // initialize pointers to A
    int offa[TM, TK, TB] = {'inca' if sparse_a or sparse_b else '0'} """
        for i, sym in enumerate(expr_a):
            ccode = _einsum.print_cc(sym, axes_m, axes_k, axes_b, 'a')
            stride = f'stride_a_{i}' if i < len(expr_a) - 1 else f'{stride_a_last}'
            src += f" + ({ccode}) * {stride}\n                            "
        src += ';'

        src += """
    TYPE *pa[TM, TK, TB] = A + offa;"""
       

        if not sparse_a and not sparse_b and use_lut_a and not lut_mode_a == _einsum.LUT_MODE.SCALAR:
            spec = '__constant__' if lut_mode_a == _einsum.LUT_MODE.CONSTANT else ''
            cast = '(int __constant__*)' if lut_mode_a == _einsum.LUT_MODE.CONSTANT else ''
            src += f"""
    int offadelta[TK] = off_k + 0 ... TK;
    int {spec} *padelta[TK]  = {cast}AD  + offadelta;
    int incda[TM, TK, TB] = (*padelta)[newaxis, :, newaxis];"""
    
        src += f"""

    // initialize pointers to B
    int offb[TK, TN, TB] = {'incb' if sparse_a or sparse_b else '0'}"""
        for i, sym in enumerate(expr_b):
            ccode = _einsum.print_cc(sym, axes_k, axes_n, axes_b, 'b')
            stride = f'stride_b_{i}' if i < len(expr_b) - 1 else f'{stride_b_last}'
            src += f" + ({ccode}) * {stride}\n                            "
        src += ';'

        src += """
    TYPE *pb[TK, TN, TB] = B + offb;"""


        if not sparse_a and not sparse_b and use_lut_b and not lut_mode_b == _einsum.LUT_MODE.SCALAR:
            spec = '__constant__' if lut_mode_b == _einsum.LUT_MODE.CONSTANT else ''
            cast = '(int __constant__*)' if lut_mode_b == _einsum.LUT_MODE.CONSTANT else ''
            src += f"""
    // initialize pointers to B look-up table
    int offbdelta[TK] = off_k + 0 ... TK;
    int *pbdelta[TK]  = BD  + offbdelta;"""

   
        rk = 'r{}'.format(''.join(map(str,axes_k)))
        src += f"""

    // prefetch 
    int prefetch_k = select(rem_k > 0, rem_k, TK);
    bool checkam[TM] = ar""" + ''.join(map(str,axes_m)) + f""" < matmul_m;
    bool checkbn[TN] = br""" + ''.join(map(str,axes_n)) + f""" < matmul_n;
    bool checkk[TK] = r{''.join(map(str, axes_k))} < prefetch_k;
    bool checka[TM, TK, TB] = checkam[:, newaxis, newaxis] && checkk[newaxis, :, newaxis];
    bool checkb[TK, TN, TB] = checkk[:, newaxis, newaxis] && checkbn[newaxis, :, newaxis];
    TYPE a[TM, TK, TB] = checka ? *pa : 0;
    TYPE b[TK, TN, TB] = checkb ? *pb : 0;"""

        if sparse_a:
            src += f"""
    // update pointers to look-up tables
    pdelta += 2;
    int incda = *(pdelta + 0);
    int incdb = *(pdelta + 1);
    pa += incda;
    pb += incdb;"""
        if sparse_b:
            src += f"""
    // update pointers to look-up tables
    pdelta += 2;
    int incdb = *(pdelta + 0);
    int incda = *(pdelta + 1);
    pa += incda;
    pb += incdb;"""

        if not sparse_a and not sparse_b and lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += """
    pa += rem_delta_a;"""
        elif not sparse_a and not sparse_b:
            src += """
    pa += incda;
    padelta += TK;
    incda = (*padelta)[newaxis, :, newaxis];"""

        if not sparse_a and not sparse_b and lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += """
    pb += rem_delta_b;"""
        elif not sparse_a and not sparse_b:
            src += """
    pb += (*pbdelta)[:, newaxis, newaxis];
    pbdelta += TK;"""

        src += f"""

    // accumulate
    float acc[TM, TN, TB] = 0;
    for(int k = matmul_k; k > 0; k -= TK) {{
        acc += a @ b;
        
        // load inputs
        checkk = k > TK;
        checka = checkam[:, newaxis, newaxis] && checkk[newaxis, :, newaxis];
        checkb = checkk[:, newaxis, newaxis] && checkbn[newaxis, :, newaxis];
        a = *?(checka)pa;
        b = *?(checkb)pb;
        
        // update pointers"""
        if sparse_a:
            src += """
        pdelta += 2;
        incda = *(pdelta + 0);
        incdb = *(pdelta + 1);
        pa += incda;
        pb += incdb;
        """
        elif sparse_b:
            src += """
        pdelta += 2;
        incdb = *(pdelta + 0);
        incda = *(pdelta + 1);
        pa += incda;
        pb += incdb;
        """
        else:
            if lut_mode_a == _einsum.LUT_MODE.SCALAR:
                src += """
        pa += stride_a_inner;"""
            else:
                src += """
        pa += incda;
        padelta += TK;
        incda = (*padelta)[newaxis, :, newaxis];"""
            if lut_mode_b == _einsum.LUT_MODE.SCALAR:
                src += """
        pb += stride_b_inner;"""
            else:
                src += """
        pb += (*pbdelta)[:, newaxis, newaxis];
        pbdelta += TK;"""

        src += f"""
    }}
    TYPE c[TM, TN, TB] = acc;
   
    // initialize pointers to C
    int offc[TM, TN, TB] = {'pid_0*TN*TN' if sparse_c else 0}"""
        for i, sym in enumerate(expr_c):
            stride = f'stride_c_{i}' if i < len(expr_c) - 1 else f'{stride_c_last}'
            ccode = _einsum.print_cc(sym, axes_m, axes_n, axes_b, 'c')
            src += f"\n                          + ({ccode}) * {stride}"
        src += ';'

        src += """
    TYPE *pc[TM, TN, TB] = C + offc;
    
    // bounds-checking
    bool checkcm[TM] = cr""" + ''.join(map(str,axes_m)) + """ < matmul_m;
    bool checkcn[TN] = cr""" + ''.join(map(str,axes_n)) + """ < matmul_n;
    bool checkc[TM, TN, TB] = checkcm[:, newaxis, newaxis] && 
                              checkcn[newaxis, :, newaxis];

    // write back
#if TZ == 1
    *?(checkc)pc = c;
#else
    int *plock = locks + pid_mn + pid_b * get_num_programs(0);
    int *pcount = plock + 1024*1024;
    // spin
    for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
    int count = *pcount;
    if(count == 0)
      *?(checkc)pc = c;
    else
      *?(checkc)pc = c + *?(checkc)pc;
    atomic_xchg(pcount, (count + 1) % (grid_z));
    atomic_xchg(plock, 0);
#endif
}
"""
        #print(src)
        # compilation options
        TM, TN, TB, TZ = [32], [32], 1, [1]
        TK = 16 if dtype==torch.float16 else 8
        defines =  {'TM': TM, 'TN': TN, 'TB': TB, 'TK': TK, 'TZ': TZ, 'TYPE': dtype}
        for d, B in blocks.items():
            defines[f'BLOCK{d}'] = B
        # create kernel
        ret = triton.kernel(src, defines=defines)
        # set constant
        if use_lut_a and lut_mode_a == _einsum.LUT_MODE.CONSTANT:
            ret.set_constant('AD', delta_a)
        if use_lut_b and lut_mode_b == _einsum.LUT_MODE.CONSTANT:
            ret.set_constant('BD', delta_b)
        return ret

    ############################
    ## Look-up Table
    ############################

    class LUT_MODE(IntEnum):
        SCALAR = 1
        CONSTANT = 2
        DRAM = 3
    
    def lut_mode(delta):
        if delta.size == 0 or np.min(delta) == np.max(delta):
            return _einsum.LUT_MODE.SCALAR
        #if delta.size < 4096:
        #    return _einsum.LUT_MODE.CONSTANT
        return _einsum.LUT_MODE.DRAM

    def symbolic_delta(symbols, axes):
        rank = len(symbols)
        strides = [sp.symbols(f'stride{d}') for d in range(rank)]
        nexts = {s: sp.symbols(f'next{s}') for s in axes}
        delta = 0
        for i in range(rank):
            delta += strides[i] * (symbols[i].subs(nexts) - symbols[i])
        return delta

    def unpack_offset(k, axes, dims):
        ret = dict()
        for d in reversed(axes):
            ret[d] = k % dims[d]
            k = k // dims[d]
        return ret
    
        
    def make_dsd_delta(axes, step, stride, dims, symbols, sparse, layout, blocks):
        # depth of reductions
        depth = layout.sum(*[i for i, d in enumerate(sparse) if d in axes])
        # outer dimension indices
        outer = depth.nonzero()
        outer = [outer[:,i] for i in range(outer.shape[1])]
        # find offset of outer dimensions
        depth = depth.view(-1)
        offsets = torch.zeros_like(depth)
        offsets[1:] = torch.cumsum(depth[:-1], 0)
        # compute delta for b
        # TODO: support multiple sparse red indices
        col = next((i for i, d in enumerate(sparse) if d in axes), None)
        block = blocks[sparse[-1].upper()]
        div = block // step
        delta_b = layout.transpose(-1, col).nonzero()[:, -1].reshape(-1).contiguous()
        delta_b *= block
        delta_b = [delta_b + step*i for i in range(div)]
        delta_b = torch.stack(delta_b, dim=1)
        delta_b = delta_b.view(-1)
        # compute delta for a
        bstride = 1
        for d in sparse[::-1]:
            if d in axes:
                break
            bstride *= blocks[d.upper()]
        order = [d for d in sparse if d not in axes] +\
                [d for d in sparse if d in axes]
        idx = [sparse.index(d) for d in order]
        #layout = layout.clone().transpose(-1, col)
        layout[layout > 0] = 1 + torch.arange(layout.sum(), device=layout.device)
        layout = layout.permute(*idx)
        delta_a = layout[layout > 0] - 1
        delta_a *= np.prod(list(blocks.values()))
        saved = delta_a[offsets]
        delta_a[1:] = delta_a[1:] - delta_a[:-1]
        delta_a = delta_a.view(-1, 1).repeat(1, div)
        delta_a[:, 1:] = step*bstride
        delta_a[:, 0] -= (div - 1)*step*bstride
        delta_a[offsets, 0] = saved
        delta_a = delta_a.view(-1)
        delta = torch.stack((delta_a, delta_b), dim=1).view(-1).contiguous()
        # form look-up table
        depth *= blocks[symbols[-1].upper()]
        offsets *= div
        header = torch.stack((offsets, depth, *outer), dim=1).view(-1).contiguous()
        nouter = 2 + len(outer)
        header[::nouter] = header[::nouter]*2 + header.shape[0]
        lut = torch.cat((header, delta)).int().int().cpu().numpy()
        return lut, nouter, _einsum.LUT_MODE.DRAM

    def make_delta(axes, step, stride, dims, symbols, sparse, layout, lut = None, nouter = None):
        # symbolic pointer increments
        symbols = [sp.symbols(x) for x in symbols]
        delta = _einsum.symbolic_delta(symbols, axes)
        args =  [f'stride{d}' for d in range(len(stride))]
        args += [f'{sk}' for sk in axes]
        args += [f'next{sk}' for sk in axes]
        fn = sp.lambdify(args, delta, 'numpy')
        if lut is None:
            # inner axes values
            inner = [dims[d] for d in axes]
            inner = np.prod(inner)
            rem = inner % step
            rem = rem if rem > 0 else step
            # k     = [0, 1, ..., step,  rem, rem + 1, ... rem + inner]
            # nextk = [rem, 1 + rem, ..., step + rem, rem + step, rem + 1 + step, ..., inner + step]
            k = np.concatenate((np.arange(step), np.arange(rem, inner))).astype(np.int32)
            nextk = np.concatenate((k[:step] + rem, k[step:] + step))
        else:
            idx   = (lut[:lut[0]:nouter] - lut[0])//2
            k     = lut[lut[0]+1::2]
            k     = np.insert(k, idx, 0)
            nextk = k[1:]
            k     = k[:-1]
        # offsets
        off      = _einsum.unpack_offset(k, axes, dims)
        nextoff  = _einsum.unpack_offset(nextk, axes, dims)
        # evaluate deltas
        args  = [s for s in stride]
        args += [off[sk] for sk in axes]
        args += [nextoff[sk] for sk in axes]
        delta = fn(*args)
        delta = np.maximum(delta, 0)
        #print(k, nextk, lut)
        #print(args)
        #exit()
        if lut is not None:
            idx   = idx[1:] + np.arange(idx.shape[0] - 1)
            delta = np.delete(delta, idx)
            lut[lut[0]+1::2] = delta
            return None, None
        return delta, _einsum.lut_mode(delta[step:-step])

    ############################
    ## Einsum parsing
    ############################

    def uniq(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def parse_axes(expr_a, expr_b, expr_c):
        sym_a = [x for s in expr_a for x in s]
        sym_b = [x for s in expr_b for x in s]
        sym_c = [x for s in expr_c for x in s]
        batch = [d for d in sym_a if d in sym_b and d in sym_c]
        outer = [d for d in sym_a if d not in sym_b and d in sym_c]
        inner = [d for d in sym_a if d in sym_b and d not in sym_c]
        variables = [d for d in sym_a if d not in sym_b and d not in sym_c]
        return _einsum.uniq(batch), _einsum.uniq(outer), _einsum.uniq(inner), variables


    def replace_subscript(expr, arrays):
        # replace array indexing by Indexed()
        indexed = re.findall('([_a-zA-Z][_a-zA-Z0-9]*)\[([_a-z]*)\]', expr)
        for x in indexed:
            arrays.append(x[0])
            expr = expr.replace(f'{x[0]}[{x[1]}]', f'Indexed({x[0]},{x[1]})')
        return expr


    def parse_expr(expr):
        # extract symbols
        sym = []
        sparse = []
        i = 0
        while i < len(expr):
            d = expr[i]
            if d == '(':
                size = expr[i:].find(')')
                d = expr[i : i + size + 1]
                sym.append(parse_expr(d))
                i += size + 1
            else:
                if d.isupper():
                  sparse += [d.lower()]
                sym.append(d.lower())
                i += 1
        return sym, sparse
  
    ############################
    ## Preprocessing
    ############################

    @staticmethod
    def pad(tensor, pad):
        pad = pad + [0] *  (2*len(tensor.shape) - len(pad))
        begin = [ x if x > 0 else None for x in pad[-1::-2]]
        end   = [-x if x > 0 else None for x in pad[-2::-2]]
        slices = [slice(b, e) for b, e in zip(begin, end)]
        tensor = torch.nn.functional.pad(tensor, pad, 'constant', 0)
        tensor = tensor[slices]
        return tensor

    @staticmethod
    def make_sdd_lut(layout_c, sparse_c, blocks):
        nnz = layout_c.nonzero()
        lut = nnz.reshape(-1).int().cuda()
        return lut


    ############################
    ## Compilation
    ############################

    class instance:

        locks = None
        kernel_cache = dict()

        @staticmethod
        def _tile(M, N, B, TMs, TNs, TBs, TZs, TK):
            smp = 15
            # occupancy estimation
            grid = lambda TM, TN, TB, TZ:   \
                        triton.cdiv(M, TM)* \
                        triton.cdiv(N, TN)* \
                        triton.cdiv(B, TB)* \
                        TZ
            occupancy = lambda TM, TN, TB, TZ: \
                           min(grid(TM, TN, TB, TZ), 4*smp)
            # arithmetic intensity estimation
            intensity = lambda TM, TN: \
                           TM * TN * TK / (TM*TK + TK*TN)
            # occupancy/intensity for all configurations
            estimates = {(TM, TN, TB, TZ): (occupancy(TM, TN, TB, TZ), intensity(TM, TN)) \
                        for TM in TMs \
                        for TN in TNs \
                        for TB in TBs \
                        for TZ in TZs }
            # returns configuration that maximizes occupancy subject to maximizing intensity
            estimates = sorted(estimates.items(), 
                               key=lambda item: item[1], 
                               reverse=True)
            return estimates[0][0]
        
        
        def __init__(self, einsum, dtype, stride_a, stride_b, shape_a, shape_b, layout_a, layout_b, layout_c, blocks):
            # parse symbols
            expr_a, expr_bc = einsum.split(",")
            expr_b, expr_c  = expr_bc.split("->")
            sym_a, sparse_a = _einsum.parse_expr(expr_a)
            sym_b, sparse_b = _einsum.parse_expr(expr_b)
            sym_c, sparse_c = _einsum.parse_expr(expr_c)
            # parse axes
            axes_b, axes_m, axes_k, var = _einsum.parse_axes(sym_a, sym_b, sym_c)
            _, axes_n, _, _           = _einsum.parse_axes(sym_b, sym_a, sym_c)
            axes = axes_b + axes_m + axes_n + axes_k
            # check block sizes
            for d in sparse_a + sparse_b + sparse_c:
                if d.upper() not in blocks:
                    raise ValueError(f'unspecified block size for dimension: {d.upper()}')
            # check layout is present
            if sparse_a and layout_a is None:
                raise ValueError('A is sparse but not layout provided')
            if sparse_b and layout_b is None:
                raise ValueError('B is sparse but not layout provided')
            if sparse_c and layout_c is None:
                raise ValueError('C is sparse but not layout provided')
            # check dimensions
            dims_a  = dict([(x, y) for x,y in zip(sym_a, shape_a) if x not in sparse_a])
            dims_b  = dict([(x, y) for x,y in zip(sym_b, shape_b) if x not in sparse_b])
            dims_La = None if layout_a is None else dict(zip([x for x in expr_a if x.isupper()], layout_a.shape))
            dims_Lb = None if layout_b is None else dict(zip([x for x in expr_b if x.isupper()], layout_b.shape))
            # TODO: could be cleaner
            read_shape = lambda d, dimsT, dimsL, sparse: dimsL[d.upper()] * blocks[d.upper()] if d in sparse else dimsT[d]
            for d in axes_b + axes_m + axes_n + axes_k:
                dim_a = read_shape(d, dims_a, dims_La, sparse_a) if d in sym_a else None
                dim_b = read_shape(d, dims_b, dims_Lb, sparse_b) if d in sym_b else None
                if d in axes_b and dim_a and dim_b and dim_a != dim_b:
                    raise ValueError(f'incomparible batch dimension {d} (A: {dim_a}, B: {dim_b})')
                if d in axes_k and dim_a and dim_b and dim_a != dim_b:
                    raise ValueError(f'incompatible inner dimension {d} (A: {dim_a}, B: {dim_b})')
            dims = dict()
            dims.update(dims_a)
            dims.update(dims_b)
            for i, d in enumerate(sparse_a):
                dims[d] = layout_a.shape[i] * blocks[d.upper()]
            for i, d in enumerate(sparse_b):
                dims[d] = layout_b.shape[i] * blocks[d.upper()]
            # allocate output
            shape_c = [dims[d] if d.islower() else blocks[d] for d in expr_c]
            if sparse_c:
                shape_c.insert(expr_c.index(sparse_c[0].upper()), int(layout_c.sum()))
            stride_c = [None] * len(shape_c)
            stride_c[-1] = 1
            for i in reversed(range(len(shape_c) - 1)):
                stride_c[i] = stride_c[i+1] * shape_c[i+1]
            # look-up tables
            TK = 16 if dtype == torch.float16 else 8
            if sparse_a and not sparse_b:
                delta_a, nouter, lut_mode_a = _einsum.make_dsd_delta(axes_k, TK, stride_a, dims, sym_a, sparse_a, layout_a, blocks)
                delta_b, lut_mode_b = _einsum.make_delta(axes_k, TK, stride_b, dims, sym_b, sparse_b, layout_b, delta_a, nouter)
            if sparse_b and not sparse_a:
                delta_b, nouter, lut_mode_b = _einsum.make_dsd_delta(axes_k, TK, stride_b, dims, sym_b, sparse_b, layout_b, blocks)
                delta_a, lut_mode_a = _einsum.make_delta(axes_k, TK, stride_a, dims, sym_a, sparse_a, layout_a, delta_b, nouter)
            if not sparse_a and not sparse_b:
                delta_a, lut_mode_a = _einsum.make_delta(axes_k, TK, stride_a, dims, sym_a, sparse_a, layout_a)
                delta_b, lut_mode_b = _einsum.make_delta(axes_k, TK, stride_b, dims, sym_b, sparse_b, layout_b)
            if sparse_c:
                delta_c = _einsum.make_sdd_lut(layout_c, sparse_c, blocks)
            # hash for recompilation
            stride_a_multiple = max([x for x in [1, 2, 4, 8] if shape_a[-1] % x == 0])
            stride_b_multiple = max([x for x in [1, 2, 4, 8] if shape_b[-1] % x == 0])
            stride_c_multiple = max([x for x in [1, 2, 4, 8] if shape_c[-1] % x == 0])
            stride_a_last = stride_a[-1]
            stride_b_last = stride_b[-1]
            stride_c_last = stride_c[-1]
            name = f'{dtype}_{expr_a}_{expr_b}_{expr_c}_{lut_mode_a}_{lut_mode_b}'\
                   f'_{stride_a_multiple}_{stride_b_multiple}_{stride_c_multiple}'\
                   f'_{stride_a_last}_{stride_b_last}_{stride_c_last}'  
            # recompile if necessary
            cache = _einsum.instance.kernel_cache
            if name not in cache:
                cachesize = len(cache)
                cache[name] = _einsum.make_kernel(f'__einsum{cachesize}',
                                                        dtype, 
                                                        sym_a, sym_b, sym_c, 
                                                        sparse_a, sparse_b, sparse_c,
                                                        axes_m, axes_n, axes_k, axes_b, 
                                                        stride_a_multiple, stride_b_multiple, stride_c_multiple,
                                                        stride_a_last, stride_b_last, stride_c_last,
                                                        lut_mode_a, lut_mode_b,
                                                        delta_a, delta_b,
                                                        blocks)
            self.kernel = cache[name]
            # Initialize locks
            if _einsum.instance.locks is None:
                _einsum.instance.locks = torch.zeros(2*1024*1024, dtype=torch.int32).cuda()
            # Kernel arguments
            dim_m = [dims[d] for d in axes_m]
            dim_n = [dims[d] for d in axes_n]
            dim_k = [dims[d] for d in axes_k]
            dim_b = [dims[d] for d in axes_b]
            M = reduce(mul, dim_m, 1)
            N = reduce(mul, dim_n, 1)
            K = reduce(mul, dim_k, 1)
            B = reduce(mul, [dims[d] for d in axes_b if d.upper() not in einsum], 1)
            stride_a = list(stride_a[:-1])
            stride_b = list(stride_b[:-1])
            stride_c = list(stride_c[:-1])
            alpha = 1.
            div_m = 1
            self.args  = [None, None, None]
            self.args += [_einsum.instance.locks]
            self.args += [alpha, M, N, K, div_m]
            self.args += dim_m
            self.args += dim_n 
            self.args += dim_k
            self.args += dim_b
            self.args += stride_a
            self.args += stride_b
            self.args += stride_c
            # LUT for A
            if lut_mode_a == _einsum.LUT_MODE.SCALAR:
                self.args += [delta_a[TK], delta_a[0]]
            elif sparse_a or lut_mode_a == _einsum.LUT_MODE.DRAM:
                self.args += [torch.from_numpy(delta_a).cuda()]
            # LUT for B
            if lut_mode_b == _einsum.LUT_MODE.SCALAR:
                self.args += [delta_b[TK], delta_b[0]]
            elif sparse_b or lut_mode_b == _einsum.LUT_MODE.DRAM:
                self.args += [torch.from_numpy(delta_b).cuda()]
            # LUT for C
            if sparse_c:
                self.args += [delta_c]
            if sparse_a or sparse_b:
                width = delta_a[0] // nouter if sparse_a else delta_b[0] // nouter
                self.args += [width]
            # Grid
            if sparse_a:
                self.grid = lambda opt: [width*triton.cdiv(N, opt.d('TN')), B, opt.d('TZ')]
            elif sparse_b:
                self.grid = lambda opt: [width*triton.cdiv(M, opt.d('TM')), B, opt.d('TZ')]
            elif sparse_c:
                width = int(layout_c.sum())
                self.grid = lambda opt: [width, B, opt.d('TZ')]
            else:
                self.grid = lambda opt: [triton.cdiv(M, opt.d('TM')) * 
                                         triton.cdiv(N, opt.d('TN')),
                                         triton.cdiv(B, opt.d('TB')),
                                         opt.d('TZ')]
            # position of dynamic arguments
            self.pos_a = 0
            self.pos_b = 1
            self.pos_c = 2
            # save information on the operation
            self.expr_a = expr_a
            self.expr_b = expr_b
            self.expr_c = expr_c
            self.matmul_B = B
            self.matmul_M = M
            self.matmul_N = N
            self.matmul_K = K
            # output shape
            self.shape_c = shape_c

                    
        def run(self, a, b):
            c = torch.empty(*self.shape_c, dtype=a.dtype, device=a.device)
            self.args[self.pos_a] = a
            self.args[self.pos_b] = b
            self.args[self.pos_c] = c
            self.kernel(*self.args, grid=self.grid)
            return c




    ############################
    ## Forward
    ############################

    instance_cache = dict()
    registry = dict()
    @staticmethod
    def forward(ctx, expr, a, b, layout_a, layout_b, layout_c, blocks):
        # compile einsum instance
        cache = _einsum.instance_cache
        key = (expr, a.dtype, 
               a.stride(), b.stride(),
               a.shape   , b.shape)
        if key not in cache:
            cache[key] = _einsum.instance(expr, a.dtype, 
                                          a.stride(), b.stride(),
                                          a.shape   , b.shape   ,
                                          layout_a, layout_b, layout_c, blocks)
        instance = cache[key]
        # run and mark as dirty c modified in-place
        c = instance.run(a, b)
        # save information in context
        ctx.expr_a = instance.expr_a
        ctx.expr_b = instance.expr_b
        ctx.expr_c = instance.expr_c
        ctx.matmul_B = instance.matmul_B
        ctx.matmul_M = instance.matmul_M
        ctx.matmul_N = instance.matmul_N
        ctx.matmul_K = instance.matmul_K
        ctx.save_for_backward(a, b)
        return c


    ############################
    ## Backward
    ############################

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        expr_a = ctx.expr_a
        expr_b = ctx.expr_b
        expr_c = ctx.expr_c
        # gradient of first argument
        da = None
        if ctx.needs_input_grad[1]:
            da = torch.empty_like(a)
            einsum(f'{expr_c},{expr_b}->{expr_a}', dy, b, da)
        # gradient of second argument
        db = None
        if ctx.needs_input_grad[2]:
            db = torch.empty_like(b)
            einsum(f'{expr_a},{expr_c}->{expr_b}', a, dy, db)
        return None, da, db, None, None, None, None, None, None, None


def einsum(expr, a, b,
           layout_a = None, layout_b = None, layout_c = None, blocks = dict()):
    return _einsum.apply(expr, a, b, layout_a, layout_b, layout_c, blocks)