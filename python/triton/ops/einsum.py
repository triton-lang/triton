import numpy as np
import torch
from math import ceil, log2
from enum import IntEnum
import triton
from functools import reduce
from operator import mul
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
from collections import OrderedDict
from collections import namedtuple
import re
from sympy.printing.ccode import C89CodePrinter

       
class _einsum(triton.function):


    #############################
    ## Triton-C code generation
    #############################
    def print_cc(expr, axes_0, axes_1, axes_2):

        class TritonCodePrinter(C89CodePrinter):
            
            def __init__(self, axes_0, axes_1, axes_2):
                super(TritonCodePrinter, self).__init__()
                self.axes_0 = axes_0
                self.axes_1 = axes_1
                self.axes_2 = axes_2

            def _print_Symbol(self, expr):
                name = super(C89CodePrinter, self)._print_Symbol(expr)
                if expr in self.axes_0:
                    return f'r{name}[:, newaxis, newaxis]'
                if expr in self.axes_1:
                    return f'r{name}[newaxis, :, newaxis]'
                if expr in self.axes_2:
                    return f'r{name}[newaxis, newaxis, :]'
                return name

            def _print_Indexed(self, expr):
                assert len(expr.indices) == 1
                return "*(%s + %s)" % (self._print(expr.base.label),
                                    self._print(expr.indices[0]))
        
        return TritonCodePrinter(axes_0, axes_1, axes_2).doprint(expr)


    def unpack_cc(tile, axes, prefix, remat):
        ret = ''
        axes = list(map(str, axes))
        for i, d in enumerate(reversed(axes)):
            if i == len(axes) - 1:
                break
            currs = ''.join(axes[: len(axes) - i])
            nexts = ''.join(axes[: len(axes) - (i + 1)])
            ty = '' if remat else 'int '
            sz = '' if remat else f'[{tile}]'
            ret += f'    {ty}{prefix}{nexts}{sz} = r{currs} / dim_{d};\n'
            ret += f'    {ty}{prefix}{d}{sz} = r{currs} % dim_{d};\n'
        return ret

    def strides_cc(name, expr):
        ret = [f'stride_{name}_{d}' for d in expr[:-1]] + ['1']
        ret = dict(zip(expr, ret))
        return ret

    def make_kernel(name,
                    expr_a, expr_b, expr_c,
                    axes_m, axes_n, axes_k, axes_b,
                    multipleof_a, multipleof_b, multipleof_c,
                    stride_a_last, stride_b_last, stride_c_last,
                    lut_mode_a, lut_mode_b,
                    delta_a, delta_b,
                    subscripted):

        use_lut_a = True
        use_lut_b = True

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
        for dim, name, mult in zip([expr_a, expr_b, expr_c],
                                         ['a', 'b', 'c'],
                                         [multipleof_a, multipleof_b, multipleof_c]):
            for d in range(len(dim) - 1):
                attr = f'__multipleof({mult})'
                src += f", int stride_{name}_{d} {attr}"
            src += "\n            "
        if lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += f", int stride_a_inner __multipleof({multipleof_a})"
            src += f", int rem_delta_a __multipleof({multipleof_a})"
        elif lut_mode_a == _einsum.LUT_MODE.DRAM:
            src += ", int* AD __noalias __readonly __aligned(16)"
        src += "\n            "
        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += f", int stride_b_inner __multipleof({multipleof_b})"
            src += f", int rem_delta_b __multipleof({multipleof_b})"
        elif lut_mode_b == _einsum.LUT_MODE.DRAM:
            src += ", int* BD"
        src += "\n"
        for ptr in subscripted:
            src += f", int* {ptr}"
        src += """) {

    // re-order outer program ids
    int grid_m = (matmul_m + TM - 1) / TM;
    int grid_n = (matmul_n + TN - 1) / TN;
    int pid_mn = get_program_id(0) / div_m;
    int pid_n = pid_mn % grid_n;
    int pid_m = (pid_mn / grid_n)*div_m + (get_program_id(0) % div_m);

    // get batch program id
    int pid_b = get_program_id(1);

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
        rk = 'r{}'.format(''.join(map(str,axes_k)))
        for axes, tile, off in zip([axes_m, axes_n, axes_b, axes_k],
                                   ['TM', 'TN', 'TB', 'TK'],
                                   ['pid_m*TM', 'pid_n*TN', 'pid_b*TB', 'off_k']):
            currs = ''.join(map(str,axes))
            if axes:
                src += f"    int r{currs}[{tile}] = {off} + 0 ... {tile};\n"
                src += _einsum.unpack_cc(tile, axes, 'r', False)

        src += """    
    // initialize pointers to A
    int offa[TM, TK, TB] = """
        for i, sym in enumerate(expr_a):
            ccode = _einsum.print_cc(sym, axes_m, axes_k, axes_b)
            stride = f'stride_a_{i}' if i < len(expr_a) - 1 else f'{stride_a_last}'
            if i > 0:
                src += ' + '
            src += f"({ccode}) * {stride}\n                            "
        src += ';'

        src += """
    TYPE *pa[TM, TK, TB] = A + offa;"""
       
        if use_lut_a and not lut_mode_a == _einsum.LUT_MODE.SCALAR:
            spec = '__constant__' if lut_mode_a == _einsum.LUT_MODE.CONSTANT else ''
            cast = '(int __constant__*)' if lut_mode_a == _einsum.LUT_MODE.CONSTANT else ''
            src += f"""
    // initialize pointers to A look-up table
    int offadelta[TK] = off_k + 0 ... TK;
    int {spec} *padelta[TK]  = {cast}AD  + offadelta;
    int incda[TM, TK, TB] = (*padelta)[newaxis, :, newaxis];"""
    
        src += """

    // initialize pointers to B
    int offb[TK, TN, TB] = """
        for i, sym in enumerate(expr_b):
            ccode = _einsum.print_cc(sym, axes_k, axes_n, axes_b)
            stride = f'stride_b_{i}' if i < len(expr_b) - 1 else f'{stride_b_last}'
            if i > 0:
                src += ' + '
            src += f"({ccode}) * {stride}\n                            "
        src += ';'

        src += """
    TYPE *pb[TK, TN, TB] = B + offb;"""


        if use_lut_b and not lut_mode_b == _einsum.LUT_MODE.SCALAR:
            spec = '__constant__' if lut_mode_b == _einsum.LUT_MODE.CONSTANT else ''
            cast = '(int __constant__*)' if lut_mode_b == _einsum.LUT_MODE.CONSTANT else ''
            src += f"""
    // initialize pointers to B look-up table
    int offbdelta[TK] = off_k + 0 ... TK;
    int *pbdelta[TK]  = BD  + offbdelta;"""

        src += f"""

    // prefetch 
    int prefetch_k = select(rem_k > 0, rem_k, TK);
    bool checkm[TM] = r""" + ''.join(map(str,axes_m)) + f""" < matmul_m;
    bool checkn[TN] = r""" + ''.join(map(str,axes_n)) + f""" < matmul_n;
    bool checkk[TK] = {rk} < prefetch_k;
    bool checka[TM, TK, TB] = checkm[:, newaxis, newaxis] && checkk[newaxis, :, newaxis];
    bool checkb[TK, TN, TB] = checkk[:, newaxis, newaxis] && checkn[newaxis, :, newaxis];
    TYPE a[TM, TK, TB] = checka ? *pa : 0;
    TYPE b[TK, TN, TB] = checkb ? *pb : 0;"""

        if lut_mode_a == _einsum.LUT_MODE.SCALAR:
            src += """
    pa += rem_delta_a;"""
        else:
            src += """
    pa += incda;
    padelta += TK;
    incda = (*padelta)[newaxis, :, newaxis];"""

        if lut_mode_b == _einsum.LUT_MODE.SCALAR:
            src += """
    pb += rem_delta_b;"""
        else:
            src += """
    pb += (*pbdelta)[:, newaxis, newaxis];
    pbdelta += TK;"""

        src += f"""
    // accumulate
    float acc[TM, TN, TB] = 0;
    for(int k = matmul_k; k > 0; k -= TK) {{
        acc += a @ b;
        #ifdef MASK
        uint32 bits[TM, TN, TB] = bitcast<uint32[TM,TN,TB]>(acc);
        acc = bitcast<TYPE[TM, TN, TB]>(bits & MASK);
        #endif

        checkk = k > TK;
        checka = checkm[:, newaxis, newaxis] && checkk[newaxis, :, newaxis];
        checkb = checkk[:, newaxis, newaxis] && checkn[newaxis, :, newaxis];
        a = *?(checka)pa;
        b = *?(checkb)pb;"""

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

    // re-materialize ranges
"""
        for axes, tile, off in zip([axes_m, axes_n, axes_b],
                                   ['TM', 'TN', 'TB'],
                                   ['pid_m*TM', 'pid_n*TN', 'pid_b*TB']):
            currs = ''.join(map(str,axes))
            if axes:
                src += f"    r{currs} = {off} + 0 ... {tile};\n"
                src += _einsum.unpack_cc(tile, axes, 'r', True)

        src += """
    // initialize pointers to C
    int offc[TM, TN, TB] = """
        for i, sym in enumerate(expr_c):
            stride = f'stride_c_{i}' if i < len(expr_c) - 1 else f'{stride_c_last}'
            ccode = _einsum.print_cc(sym, axes_m, axes_n, axes_b)
            if i > 0:
                src += ' + '
            src += f"({ccode}) * {stride}\n                            "
        src += ';'

        src += """
    TYPE *pc[TM, TN, TB] = C + offc;
    
    // bounds-checking
    checkm = r""" + ''.join(map(str,axes_m)) + """ < matmul_m;
    checkn = r""" + ''.join(map(str,axes_n)) + """ < matmul_n;
    bool checkc[TM, TN, TB] = checkm[:, newaxis, newaxis] && 
                              checkn[newaxis, :, newaxis];

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

        ret = triton.kernel(src)
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
    
    def make_delta(axes, step, stride, dims, symbols, arrays):
        # symbolic pointer increments
        delta = _einsum.symbolic_delta(symbols, axes)
        args =  [f'stride{d}' for d in range(len(stride))]
        args += [f'{sk}' for sk in axes]
        args += [f'next{sk}' for sk in axes]
        args += [f'{sk}' for sk, _ in arrays]
        fn = sp.lambdify(args, delta, 'numpy')
        # inner axes values
        inner = [dims[d] for d in axes]
        inner = np.prod(inner)
        rem = inner % step
        rem = rem if rem > 0 else step
        # k = [0, 1, ..., step, 
        #      rem, rem + 1, ... rem + inner]
        k = np.concatenate((np.arange(step), 
                            np.arange(rem, inner))).astype(np.int32)
        # nextk = [rem, 1 + rem, ..., step + rem, 
        #          rem + step, rem + 1 + step, ..., inner + step]
        nextk = np.concatenate((k[:step] + rem, 
                                k[step:] + step))
        # offsets
        off      = _einsum.unpack_offset(k, axes, dims)
        nextoff  = _einsum.unpack_offset(nextk, axes, dims)
        # evaluate deltas
        args  = [s for s in stride]
        args += [off[sk] for sk in axes]
        args += [nextoff[sk] for sk in axes]
        args += [x for _, x in arrays]
        delta = fn(*args)
        return delta, _einsum.lut_mode(delta[step:-step])

    ############################
    ## Einsum parsing
    ############################

    def uniq(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def parse_axes(expr_a, expr_b, expr_c, subscripted):
        is_index = lambda x: type(x) == sp.indexed.Indexed or str(x) in subscripted
        sym_a = [x for s in expr_a for x in s.free_symbols if not is_index(x)]
        sym_b = [x for s in expr_b for x in s.free_symbols if not is_index(x)]
        sym_c = [x for s in expr_c for x in s.free_symbols]
        batch = [d for d in sym_a if d in sym_b and d in sym_c]
        outer = [d for d in sym_a if d not in sym_b and d in sym_c]
        inner = [d for d in sym_a if d in sym_b and d not in sym_c]
        illegal = [d for d in sym_a if d not in sym_b and d not in sym_c]
        if illegal:
            raise ValueError(f"einsum labels {illegal} ({expr_a}) "\
                             f"not present in {expr_b} or {expr_c}")
        return _einsum.uniq(batch), _einsum.uniq(outer), _einsum.uniq(inner)


    def replace_subscript(expr, arrays):
        # replace array indexing by Indexed()
        indexed = re.findall('([_a-zA-Z][_a-zA-Z0-9]*)\[([_a-z]*)\]', expr)
        for x in indexed:
            arrays.append(x[0])
            expr = expr.replace(f'{x[0]}[{x[1]}]', f'Indexed({x[0]},{x[1]})')
        return expr


    def parse_expr(expr, arrays):
        # extract symbols
        sym = []
        i = 0
        while i < len(expr):
            d = expr[i]
            if d == '(':
                size = expr[i:].find(')')
                d = expr[i : i + size + 1]
                d = _einsum.replace_subscript(d, arrays)
                sym.append(parse_expr(d))
                i += size + 1
            else:
                sym.append(parse_expr(d))
                i += 1
        return sym
  
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


    ############################
    ## Compilation
    ############################

    class instance:

        locks = None
        kernel_cache = dict()

        def __init__(self, einsum, dtype, stride_a, stride_b, stride_c, shape_a, shape_b, shape_c, arrays, mask):
            # parse symbols
            expr_a, expr_bc = einsum.split(",")
            expr_b, expr_c  = expr_bc.split("->")
            subscripted = []
            sym_a = _einsum.parse_expr(expr_a, subscripted)
            sym_b = _einsum.parse_expr(expr_b, subscripted)
            sym_c = _einsum.parse_expr(expr_c, subscripted)
            # parse axes
            axes_b, axes_m, axes_k = _einsum.parse_axes(sym_a, sym_b, sym_c, subscripted)
            _, axes_n, _           = _einsum.parse_axes(sym_b, sym_a, sym_c, subscripted)
            axes = axes_b + axes_m + axes_n + axes_k
            # check dimensions
            dims_a  = dict(zip(sym_a, shape_a))
            dims_b  = dict(zip(sym_b, shape_b))
            dims_c  = dict(zip(sym_c, shape_c))
            for axes in [axes_b, axes_k]:
                for d in axes:
                    dim_a = dims_a[d] if d in sym_a else None
                    dim_b = dims_b[d] if d in sym_b else None
                    if dim_a and dim_b and dim_a != dim_b:
                        raise ValueError(f'incompatible dimension {d}'
                                        f' (a: {dim_a}; b: {dim_b})')
            dims = dict()
            dims.update(dims_a)
            dims.update(dims_b)
            dims.update(dims_c)
            # look-up tables
            TK = 16 if dtype == triton.fw.torch.float16 else 8
            arrays = [(x, arrays[x]) for x in subscripted]
            delta_a, lut_mode_a = _einsum.make_delta(axes_k, TK, stride_a, dims, sym_a, arrays)
            delta_b, lut_mode_b = _einsum.make_delta(axes_k, TK, stride_b, dims, sym_b, arrays)
            # hash for recompilation
            stride_a_multiple = max([x for x in [1, 2, 4, 8] if shape_a[-1] % x == 0])
            stride_b_multiple = max([x for x in [1, 2, 4, 8] if shape_b[-1] % x == 0])
            stride_c_multiple = max([x for x in [1, 2, 4, 8] if shape_c[-1] % x == 0])
            stride_a_last = stride_a[-1]
            stride_b_last = stride_b[-1]
            stride_c_last = stride_c[-1]
            name = f'{expr_a}_{expr_b}_{expr_c}_{lut_mode_a}_{lut_mode_b}'\
                   f'_{stride_a_multiple}_{stride_b_multiple}_{stride_c_multiple}'\
                   f'_{stride_a_last}_{stride_b_last}_{stride_c_last}'  
            # recompile if necessary
            cache = _einsum.instance.kernel_cache
            if name not in cache:
                cachesize = len(cache)
                cache[name] = _einsum.make_kernel(f'__einsum{cachesize}', 
                                                        sym_a, sym_b, sym_c, 
                                                        axes_m, axes_n, axes_k, axes_b, 
                                                        stride_a_multiple, stride_b_multiple, stride_c_multiple,
                                                        stride_a_last, stride_b_last, stride_c_last,
                                                        lut_mode_a, lut_mode_b,
                                                        delta_a, delta_b,
                                                        subscripted)
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
            B = reduce(mul, dim_b, 1)
            stride_a = list(stride_a[:-1])
            stride_b = list(stride_b[:-1])
            stride_c = list(stride_c[:-1])
            arrays = [torch.from_numpy(x).cuda() for _, x in arrays]
            alpha = 1.
            div_m = 1
            self.args = [None, None, None,
                         _einsum.instance.locks, 
                         alpha, M, N, K, div_m] +\
                         dim_m + dim_n +  dim_k + dim_b +\
                         stride_a + stride_b + stride_c
            # LUT for A
            if lut_mode_a == _einsum.LUT_MODE.SCALAR:
                self.args += [delta_a[TK], delta_a[0]]
            if lut_mode_a == _einsum.LUT_MODE.DRAM:
                self.args += [torch.from_numpy(delta_a).cuda()]
            # LUT for B
            if lut_mode_b == _einsum.LUT_MODE.SCALAR:
                self.args += [delta_b[TK], delta_b[0]]
            if lut_mode_b == _einsum.LUT_MODE.DRAM:
                self.args += [torch.from_numpy(delta_b).cuda()]
            # Einsum dependents
            self.args += arrays
            self.grid = lambda opt: [triton.cdiv(M, opt.d('TM')) * 
                                     triton.cdiv(N, opt.d('TN')),
                                     triton.cdiv(B, opt.d('TB')),
                                     opt.d('TZ')]
            # position of dynamic arguments
            self.pos_a = 0
            self.pos_b = 1
            self.pos_c = 2
            # pre-processor macros
            TM = [16] + [x for x in [32, 64, 128] if x <= M]
            TN = [16] + [x for x in [32, 64, 128] if x <= N]
            TB = [x for x in [1, 2, 4] if x <= B]
            MAX_GZ = K // 2048
            MIN_GM = M // max(TM)
            MIN_GN = N // max(TN)
            MIN_GB = B // max(TB)
            TZ = [x for x in [1, 2, 4, 8, 16, 32] \
                    if x < MAX_GZ and x*MIN_GM*MIN_GN*MIN_GB < 256]
            TZ = [1] if not TZ else [TZ[-1], TZ[-1]*2]
            TM, TN, TB, TZ = 64, 64, 1, 1
            self.macros = {  'TM': TM, 'TN': TN, 'TB': TB, 'TK': TK, 'TZ': TZ, 'TYPE': dtype}
            if mask:
                self.macros['MASK'] = '{0:#0{1}x}'.format(mask, 10)
            # information on compute
            self.dtype = dtype
            self.flops = 2 * B * M * N * K
            self.sym_a = sym_a
            self.sym_b = sym_b
            self.sym_c = sym_c
            # save equivalent mat-mul dimensions
            self.matmul_B = B
            self.matmul_M = M
            self.matmul_N = N
            self.matmul_K = K
                    
        def run(self, a, b, c, bench):
            self.args[self.pos_a] = a
            self.args[self.pos_b] = b
            self.args[self.pos_c] = c
            return self.kernel(*self.args, grid=self.grid, bench=bench, defines=self.macros)




    ############################
    ## Forward
    ############################

    instance_cache = dict()

    @staticmethod
    def forward(ctx, einsum, a, b, shape_c, **kwargs):
        bench = kwargs['bench'] if 'bench' in kwargs else False
        arrays = kwargs['arrays'] if 'arrays' in kwargs else dict()
        mask = kwargs['mask'] if 'mask' in kwargs else None
        # allocate output
        dtype = a.dtype
        c = triton.empty(shape_c, dtype=dtype)
        # compile einsum instance
        cache = _einsum.instance_cache
        key = (einsum, dtype, 
               a.stride(), b.stride(), c.stride(), 
               a.shape, b.shape, c.shape,
               mask)
        if key not in cache:
            cache[key] = _einsum.instance(einsum, dtype, 
                                          a.stride(), b.stride(), c.stride(),
                                          a.shape, b.shape, c.shape, arrays,
                                          mask)
        instance = cache[key]
        speed = instance.run(a, b, c, bench)
        # save information in context
        ctx.flops = instance.flops
        ctx.sym_a = instance.sym_a
        ctx.sym_b = instance.sym_b
        ctx.sym_c = instance.sym_c
        ctx.matmul_B = instance.matmul_B
        ctx.matmul_M = instance.matmul_M
        ctx.matmul_N = instance.matmul_N
        ctx.matmul_K = instance.matmul_K
        ctx.bench = bench
        ctx.forward_ms = speed
        ctx.mask = mask
        ctx.save_for_backward(a, b)
        return c

    ############################
    ## Backward
    ############################

    @staticmethod
    def sym_invert(sym_c, sym_x, prefix, renamed, inverse):
        for i, expr in enumerate(sym_x):
           if expr.is_symbol:
               continue
           sc = [x for x in expr.free_symbols if x in sym_c][0]
           sx = sp.symbols(f'{prefix}{i}')
           renamed[expr] = sx
           inverse[sc] = sp.solve(sp.Eq(expr, sx), sc)[0]

    @staticmethod
    def sym_to_expr(sym):
        res = [f'({x})' for x in sym]
        res = ''.join(res)
        return res

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        sym_a = ctx.sym_a
        sym_b = ctx.sym_b
        sym_c = ctx.sym_c
        inverse = dict()
        renamed = dict()
        _einsum.sym_invert(sym_c, sym_a, 'a', renamed, inverse)
        _einsum.sym_invert(sym_c, sym_b, 'b', renamed, inverse)
        sym_a = [renamed[x] if x in renamed else x for x in sym_a]
        sym_b = [renamed[x] if x in renamed else x for x in sym_b]
        sym_c = [inverse[x] if x in inverse else x for x in sym_c]
        expr_a =  _einsum.sym_to_expr(sym_a)
        expr_b =  _einsum.sym_to_expr(sym_b)
        expr_c =  _einsum.sym_to_expr(sym_c)
        da = einsum(f'{expr_c},{expr_b}->{expr_a}', dy, b, a.shape, mask=ctx.mask)
        db = einsum(f'{expr_a},{expr_c}->{expr_b}', a, dy, b.shape, mask=ctx.mask)
        return None, da, db, None, None


einsum = _einsum.apply