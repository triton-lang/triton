"""
Numba compatibility layer for Triton kernels.

Enables launching Triton kernels from @numba.njit functions by generating
a per-signature C trampoline that calls cuLaunchKernelEx directly,
bypassing the Python C API launcher.

Usage:
    numba_add = add_kernel.as_numba_kernel(
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32', 'n': 'i32'},
        constexprs={'BLOCK_SIZE': 1024})

    # Extract the @njit launch function into a module-level variable
    launch_add = numba_add.launch

    @numba.njit
    def f(x_ptr, y_ptr, out_ptr, n, stream):
        grid = (n + 1023) // 1024
        launch_add(grid, 1, 1, stream, x_ptr, y_ptr, out_ptr, n)

Note: The launch function must be extracted into a variable (e.g. ``launch_add = numba_add.launch``)
before being used inside ``@numba.njit``. Numba cannot resolve attribute access on custom Python
objects within compiled code.

V1 limitations:
    - No dynamic specialization (exact signature + constexprs required upfront)
    - No scratch memory (asserts global_scratch_size == 0 and profile_scratch_size == 0)
    - No launch hooks
    - Stream must be passed explicitly as uint64
    - NVIDIA only (CUDA driver API)
"""

import ctypes
import hashlib

import numba

from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.runtime.build import compile_module_from_src

# ---------------------------------------------------------------------------
# Type mapping: Triton type strings → (C type string, ctypes type)
# ---------------------------------------------------------------------------

_TRITON_TYPE_MAP = {
    'i1': ('int8_t', ctypes.c_int8),
    'i8': ('int8_t', ctypes.c_int8),
    'i16': ('int16_t', ctypes.c_int16),
    'i32': ('int32_t', ctypes.c_int32),
    'i64': ('int64_t', ctypes.c_int64),
    'u1': ('uint8_t', ctypes.c_uint8),
    'u8': ('uint8_t', ctypes.c_uint8),
    'u16': ('uint16_t', ctypes.c_uint16),
    'u32': ('uint32_t', ctypes.c_uint32),
    'u64': ('uint64_t', ctypes.c_uint64),
    'fp16': ('uint16_t', ctypes.c_uint16),  # passed as bits
    'bf16': ('uint16_t', ctypes.c_uint16),
    'fp32': ('float', ctypes.c_float),
    'fp64': ('double', ctypes.c_double),
}


def _resolve_triton_type(ty_str):
    """Resolve a Triton type string to (c_type_str, ctypes_type).

    Pointer types ('*fp32', '*fp64', etc.) map to uint64_t / c_uint64.
    """
    if ty_str.startswith('*'):
        return ('uint64_t', ctypes.c_uint64)
    if ty_str not in _TRITON_TYPE_MAP:
        raise ValueError(f"Unsupported Triton type: {ty_str!r}")
    return _TRITON_TYPE_MAP[ty_str]


# ---------------------------------------------------------------------------
# C trampoline generation
# ---------------------------------------------------------------------------

def _generate_launch_trampoline_src(arg_c_types, func_name):
    """Generate C source for a launch trampoline.

    Parameters
    ----------
    arg_c_types : list of (c_type_str, is_pointer)
        Each element is (C type string, bool indicating if original Triton type was a pointer).
    func_name : str
        Name for the generated function and Python module.

    Returns
    -------
    str
        Complete C source file.
    """
    # Build the function parameter list
    # Fixed params: gridX, gridY, gridZ, num_warps, num_ctas, coop, pdl, shared_memory, stream, function
    # Then: arg0, arg1, ..., argN
    fixed_params = [
        ('int', 'gridX'),
        ('int', 'gridY'),
        ('int', 'gridZ'),
        ('int', 'num_warps'),
        ('int', 'num_ctas'),
        ('int', 'coop'),
        ('int', 'pdl'),
        ('int', 'shared_memory'),
        ('uint64_t', 'stream'),
        ('uint64_t', 'function'),
    ]

    arg_params = []
    for i, (c_type, _is_ptr) in enumerate(arg_c_types):
        arg_params.append((c_type, f'arg{i}'))

    all_params = fixed_params + arg_params
    param_str = ',\n    '.join(f'{t} {n}' for t, n in all_params)

    # Build the body: declare local vars, build params array
    n_args = len(arg_c_types)
    body_lines = []

    # Early exit if grid is empty
    body_lines.append('    if (gridX * gridY * gridZ <= 0) return 0;')
    body_lines.append('')

    # Declare local variables for each arg and build params array
    for i, (c_type, is_ptr) in enumerate(arg_c_types):
        if is_ptr:
            body_lines.append(f'    CUdeviceptr p{i} = (CUdeviceptr)arg{i};')
        else:
            body_lines.append(f'    {c_type} v{i} = arg{i};')

    # Two scratch pointers (NULL) appended to params
    body_lines.append('    CUdeviceptr scratch0 = 0, scratch1 = 0;')
    body_lines.append('')

    # Build void* params array
    n_total = n_args + 2  # args + 2 scratch
    param_ptrs = []
    for i, (_c_type, is_ptr) in enumerate(arg_c_types):
        if is_ptr:
            param_ptrs.append(f'&p{i}')
        else:
            param_ptrs.append(f'&v{i}')
    param_ptrs.append('&scratch0')
    param_ptrs.append('&scratch1')

    body_lines.append(f'    void *params[{n_total}] = {{{", ".join(param_ptrs)}}};')
    body_lines.append('')

    # Build launch config (mirrors driver.c _launch)
    body_lines.append('    CUlaunchAttribute launchAttr[4];')
    body_lines.append('    int num_attrs = 0;')
    body_lines.append('')
    body_lines.append('    if (pdl != 0) {')
    body_lines.append('        CUlaunchAttribute a = {0};')
    body_lines.append('        a.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;')
    body_lines.append('        a.value.programmaticStreamSerializationAllowed = 1;')
    body_lines.append('        launchAttr[num_attrs++] = a;')
    body_lines.append('    }')
    body_lines.append('')
    body_lines.append('    if (coop != 0) {')
    body_lines.append('        CUlaunchAttribute a = {0};')
    body_lines.append('        a.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;')
    body_lines.append('        a.value.cooperative = 1;')
    body_lines.append('        launchAttr[num_attrs++] = a;')
    body_lines.append('    }')
    body_lines.append('')
    body_lines.append('    if (num_ctas != 1) {')
    body_lines.append('        CUlaunchAttribute clusterDim = {0};')
    body_lines.append('        clusterDim.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;')
    body_lines.append('        clusterDim.value.clusterDim.x = num_ctas;')
    body_lines.append('        clusterDim.value.clusterDim.y = 1;')
    body_lines.append('        clusterDim.value.clusterDim.z = 1;')
    body_lines.append('        launchAttr[num_attrs++] = clusterDim;')
    body_lines.append('')
    body_lines.append('        CUlaunchAttribute schedPolicy = {0};')
    body_lines.append('        schedPolicy.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;')
    body_lines.append('        schedPolicy.value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;')
    body_lines.append('        launchAttr[num_attrs++] = schedPolicy;')
    body_lines.append('    }')
    body_lines.append('')
    body_lines.append('    if (num_ctas == 16) {')
    body_lines.append('        cuFuncSetAttribute(')
    body_lines.append('            (CUfunction)function,')
    body_lines.append('            CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);')
    body_lines.append('    }')
    body_lines.append('')
    body_lines.append('    CUlaunchConfig config;')
    body_lines.append('    config.gridDimX = gridX * num_ctas;')
    body_lines.append('    config.gridDimY = gridY;')
    body_lines.append('    config.gridDimZ = gridZ;')
    body_lines.append('    config.blockDimX = 32 * num_warps;')
    body_lines.append('    config.blockDimY = 1;')
    body_lines.append('    config.blockDimZ = 1;')
    body_lines.append('    config.sharedMemBytes = shared_memory;')
    body_lines.append('    config.hStream = (CUstream)stream;')
    body_lines.append('    config.attrs = launchAttr;')
    body_lines.append('    config.numAttrs = num_attrs;')
    body_lines.append('')
    body_lines.append('    cuLaunchKernelEx_t launch_fn = get_launch_handle();')
    body_lines.append('    if (!launch_fn) return -1;')
    body_lines.append('    return (int)launch_fn(&config, (CUfunction)function, params, 0);')

    body = '\n'.join(body_lines)

    src = f"""\
#include "cuda.h"
#include <dlfcn.h>
#include <stdint.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig *config,
                                       CUfunction f, void **kernelParams,
                                       void **extra);

static cuLaunchKernelEx_t get_launch_handle(void) {{
    static cuLaunchKernelEx_t fn = NULL;
    if (!fn) {{
        void *lib = dlopen("libcuda.so.1", RTLD_LAZY);
        if (lib) fn = (cuLaunchKernelEx_t)dlsym(lib, "cuLaunchKernelEx");
    }}
    return fn;
}}

int {func_name}(
    {param_str}
) {{
{body}
}}

static PyObject *get_fn_ptr(PyObject *self, PyObject *args) {{
    return PyLong_FromVoidPtr((void *){func_name});
}}

static PyMethodDef methods[] = {{
    {{"get_fn_ptr", get_fn_ptr, METH_NOARGS, ""}},
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef moddef = {{
    PyModuleDef_HEAD_INIT, "{func_name}", NULL, -1, methods
}};

PyMODINIT_FUNC PyInit_{func_name}(void) {{
    return PyModule_Create(&moddef);
}}
"""
    return src


# ---------------------------------------------------------------------------
# Trampoline compilation
# ---------------------------------------------------------------------------

def _get_or_compile_trampoline(arg_types):
    """Compile (or load from cache) a C trampoline for the given arg types.

    Parameters
    ----------
    arg_types : list of str
        Triton type strings (e.g. ['*fp32', '*fp32', '*fp32', 'i32']).

    Returns
    -------
    ctypes function pointer
        A callable C function taking (gridX, gridY, gridZ, num_warps, num_ctas,
        coop, pdl, shared_memory, stream, function, arg0, ..., argN) → int.
    """
    from triton.backends.nvidia.driver import include_dirs, library_dirs, libraries

    # Resolve types
    resolved = []
    for ty in arg_types:
        c_type_str, ct = _resolve_triton_type(ty)
        is_ptr = ty.startswith('*')
        resolved.append((c_type_str, is_ptr, ct))

    # Generate a unique name based on the type signature
    sig_key = ','.join(arg_types)
    sig_hash = hashlib.sha256(sig_key.encode()).hexdigest()[:16]
    func_name = f'numba_launch_{sig_hash}'

    arg_c_types = [(c_type_str, is_ptr) for c_type_str, is_ptr, _ in resolved]
    src = _generate_launch_trampoline_src(arg_c_types, func_name)

    mod = compile_module_from_src(
        src=src,
        name=func_name,
        library_dirs=library_dirs(),
        include_dirs=include_dirs,
        libraries=libraries,
    )

    # Get the raw function pointer
    fn_ptr = mod.get_fn_ptr()

    # Build the ctypes function type
    # Fixed params: int*8 (gridX..shared_memory) + uint64*2 (stream, function) + per-arg types
    fixed_ctypes = [
        ctypes.c_int,     # gridX
        ctypes.c_int,     # gridY
        ctypes.c_int,     # gridZ
        ctypes.c_int,     # num_warps
        ctypes.c_int,     # num_ctas
        ctypes.c_int,     # coop
        ctypes.c_int,     # pdl
        ctypes.c_int,     # shared_memory
        ctypes.c_uint64,  # stream
        ctypes.c_uint64,  # function
    ]
    arg_ctypes = [ct for _, _, ct in resolved]
    all_ctypes = fixed_ctypes + arg_ctypes

    cfunc_type = ctypes.CFUNCTYPE(ctypes.c_int, *all_ctypes)
    cfunc = cfunc_type(fn_ptr)

    # Prevent garbage collection of the underlying module
    cfunc._mod = mod

    return cfunc


# ---------------------------------------------------------------------------
# njit launcher generation
# ---------------------------------------------------------------------------

def _make_njit_launcher(cfunc, function_handle, num_warps, num_ctas,
                        shared_mem, coop, pdl, arg_names):
    """Generate an @njit function that calls the C trampoline.

    Parameters
    ----------
    cfunc : ctypes function
        The compiled C trampoline.
    function_handle : int
        CUfunction handle (uint64).
    num_warps : int
        Number of warps per CTA.
    num_ctas : int
        Number of CTAs per cluster.
    shared_mem : int
        Shared memory in bytes.
    coop : int
        Whether to use cooperative grid launch.
    pdl : int
        Whether to use programmatic dependent launch.
    arg_names : list of str
        Names of the kernel arguments (for the generated function signature).

    Returns
    -------
    numba.core.registry.CPUDispatcher
        An @njit function with signature:
        launch(gridX, gridY, gridZ, stream, arg0, arg1, ..., argN)
    """
    # We generate the function source as a string and exec it, because
    # numba.njit doesn't support *args — we need explicit argument names.
    arg_list = ', '.join(arg_names)
    if arg_list:
        sig_args = f'gridX, gridY, gridZ, stream, {arg_list}'
        call_args = (f'gridX, gridY, gridZ, '
                     f'num_warps, num_ctas, coop, pdl, shared_mem, '
                     f'stream, function_handle, {arg_list}')
    else:
        sig_args = 'gridX, gridY, gridZ, stream'
        call_args = (f'gridX, gridY, gridZ, '
                     f'num_warps, num_ctas, coop, pdl, shared_mem, '
                     f'stream, function_handle')

    src = f"""\
def _launch({sig_args}):
    _cfunc({call_args})
"""

    # The closure namespace — these become compile-time constants for numba
    namespace = {
        '_cfunc': cfunc,
        'num_warps': num_warps,
        'num_ctas': num_ctas,
        'coop': coop,
        'pdl': pdl,
        'shared_mem': shared_mem,
        'function_handle': function_handle,
    }

    exec(src, namespace)
    launch_fn = namespace['_launch']
    return numba.njit(launch_fn)


# ---------------------------------------------------------------------------
# NumbaTritonKernel
# ---------------------------------------------------------------------------

class NumbaTritonKernel:
    """A Triton kernel compiled for a fixed signature, callable from @numba.njit.

    Parameters
    ----------
    jit_fn : triton.runtime.jit.JITFunction
        The @triton.jit decorated function.
    signature : dict
        Mapping from parameter names to Triton type strings.
        Example: {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'out_ptr': '*fp32', 'n': 'i32'}
    constexprs : dict
        Mapping from parameter names to constant values.
        Example: {'BLOCK_SIZE': 1024}
    """

    def __init__(self, jit_fn, signature, constexprs):
        # 1. Compile the Triton kernel
        src = ASTSource(jit_fn, signature, constexprs)
        compiled_kernel = triton_compile(src)

        # 2. Initialize handles to get CUfunction
        compiled_kernel._init_handles()

        # 3. Extract metadata
        metadata = compiled_kernel.metadata
        num_warps = metadata.num_warps
        num_ctas = getattr(metadata, 'num_ctas', 1)
        shared = metadata.shared
        coop = int(metadata.launch_cooperative_grid)
        pdl = int(metadata.launch_pdl)

        # 4. V1 limitation: no scratch memory
        global_scratch = getattr(metadata, 'global_scratch_size', 0)
        profile_scratch = getattr(metadata, 'profile_scratch_size', 0)
        if global_scratch != 0:
            raise NotImplementedError(
                f"NumbaTritonKernel does not support global scratch memory "
                f"(kernel requires {global_scratch} bytes)")
        if profile_scratch != 0:
            raise NotImplementedError(
                f"NumbaTritonKernel does not support profile scratch memory "
                f"(kernel requires {profile_scratch} bytes)")

        # 5. Get CUfunction handle as integer
        function_handle = compiled_kernel.function
        if not isinstance(function_handle, int):
            function_handle = int(function_handle)

        # 6. Build the list of arg types (excluding constexprs)
        # The signature dict keys are the parameter names in order;
        # constexpr params are not included in the launch signature.
        arg_names = []
        arg_types = []
        for name, ty in signature.items():
            if name not in constexprs:
                arg_names.append(name)
                arg_types.append(ty)

        # 7. Compile the C trampoline
        cfunc = _get_or_compile_trampoline(arg_types)

        # 8. Generate the @njit launch function
        self._njit_launch = _make_njit_launcher(
            cfunc=cfunc,
            function_handle=function_handle,
            num_warps=num_warps,
            num_ctas=num_ctas,
            shared_mem=shared,
            coop=coop,
            pdl=pdl,
            arg_names=arg_names,
        )

        # Keep references to prevent GC
        self._compiled_kernel = compiled_kernel
        self._cfunc = cfunc

    @property
    def launch(self):
        """The @njit function to call from within numba-compiled code.

        Signature: launch(gridX, gridY, gridZ, stream, arg0, arg1, ..., argN)

        All arguments must be scalar C types (int32, int64, uint64 for pointers, etc.).
        Stream must be passed as a uint64 (e.g., torch.cuda.current_stream().cuda_stream).
        """
        return self._njit_launch
