import json


def ty_to_torch_cpp(ty):
    if ty[0] == "*":
        return "at::Tensor"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


class TorchExtTemplates:
    PREFIX = """
#include <torch/extension.h>
#include<iostream>
#include <filesystem>
#include<pybind11/functional.h>
#include<pybind11/stl.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_DRIVER_CHECK(EXPR)                    do {                                                   CUresult code = EXPR;                              const char *msg;                                   cuGetErrorString(code, &msg);                      if (code != CUDA_SUCCESS) {                            throw std::runtime_error(                              std::string("CUDA driver error: ") +               std::string(msg));                         }                                              } while (0);

static inline CUfunction loadKernel(
        std::string filePath,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    
    auto device = c10::cuda::current_device();
    int shared_optin;
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(
        &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));

    if (sharedMemBytes > 49152 && shared_optin > 49152) {
        CUDA_DRIVER_CHECK(cuFuncSetCacheConfig(func, CU_FUNC_CACHE_PREFER_SHARED));
        int shared_total, shared_static;
        CUDA_DRIVER_CHECK(cuDeviceGetAttribute(
            &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            device));
        CUDA_DRIVER_CHECK(cuFuncGetAttribute(
            &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_optin - shared_static));
    }
    
    return func;
}

static inline void launchKernel(
        CUfunction func,
        uint32_t gridX,
        uint32_t gridY,
        uint32_t gridZ,
        uint32_t numWarps,
        uint32_t sharedMemBytes,
        void* args[],
        cudaStream_t stream) {
    CUDA_DRIVER_CHECK(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
    ));
}
"""
    KERNEL_INTERFACE = """
    class {KERNEL_INTERFACE_NAME} {{
    private: 
        const std::string cubinPath;
    public:
        {KERNEL_INTERFACE_NAME}(const std::string &cubinPath) : cubinPath(cubinPath) {{}}
    """

    KERNEL_LAMBDA = """
    std::function<void(at::Tensor, at::Tensor, at::Tensor, int)> operator[](const std::vector<int> &grid) {{
        return [this, grid](at::Tensor x, at::Tensor y, at::Tensor out, int n_elements) {{ this->run(grid, x, y, out, n_elements); }};
    }}

}};
"""
    KERNEL_HANDLE = """
    static CUfunction {KERNEL_NAME}_0 = nullptr;
    """

    __CUBIN_LOADER = """
    static inline CUfunction loadCubin(
        const std::string &funcName,
        uint32_t sharedMemBytes) {

    CUmodule mod;
    CUfunction func;
    void *bin = (void *)&CUBIN_NAME;
    CUDA_DRIVER_CHECK(cuModuleLoadData(&mod, &bin));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}
    """
    CUBIN_LOADER = """
    // globals
#define CUBIN_NAME {KERNEL_NAME}_cubin
CUmodule {KERNEL_NAME}_mod = NULL;
CUfunction {KERNEL_NAME}_func = NULL;
unsigned char CUBIN_NAME[{BIN_SIZE}] = {{ {BIN_DATA} }};

void unload_{KERNEL_NAME}(void) {{
    CUDA_DRIVER_CHECK(cuModuleUnload({KERNEL_NAME}_mod));
}}

// TODO: some code duplication with `runtime/backend/cuda.c`
void load_{KERNEL_NAME}() {{
    int dev = 0;
    void *bin = (void *)&CUBIN_NAME;
    int shared = {SHARED};
    CUDA_DRIVER_CHECK(cuModuleLoadData(&{KERNEL_NAME}_mod, bin));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&{KERNEL_NAME}_func, {KERNEL_NAME}_mod, "{MANGLED_NAME}"));
    // set dynamic shared memory if necessary
    int shared_optin;
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    if (shared > 49152 && shared_optin > 49152) {{
      CUDA_DRIVER_CHECK(cuFuncSetCacheConfig({KERNEL_NAME}_func, CU_FUNC_CACHE_PREFER_SHARED));
      CUDA_DRIVER_CHECK(cuFuncSetAttribute({KERNEL_NAME}_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin));
    }}
}}
void test_cubin() {{
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUdeviceptr A, B, C;
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);
    cuStreamCreate(&stream, 0);
    load_{KERNEL_NAME}();
    cuCtxDestroy(ctx);

}}
"""
    __CUBIN_IMAGE = """
    #define CUBIN_NAME {KERNEL_NAME}_cubin
    const char CUBIN_NAME[] = {{ {BIN_DATA} }};
    """

    CUDA_CONTEXT = """
        at::cuda::CUDAGuard device_guard(0);
        cudaStream_t stream0 = at::cuda::getCurrentCUDAStream(0);
    """

    POINTER_TEMPLATE = (
        "CUdeviceptr {PREFIX}_{i} = reinterpret_cast<CUdeviceptr>({ARG}.data_ptr())"
    )

    ARG_DECL = """
        void* kernel_args[] = {{{}}};
    """

    KERNEL_LOADER_TEMPLATE = """
        if ({KERNEL_NAME}_0 == nullptr) {{
            {KERNEL_NAME}_0 = loadKernel(cubinPath, "{KERNEL_MANGLED_NAME}", {SHARED_MEM_BYTES});
        }}
    """

    GRID_DECL = """
        dim3 grid({GRID});
    """

    LAUNCH_KERNEL_TEMPLATE = """
        launchKernel(add_kernel_0, grid.x, grid.y, grid.z, {NUM_WARPS}, {SHARED_MEM_BYTES}, kernel_args, stream0);
    """

    CONST_DECL = """
        m.attr("{NAME}") = py::int_({VALUE});
    """

    STATIC_PROP_TEMPLATE = """
    .def_property_readonly_static("{NAME}", [](py::object) {{ return {VALUE}; }})
    """

    KERNEL_INTERFACE_BINDING_DOC = """
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
        m.doc() = R"(Interface to the {KERNEL_NAME} kernel.  Use {KERNEL_INTERFACE_NAME} to get a handle to the kernel.
        Construct the kernel by passing the path to the cubin file to the constructor.  The kernel can be launched by either calling the launch method or by indexing the object with a grid.
        If using launch, the first argument is the grid and the remaining arguments are the kernel arguments {KERNEL_ARGS_WITH_TYPES}.
        If using indexing, the grid should be a tuple or list of len 3 and the remaining arguments are the kernel arguments {KERNEL_ARGS_WITH_TYPES}.
        Example:
            kernel = {KERNEL_INTERFACE_NAME}(\'{KERNEL_NAME}.cubin\')
            kernel.launch((1, 1, 1), {KERNEL_ARGS})
        Equivalently
            kernel[(1, 1, 1)]({KERNEL_ARGS})
        IMPORTANT NOTES: 
            - The grid should be sized according to the same heuristics as the original kernel, as constexprs are baked into the kernel.
            E.g., if the original grid was lambda meta: (triton.cdiv(n_elements, meta[\'BLOCK_SIZE\']), 1, 1), then the grid should be (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1).
            - The kernel is launched on device 0.
        Constants are also exposed as attributes.  
        E.g., if the kernel has a constant named BLOCK_SIZE, then the constant can be accessed as kernel.BLOCK_SIZE.)";
    """

    KERNEL_INTERFACE_BINDING_PROPER = """
        py::class_<{KERNEL_INTERFACE_NAME}>(m, "{KERNEL_INTERFACE_NAME}")        
        .def(py::init<const std::string&>(), "Loads the cubin file from the given path", py::arg("cubin_path"))
        .def("__getitem__", &{KERNEL_INTERFACE_NAME}::operator[])
        .def("launch", &{KERNEL_INTERFACE_NAME}::run, "Launches the kernel with the given grid size, grid should be a tuple or list of len 3", py::arg("grid"), {PY_ARG_NAMES})\t{STATIC_PROPS};
    """

    EXTENSION_DECL = """
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
        m.def("{KERNEL_NAME}", torch::wrap_pybind_function({KERNEL_NAME}), "{KERNEL_NAME}");
    """
    BINDING_DECL = """
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("test_launch", &test_launch, "test_launch");
                
}"""


class TorchExtCodeGen:
    """Generates Torch extension for Triton Kernel

    Maintains same API as original kernel with auto-generated grid
    """

    TEMPLATES = TorchExtTemplates

    def __init__(
        self,
        *,
        kernel_name,
        metadata,
        metadata_group=None,
        var_prefix="var",
    ):
        """Generates Torch extension for Triton Kernel

        Args:
            kernel_name (str): name of original jitted kernel
            metadata (dict): metadata of jitted kernel
            var_prefix (str, optional): prefix to use for each arg passed to actual CUDA kernel. Defaults to "var".
        """
        self.kernel_name = kernel_name
        self.mangled_name = metadata.pop("name")  # metadata["name"]
        self.num_warps = metadata["num_warps"]
        self.num_ctas = metadata["num_ctas"]
        self.shared_mem = metadata["shared"]
        self.signature = metadata.pop("signature")  # metadata["signature"]
        self.arg_names = metadata.pop("arg_names")  # metadata.get("arg_names")
        self.constants = {int(k): v for k, v in metadata.pop("constants").items()}
        self.specializations = metadata.get("attrs")
        self.args = [
            self.arg_names[i]
            for i in range(len(self.arg_names))
            if not (
                (i in self.constants.keys())
                and not (i in self.specializations["equal_to_1"])
            )
        ]
        self.constexprs = {
            k: v
            for k, v in self.constants.items()
            if k not in self.specializations["equal_to_1"]
        }

        self.constant_names = [self.arg_names[i] for i in self.constexprs.keys()]
        self.constant_values = list(self.constexprs.values())
        self.metadata_group = metadata_group
        assert len(self.args) == len(self.signature)
        assert len(self.arg_names) == len(self.args) + len(self.constexprs)
        assert len(self.constant_names) == len(self.constant_values)

        self._metadata = metadata
        self.id = (
            hash(metadata.values())
            + hash(metadata_group.values() if metadata_group else 0)
            + hash(self.constexprs.values())
        )
        self.prefix = var_prefix
        self.kernel_interface_name = "".join(
            [s.title() for s in self.kernel_name.split("_") if s.lower() != "kernel"]
            + [f"Kernel_{self.id}"]
        )

    def make_kernel_decl_args(self):
        arg_names = self.args
        arg_types = list(self.signature.values())
        arg_types = [ty_to_torch_cpp(ty) for ty in arg_types]

        return ", ".join([f"{ty} {name}" for ty, name in zip(arg_types, arg_names)])

    def make_kernel_decl(
        self,
    ):
        return f"void {self.kernel_name}({self.make_kernel_decl_args()})" + "{\n"

    def make_kernel_interface_impl(self):
        return f"""
    void run(const std::vector<int> &grid, {self.make_kernel_decl_args()}) {{
        auto device = c10::cuda::current_device();
        at::cuda::CUDAGuard device_guard(device);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
        {self.make_kernel_loader()}
        {self.make_normalized_arg_assigns()}
        {self.make_kernel_args_array()};
        
        TORCH_CHECK(grid.size() == 3, "Grid must be 3-dimensional");
        
        launchKernel({self.kernel_name}_0, grid[0], grid[1], grid[2], {self.num_warps}, {self.shared_mem}, kernel_args,
               stream);
    }}
    """

    def make_kernel_lambda(self):
        return f"""
        std::function<void({self.make_kernel_decl_args()})> operator[](const std::vector<int> &grid) {{
            return [this, grid]({self.make_kernel_decl_args()}) {{ this->run(grid, {", ".join(self.args)}); }};
        }}
        """

    def make_normalized_arg_assigns(self):
        normalized_args = []

        for i, (arg, arg_ty) in enumerate(zip(self.args, self.signature.values())):
            if arg_ty[0] == "*":
                normalized_args.append(
                    self.TEMPLATES.POINTER_TEMPLATE.format(
                        i=i, ARG=arg, PREFIX=self.prefix
                    )
                )
            else:
                normalized_args.append(
                    f"{ty_to_torch_cpp(arg_ty)} {self.prefix}_{i} = {arg}"
                )
        return ";\n\t".join(normalized_args) + ";"

    def make_kernel_args_array(self):
        args = ", ".join(
            [
                f"&{self.prefix}_{i}"
                for i in range(len(self.signature))
                if i not in self.specializations["equal_to_1"]
            ]
        )
        return self.TEMPLATES.ARG_DECL.format(args)

    def make_kernel_loader(self):
        return self.TEMPLATES.KERNEL_LOADER_TEMPLATE.format(
            KERNEL_NAME=self.kernel_name,
            KERNEL_MANGLED_NAME=self.mangled_name,
            SHARED_MEM_BYTES=self.shared_mem,
        )

    def make_kernel_launcher(self):
        return self.TEMPLATES.LAUNCH_KERNEL_TEMPLATE.format(
            NUM_WARPS=self.num_warps, SHARED_MEM_BYTES=self.shared_mem
        )

    def make_constants(self):
        for name, value in zip(self.constant_names, self.constant_values):
            yield self.TEMPLATES.CONST_DECL.format(NAME=name, VALUE=value)

    def make_props(self):
        constants = [
            self.TEMPLATES.STATIC_PROP_TEMPLATE.format(
                KERNEL_INTERFACE_NAME=self.kernel_interface_name, NAME=name, VALUE=value
            )
            if isinstance(value, int)
            else self.TEMPLATES.STATIC_PROP_TEMPLATE.format(
                KERNEL_INTERFACE_NAME=self.kernel_interface_name,
                NAME=name,
                VALUE=f'"{value}"',
            )
            for name, value in zip(self.constant_names, self.constant_values)
        ]
        metadata = []

        def make_initializer_list(vs):
            return "{ " + ", ".join(str(v) for v in vs) + " }"

        for k, v in self._metadata.items():
            if k == "target":
                v = ":".join(str(i) for i in v)
                v = '"' + v + '"'
            elif isinstance(v, list):
                v = f"std::vector<int>({make_initializer_list(v)})"
            elif isinstance(v, dict):
                vs = ", ".join(
                    f'{{"{k}", {make_initializer_list(v)}}}' for k, v in v.items()
                )
                v = f"std::map<std::string, std::vector<int>>({{{vs}}})"
            elif not v:
                v = "nullptr"
            elif isinstance(v, str):
                v = f'"{v}"'
            else:
                v = json.dumps(v)
            metadata.append(
                self.TEMPLATES.STATIC_PROP_TEMPLATE.format(
                    KERNEL_INTERFACE_NAME=self.kernel_interface_name,
                    NAME=k.upper(),
                    VALUE=v,
                )
            )
        if self.metadata_group:
            for k, v in self.metadata_group.items():
                metadata.append(
                    self.TEMPLATES.STATIC_PROP_TEMPLATE.format(
                        KERNEL_INTERFACE_NAME=self.kernel_interface_name,
                        NAME=k.upper(),
                        VALUE=f'"{v}"',
                    )
                )
        return constants + metadata

    def make_binding(self):
        # binding = self.TEMPLATES.EXTENSION_DECL.format(KERNEL_NAME=self.kernel_name)
        py_arg_names = [f'py::arg("{name}")' for name in self.args]

        kernel_args = ", ".join(self.args)
        kernel_args_with_types = ", ".join(
            [f"{name}:{ty}" for name, ty in zip(self.args, self.signature.values())]
        )

        props = "\n".join(self.make_props())
        binding_doc = self.TEMPLATES.KERNEL_INTERFACE_BINDING_DOC.format(
            KERNEL_INTERFACE_NAME=self.kernel_interface_name,
            KERNEL_NAME=self.kernel_name,
            KERNEL_ARGS=kernel_args,
            KERNEL_ARGS_WITH_TYPES=kernel_args_with_types,
            PY_ARG_NAMES=", ".join(py_arg_names),
            STATIC_PROPS=props,
        )
        binding_cls = self.TEMPLATES.KERNEL_INTERFACE_BINDING_PROPER.format(
            KERNEL_INTERFACE_NAME=self.kernel_interface_name,
            KERNEL_NAME=self.kernel_name,
            KERNEL_ARGS=kernel_args,
            PY_ARG_NAMES=", ".join(py_arg_names),
            STATIC_PROPS=props,
        )
        return "\n".join([binding_doc, binding_cls]) + "\n}"

    def generate(self, save_dir=None):
        src = ""
        src += self.TEMPLATES.PREFIX
        # src += self.make_cubin_image()
        src += self.TEMPLATES.KERNEL_HANDLE.format(KERNEL_NAME=self.kernel_name)
        src += self.TEMPLATES.KERNEL_INTERFACE.format(
            KERNEL_INTERFACE_NAME=self.kernel_interface_name,
            KERNEL_NAME=self.kernel_name,
            KERNEL_MANGLED_NAME=self.mangled_name,
            NUM_WARPS=self.num_warps,
            SHARED_MEM_BYTES=self.shared_mem,
        )
        src += self.make_kernel_interface_impl()
        src += self.make_kernel_lambda()
        src += "};\n"
        src += self.make_binding()
        if save_dir:
            with open(save_dir, "w") as f:
                f.write(src)
        return src
