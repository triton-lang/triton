// Python bindings for the Apple MPS Triton backend.
// Exposes MLIR passes to Python compiler pipeline.

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "TritonAppleGPUTransforms/Passes.h"
#include "TritonAppleGPUToLLVM/Passes.h"
#include "mlir/Pass/PassManager.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Pass bindings for make_ttgir stage
void init_triton_apple_passes_ttgpuir(py::module &&m) {
    using namespace mlir::triton;

    m.def("add_accelerate_matmul",
        [](mlir::PassManager &pm) {
            pm.addPass(applegpu::createAccelerateAppleMatmulPass());
        },
        "Rewrite tt.dot ops to use AppleMmaEncoding (simdgroup_multiply_accumulate)");

    m.def("add_to_llvmir",
        [](mlir::PassManager &pm) {
            pm.addPass(applegpu::createConvertTritonAppleGPUToLLVMPass());
        },
        "Lower TritonGPU IR with AppleMmaEncoding to LLVM IR with simdgroup intrinsics");
}

// Dialect registration
void init_triton_apple_dialect(py::module &&m) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::applegpu::TritonAppleGPUDialect>();
    // expose for use in triton IR contexts
    m.def("register_dialect", [registry = std::move(registry)](mlir::MLIRContext &ctx) mutable {
        ctx.appendDialectRegistry(registry);
    });
}

// Metal compilation: LLVM IR → metallib via MetalASM or xcrun
void init_triton_apple_metal(py::module &&m) {
    m.def("compile_metal_ir",
        [](const std::string &llvmIR, const std::string &arch) -> py::bytes {
            // TODO: call MetalASM in-process OR xcrun metal-as + metallib
            // Returns metallib bytes ready for MTLDevice.makeLibrary(data:)
            throw std::runtime_error(
                "compile_metal_ir: MetalASM integration not yet implemented");
        },
        py::arg("llvm_ir"), py::arg("arch") = "apple_m1");

    m.def("load_metallib",
        [](py::bytes metallib) -> uint64_t {
            // TODO: call MTLDevice.makeLibrary(data:) via ObjC runtime
            // Returns opaque handle to MTLComputePipelineState
            throw std::runtime_error(
                "load_metallib: Metal runtime integration not yet implemented");
        });
}

PYBIND11_MODULE(triton_apple, m) {
    m.doc() = "Apple MPS backend for Triton";

    init_triton_apple_passes_ttgpuir(m.def_submodule("passes")
                                      .def_submodule("ttgpuir"));
    init_triton_apple_dialect(m.def_submodule("dialect"));
    init_triton_apple_metal(m.def_submodule("metal"));
}
