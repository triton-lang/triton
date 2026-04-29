#include "TritonMetalGPUToLLVM/MetalKernelArgs.h"
#include "TritonMetalGPUToLLVM/Passes.h"
#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace llvm;

namespace {
const char *const metalTargetTriple = "metal";

void init_triton_metal_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir", [](mlir::PassManager &pm, const std::string &arch) {
    pm.addPass(mlir::triton::createConvertTritonMetalGPUToLLVMPass(arch));
  });
  // m.def("add_accelerate_matmul", [](mlir::PassManager &pm) {
  //   pm.addPass(mlir::createTritonMetalGPUAccelerateMatmul());
  // });
  m.def("add_inject_tensor_stride_args", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonMetalGPUInjectTensorStrideArgs());
  });
  m.def("add_allocate_smem_for_simdgroup_matmul", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonMetalGPUAllocateSmemForSimdgroupMatmul());
  });
}

void addAirKernelMetadata(llvm::Module *mod) {
  auto &ctx = mod->getContext();
  auto *airKernelMD = mod->getOrInsertNamedMetadata("air.kernel");

  for (auto &func : mod->functions()) {
    if (func.isDeclaration() ||
        func.getLinkage() != llvm::GlobalValue::ExternalLinkage)
      continue;

    // build arg metadata
    SmallVector<Metadata *> allArgMetadata;
    unsigned numArgs = func.arg_size();
    unsigned numUserArgs = numArgs - mlir::triton::metal::kNumExtraArgs;

    // loop through args and create arg metadata
    for (unsigned i = 0; i < numArgs; i++) {
      llvm::Argument &arg = *std::next(func.arg_begin(), i);
      SmallVector<Metadata *> argMetadata;

      // arg idx
      argMetadata.push_back(
          ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), i)));

      if (i < numUserArgs || i == numUserArgs || i == numUserArgs + 1) {
        // buffer arg (user args + scratch ptrs)
        bool isScalar = arg.getDereferenceableBytes() > 0;

        argMetadata.push_back(MDString::get(ctx, "air.buffer"));

        // scalar args passed as device buffers need extra metadata
        if (isScalar) {
          argMetadata.push_back(MDString::get(ctx, "air.buffer_size"));
          argMetadata.push_back(ConstantAsMetadata::get(
              ConstantInt::get(Type::getInt32Ty(ctx), 4)));
        }

        argMetadata.push_back(MDString::get(ctx, "air.location_index"));
        argMetadata.push_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(ctx), i)));

        // TODO what do these mean?
        argMetadata.push_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(ctx), 1)));
        argMetadata.push_back(MDString::get(ctx, "air.read_write"));

        // TODO change for the scratch ptrs
        argMetadata.push_back(MDString::get(ctx, "air.address_space"));
        argMetadata.push_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(ctx), 1)));
        argMetadata.push_back(MDString::get(ctx, "air.arg_type_size"));
        argMetadata.push_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(ctx), 4)));
        argMetadata.push_back(MDString::get(ctx, "air.arg_type_align_size"));
        argMetadata.push_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(ctx), 4)));
        argMetadata.push_back(MDString::get(ctx, "air.arg_type_name"));
        argMetadata.push_back(MDString::get(ctx, "float"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_name"));
        argMetadata.push_back(MDString::get(ctx, "arg_" + std::to_string(i)));
      } else if (i == numArgs - mlir::triton::metal::kNumProgramsFromEnd) {
        argMetadata.push_back(MDString::get(ctx, "air.threadgroups_per_grid"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_type_name"));
        argMetadata.push_back(MDString::get(ctx, "uint"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_name"));
        argMetadata.push_back(MDString::get(ctx, "num_threadgroups"));
      } else if (i == numArgs - mlir::triton::metal::kThreadIdxFromEnd) {
        argMetadata.push_back(
            MDString::get(ctx, "air.thread_position_in_grid"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_type_name"));
        argMetadata.push_back(MDString::get(ctx, "uint"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_name"));
        argMetadata.push_back(MDString::get(ctx, "thread_idx"));
      } else if (i == numArgs - mlir::triton::metal::kSimdgroupIdxFromEnd) {
        argMetadata.push_back(
            MDString::get(ctx, "air.simdgroup_index_in_threadgroup"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_type_name"));
        argMetadata.push_back(MDString::get(ctx, "uint"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_name"));
        argMetadata.push_back(MDString::get(ctx, "simdgroup_idx"));
      } else { // i == numArgs - 1
        argMetadata.push_back(
            MDString::get(ctx, "air.threadgroup_position_in_grid"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_type_name"));
        argMetadata.push_back(MDString::get(ctx, "uint"));
        argMetadata.push_back(MDString::get(ctx, "air.arg_name"));
        argMetadata.push_back(MDString::get(ctx, "threadgroup_idx"));
      }

      // create new node from this arg's metadata and add to all arg metadata
      allArgMetadata.push_back(MDNode::get(ctx, argMetadata));
    }

    // !{ptr @func, !{}, !{arg descriptors}}
    SmallVector<Metadata *> kernelEntry = {
        ConstantAsMetadata::get(&func),
        MDNode::get(ctx, {}), // empty, what does this mean?
        MDNode::get(ctx, allArgMetadata),
    };
    airKernelMD->addOperand(MDNode::get(ctx, kernelEntry));
  }

  // air compile options
  {
    std::vector<std::string> compileOptions = {
        "air.compile.denorms_disable", "air.compile.fast_math_enable",
        "air.compile.framebuffer_fetch_enable"};
    auto *airCompileOptionsMD =
        mod->getOrInsertNamedMetadata("air.compile_options");

    for (std::string &option : compileOptions) {
      SmallVector<Metadata *> compileMD;
      compileMD.push_back(MDString::get(ctx, option));
      airCompileOptionsMD->addOperand(MDNode::get(ctx, compileMD));
    }
  }

  // llvm.ident
  {
    auto *llvmIdentMD = mod->getOrInsertNamedMetadata("llvm.ident");
    SmallVector<Metadata *> identMD;
    // TODO don't hardcode
    identMD.push_back(MDString::get(
        ctx, "Apple metal version 32023.404 (metalfe-32023.404)"));
    llvmIdentMD->addOperand(MDNode::get(ctx, identMD));
  }

  // air.version = !{i32 2, i32 7, i32 0}
  {
    auto *airVersionMD = mod->getOrInsertNamedMetadata("air.version");
    SmallVector<Metadata *> versionMD;
    versionMD.push_back(
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 2)));
    versionMD.push_back(
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 7)));
    versionMD.push_back(
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 0)));
    airVersionMD->addOperand(MDNode::get(ctx, versionMD));
  }

  // air.language_version = !{!"Metal", i32 3, i32 2, i32 0}
  {
    auto *airLangMD = mod->getOrInsertNamedMetadata("air.language_version");
    SmallVector<Metadata *> langMD;
    langMD.push_back(MDString::get(ctx, "Metal"));
    langMD.push_back(
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 3)));
    langMD.push_back(
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 2)));
    langMD.push_back(
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), 0)));
    airLangMD->addOperand(MDNode::get(ctx, langMD));
  }

  // air.source_file_name = !{!"filename"}
  {
    auto *airSrcMD = mod->getOrInsertNamedMetadata("air.source_file_name");
    SmallVector<Metadata *> srcMD;
    srcMD.push_back(MDString::get(ctx, mod->getSourceFileName()));
    airSrcMD->addOperand(MDNode::get(ctx, srcMD));
  }
}

} // namespace

void init_triton_metal(py::module &&m) {
  m.doc() = "Python bindings to the Metal Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_metal_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  m.def("add_kernel_metadata",
        [](llvm::Module *mod) { addAirKernelMetadata(mod); });
}
