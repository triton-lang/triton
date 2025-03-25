//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// clang-format off
#include "mlir/Dialect/Linalg/Passes.h"             // mlir::createLinalgElementwiseOpFusionPass()
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"        // LLVM::LLVMFuncOp
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // mlir::memref::createExpandStridedMetadataPass
#include "mlir/IR/BuiltinOps.h"                     // mlir::ModuleOp
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallVector.h" // llvm::SmallVector
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"    // llvm::ConstantInt
#include "llvm/IR/LLVMContext.h"  // llvm::LLVMContext

#include "passes.h"

// mlir::triton::createTritonToLinalgExperimentalPass

#include "triton/Conversion/TritonToTritonXPU/Passes.h"    // mlir::triton::createConvertTritonToTritonXPUPass
#include "triton/Conversion/TritonXPUToLLVM/Passes.h"      // mlir::triton::createConvertTritonXPUToLLVMPass
#include "triton/Dialect/TritonXPU/IR/Dialect.h"           // mlir::triton::xpu::TritonXPUDialect
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"    // mlir::createTritonXPUGM2LMPass()

#include "triton/Target/LLVMXPU/LLVMXPUToLLVMIRTranslation.h"  // registerLLVMXPUDialectTranslation
// clang-format on

namespace py = pybind11;

std::string translateLLVMIRToASM(llvm::Module &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion, bool isObject);

void init_triton_xpu_passes_conversion(py::module &&m) {
  m.def("add_convert_triton_to_tritonxpu_pass",
        [](mlir::PassManager &self, uint32_t xpu_arch, uint32_t buffer_size,
           uint32_t core_num) {
          self.addPass(mlir::triton::createConvertTritonToTritonXPUPass(
              xpu_arch, buffer_size, core_num));
        });

  m.def("add_convert_tritonxpu_to_llvm_pass",
        [](mlir::PassManager &self, uint32_t xpu_arch, uint32_t buffer_size) {
          self.addPass(mlir::triton::createConvertTritonXPUToLLVMPass(
              xpu_arch, buffer_size));
        });
}

void init_triton_xpu_passes_transform(py::module &&m) {
  // Function Pass
  m.def("add_tritonxpu_gm2lm_pass",
        [](mlir::PassManager &self, uint32_t xpu_arch, bool atomicSim) {
          self.addPass(mlir::triton::xpu::createTritonXPUCreateGM2LM(
              {xpu_arch, atomicSim}));
        });

  m.def("add_tritonxpu_legalize_pass",
        [](mlir::PassManager &self, uint32_t buffer_size, uint32_t core_num) {
          self.addPass(mlir::triton::xpu::createTritonXPULegalize(
              {buffer_size, core_num}));
        });

  m.def("add_tritonxpu_mask_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUMask());
  });

  m.def("add_tritonxpu_alloca_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUAlloca());
  });

  m.def("add_tritonxpu_dtype_convert_pass", [](mlir::PassManager &self,
                                               uint32_t xpu_arch) {
    self.addPass(mlir::triton::xpu::createTritonXPUDtypeConvert({xpu_arch}));
  });

  m.def("add_tritonxpu_loop_grid_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPULoopGrid());
  });

  m.def("add_tritonxpu_unroll_control_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUUnrollControl());
  });

  m.def("add_tritonxpu_other_sim_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUOtherSim());
  });

  // Optimization Pass
  m.def("add_tritonxpu_offset_state_pass", [](mlir::PassManager &self,
                                              bool dump_flag) {
    self.addPass(mlir::triton::xpu::createTritonXPUOffsetAnalysis({dump_flag}));
  });

  m.def("add_tritonxpu_core_tiling_pass",
        [](mlir::PassManager &self, bool dump_flag, uint32_t buffer_size) {
          self.addPass(mlir::triton::xpu::createTritonXPUCoreTiling(
              {dump_flag, buffer_size}));
        });

  m.def("add_tritonxpu_vectorize_pass", [](mlir::PassManager &self,
                                           bool dump_flag) {
    self.addPass(mlir::triton::xpu::createTritonXPUVectorize({dump_flag}));
  });

  m.def("add_tritonxpu_memory_async_pass", [](mlir::PassManager &self,
                                              bool dump_flag) {
    self.addPass(mlir::triton::xpu::createTritonXPUMemoryAsync({dump_flag}));
  });

  m.def("add_tritonxpu_interleave_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUInterleave());
  });

  m.def("add_tritonxpu_store_control_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUStoreControl());
  });
}

namespace mlir::triton::xpu {

// Describes XPU Metadata. It is used to record the XPU related meta
// information from mlir module.
struct XPUMetadata {
  int maxntidx{-1};
  bool isKernel{};
  // Free to extend with other information.
};

static void
extractXPUMetadata(mlir::ModuleOp module,
                   llvm::DenseMap<llvm::StringRef, XPUMetadata> *dic) {
  for (auto op : module.getOps<LLVM::LLVMFuncOp>()) {
    XPUMetadata meta;

    bool hasMetadata{};

    // maxntid
    if (op->hasAttr("xpu.maxntid")) {
      auto attr = op->getAttr("xpu.maxntid");
      meta.maxntidx = mlir::dyn_cast<IntegerAttr>(attr).getInt();
      hasMetadata = true;
    }

    // kernel
    if (op->hasAttr("xpu.kernel")) {
      meta.isKernel = true;
      hasMetadata = true;
    }

    if (hasMetadata)
      dic->try_emplace(op.getNameAttr().strref(), std::move(meta));
  }
}

// Add the xpu related metadata to LLVM IR.
static void amendLLVMFunc(llvm::Function *func, const XPUMetadata &metadata,
                          int xpu_arch) {
  auto *module = func->getParent();
  auto &ctx = func->getContext();
  auto targetArch = std::string("xpu") + std::to_string(xpu_arch);

  if (metadata.maxntidx > 0) {
    auto warps = llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32),
                                        llvm::APInt(32, metadata.maxntidx));

    llvm::Metadata *md_args[] = {llvm::ValueAsMetadata::get(func),
                                 llvm::MDString::get(ctx, "maxntidx"),
                                 llvm::ValueAsMetadata::get(warps)};

    module->getOrInsertNamedMetadata("xpu.annotations")
        ->addOperand(llvm::MDNode::get(ctx, md_args));
  }

  if (metadata.isKernel) {
    func->setDSOLocal(true);
    func->setCallingConv(llvm::CallingConv::XPU_KERNEL);
    llvm::Metadata *mdArgs[] = {
        llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
        llvm::ValueAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
    module->getOrInsertNamedMetadata("xpu.annotations")
        ->addOperand(llvm::MDNode::get(ctx, mdArgs));
  }

  llvm::AttrBuilder funcAttrs(ctx);
  funcAttrs.addAttribute("correctly-rounded-divide-sqrt-fp-math", "false");
  funcAttrs.addAttribute("disable-tail-calls", "false");
  funcAttrs.addAttribute("frame-pointer", "all");
  funcAttrs.addAttribute("less-precise-fpmad", "false");
  funcAttrs.addAttribute("min-legal-vector-width", "0");
  funcAttrs.addAttribute("no-infs-fp-math", "false");
  funcAttrs.addAttribute("no-jump-tables", "false");
  funcAttrs.addAttribute("no-nans-fp-math", "false");
  funcAttrs.addAttribute("no-signed-zeros-fp-math", "false");
  funcAttrs.addAttribute("no-trapping-math", "false");
  funcAttrs.addAttribute("stack-protector-buffer-size", "8");
  funcAttrs.addAttribute("target-cpu", targetArch);
  funcAttrs.addAttribute("unsafe-fp-math", "false");
  funcAttrs.addAttribute("use-soft-float", "false");
  func->addFnAttrs(funcAttrs);
}

} // namespace mlir::triton::xpu

using ret = py::return_value_policy;

void init_triton_xpu_llvm(py::module &&m) {

  m.def("get_kernel_name", [](llvm::Module &mod) {
    for (auto &F : mod) {
      if (F.getCallingConv() == llvm::CallingConv::XPU_KERNEL) {
        std::string name = F.getName().str();
        return py::str(name);
      }
    }

    auto MD = mod.getNamedMetadata("xpu.annotations");
    std::string name;
    for (auto *Op : MD->operands()) {
      if (Op->getNumOperands() != 3)
        continue;
      auto *Prop = llvm::dyn_cast<llvm::MDString>(Op->getOperand(1));
      name = Prop->getString();
    }
    return py::str(name);
  });

  m.def("amend_func", [](llvm::Module *llvm_mod, mlir::ModuleOp mlir_mod,
                         llvm::LLVMContext &ctx, int xpu_arch) {
    llvm::DenseMap<llvm::StringRef, mlir::triton::xpu::XPUMetadata> XPUMetadata;
    extractXPUMetadata(mlir_mod, &XPUMetadata);

    for (auto &func : llvm_mod->functions()) {
      auto it = XPUMetadata.find(func.getName());
      if (it != XPUMetadata.end())
        mlir::triton::xpu::amendLLVMFunc(&func, it->second, xpu_arch);
    }
  });

  m.def("need_extern_lib", [](mlir::ModuleOp module) {
    llvm::SmallVector<mlir::LLVM::LLVMFuncOp> funcs;
    module.walk([&](mlir::LLVM::LLVMFuncOp func) {
      if (func.isExternal())
        funcs.push_back(func);
    });

    return funcs.empty() ? false : true;
  });

  m.def(
      "translate_to_asm",
      [](llvm::Module &module, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> py::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
          py::gil_scoped_release allow_threads;
          // create LLVM module from C++
          obj = translateLLVMIRToASM(module, triple, proc, features, flags,
                                     enable_fp_fusion, isObject);
        }
        if (isObject)
          return py::bytes(obj);
        else
#if !defined(TRITON_CONCEAL_IR) || (TRITON_CONCEAL_IR == 0)
          return py::str(obj);
#else
          return py::str("");
#endif
      },
      ret::take_ownership);
}

void init_triton_xpu(py::module &&m) {
  m.doc() = "Python bindings to the XPU Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_xpu_passes_conversion(passes.def_submodule("ttxpuir"));
  init_triton_xpu_passes_transform(passes.def_submodule("ttxpuir"));
  init_triton_xpu_llvm(m.def_submodule("llvm"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::xpu::TritonXPUDialect>();
    registerLLVMXPUDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  struct LutInfo {
    int dtype; // 0 - f32; 1 - f16
    int mode;  // 0 - KB;  1 - INTER
    int size;
    double min;
    double interval;
  };

  m.def("get_buffer_len", [](mlir::ModuleOp &mod, unsigned maxBufferSize) {
    unsigned bufferLen = maxBufferSize;

    auto _get_buffer_len = [&](mlir::Type &ptrTy, unsigned maxBufferSize,
                               unsigned &bufferLen) {
      mlir::Type ptrdataTy;
      if (auto ptrTensorTy = mlir::dyn_cast<mlir::RankedTensorType>(ptrTy)) {
        ptrdataTy =
            mlir::cast<mlir::triton::PointerType>(ptrTensorTy.getElementType())
                .getPointeeType();
      } else {
        ptrdataTy =
            mlir::cast<mlir::triton::PointerType>(ptrTy).getPointeeType();
      }
      if (ptrdataTy.isBF16() || ptrdataTy.isF16() || ptrdataTy.isF32()) {
        unsigned bitWidth = ptrdataTy.getIntOrFloatBitWidth();
        bufferLen = std::min(bufferLen, maxBufferSize / (bitWidth / 8));
      }
    };

    mod.walk([&](mlir::triton::LoadOp loadOp) {
      auto ptrTy = loadOp.getPtr().getType();
      _get_buffer_len(ptrTy, maxBufferSize, bufferLen);
    });
    mod.walk([&](mlir::triton::StoreOp storeOp) {
      auto ptrTy = storeOp.getPtr().getType();
      _get_buffer_len(ptrTy, maxBufferSize, bufferLen);
    });
    return bufferLen;
  });
}
