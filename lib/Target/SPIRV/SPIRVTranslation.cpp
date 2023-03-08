#include "mlir/Conversion/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonGPUToSPIRV/TritonGPUToSPIRVPass.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

#include <spirv-tools/libspirv.hpp>

namespace mlir {
namespace triton {

LogicalResult assembleSPIRV(std::string spirvCode, raw_ostream &output) {
  auto DisMessagePrinter = [](spv_message_level_t Level,
                              const char* source,
                              const spv_position_t& position,
                              const char* message) -> void {};
  spvtools::SpirvTools SpvTool(SPV_ENV_OPENCL_2_0);
  SpvTool.SetMessageConsumer(DisMessagePrinter);

  std::vector<uint32_t> binary;
  if (!SpvTool.Assemble(spirvCode, &binary, SPV_TEXT_TO_BINARY_OPTION_NONE)) {
    return failure("SPIRV: Failed to assemble the code");
  }
  std::stringstream is;
  is.rdbuf()->pubsetbuf(reinterpret_cast<char*>(&binary[0]), binary.size() * sizeof(uint32_t));
  output << is.str();
  return mlir::success();
}

LogicalResult disassembleSPIRV(uint32_t* binary_ptr, size_t binary_size, raw_ostream &output) {
  auto DisMessagePrinter = [](spv_message_level_t Level,
                              const char* source,
                              const spv_position_t& position,
                              const char* message) -> void {};
  spvtools::SpirvTools SpvTool(SPV_ENV_OPENCL_2_0);
  SpvTool.SetMessageConsumer(DisMessagePrinter);

  std::string spriv_code;
  if (!SpvTool.Disassemble(binary_ptr, binary_size, &spriv_code)) {
    return failure("SPIRV: Failed to generate textual assembly");
  }
  output << spriv_code;
  return mlir::success();
}

static LogicalResult
getInterfaceVariables(spirv::FuncOp funcOp,
                      SmallVectorImpl<Attribute> &interfaceVars) {
  auto module = funcOp->getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return failure();
  }
  SetVector<Operation *> interfaceVarSet;

  // TODO: This should in reality traverse the entry function
  // call graph and collect all the interfaces. For now, just traverse the
  // instructions in this function.
  funcOp.walk([&](spirv::AddressOfOp addressOfOp) {
    auto var =
            module.lookupSymbol<spirv::GlobalVariableOp>(addressOfOp.getVariable());
    // TODO: Per SPIR-V spec: "Before version 1.4, the interface’s
    // storage classes are limited to the Input and Output storage classes.
    // Starting with version 1.4, the interface’s storage classes are all
    // storage classes used in declaring all global variables referenced by the
    // entry point’s call tree." We should consider the target environment here.
    switch (var.getType().cast<spirv::PointerType>().getStorageClass()) {
      case spirv::StorageClass::Input:
      case spirv::StorageClass::Output:
        interfaceVarSet.insert(var.getOperation());
        break;
      default:
        break;
    }
  });
  for (auto &var : interfaceVarSet) {
    interfaceVars.push_back(SymbolRefAttr::get(
            funcOp.getContext(), cast<spirv::GlobalVariableOp>(var).getSymName()));
  }
  return success();
}

static LogicalResult translateTritonSPIRVToSPIRVIR(ModuleOp module, raw_ostream &output) {
  if (!module)
    return failure();

  SmallVector<uint32_t, 0> binary;

  SmallVector<spirv::ModuleOp, 1> spirvModules;
  OpBuilder builder(module->getContext());

  module.walk([&](ModuleOp op) {
    auto newModuleOp =
            builder.create<spirv::ModuleOp>(op.getLoc(), op.getName());

    auto& region = op.getRegion();
    auto& parent = *newModuleOp.getBody()->getParent();
    auto iter = newModuleOp.getBody()->getIterator();

    parent.getBlocks().splice(iter, region.getBlocks());

    // Remove the terminator block that was automatically added by builder
    auto& last_block = newModuleOp.getBodyRegion().back();
    last_block.getParent()->getBlocks().remove(last_block);

    //copy the attributes
    newModuleOp->setAttrs(op->getAttrDictionary());

    //Set the spirv module attributes
    newModuleOp->setAttr("addressing_model",
                         builder.getAttr<spirv::AddressingModelAttr>(
                                 spirv::AddressingModel::Physical64));
    newModuleOp->setAttr("memory_model",
                         builder.getAttr<spirv::MemoryModelAttr>(
                                 spirv::MemoryModel::OpenCL));
    spirv::Capability caps_opencl[] = {
            // clang-format off
            spirv::Capability::Addresses,
            spirv::Capability::Float16Buffer,
            spirv::Capability::Int64,
            spirv::Capability::Int16,
            spirv::Capability::Int8,
            spirv::Capability::Kernel,
            spirv::Capability::Linkage,
            spirv::Capability::Vector16,
            spirv::Capability::GenericPointer,
            spirv::Capability::Groups,
            spirv::Capability::Float16,
            spirv::Capability::Float64,
            spirv::Capability::AtomicFloat32AddEXT,
            spirv::Capability::ExpectAssumeKHR,
            // clang-format on
    };
    spirv::Extension exts_opencl[] = {
            spirv::Extension::SPV_EXT_shader_atomic_float_add,
            spirv::Extension::SPV_KHR_expect_assume};
    newModuleOp->setAttr("vce_triple",
                       spirv::VerCapExtAttr::get(
                               spirv::Version::V_1_0, caps_opencl,
                               exts_opencl, builder.getContext()));

    spirvModules.push_back(newModuleOp);
  });

  if (spirvModules.empty())
    return module.emitError("found no 'spv.module' op");

  if (spirvModules.size() != 1)
    return module.emitError("found more than one 'spv.module' op");

  for(auto &sprivModule: spirvModules) {
    int threadsPerWarp = sprivModule->getAttr("triton_gpu.threads-per-warp").cast<IntegerAttr>().getInt();
    sprivModule.walk([&](spirv::FuncOp op) {
      auto entryPointAttrName = spirv::getEntryPointABIAttrName();
      auto entryPointAttr =
              op->getAttrOfType<spirv::EntryPointABIAttr>(entryPointAttrName);
      if (!entryPointAttr) {
        return;
      }

      OpBuilder::InsertionGuard moduleInsertionGuard(builder);
      auto spirvModule = op->getParentOfType<spirv::ModuleOp>();
      builder.setInsertionPointToEnd(spirvModule.getBody());

      // Adds the spv.EntryPointOp after collecting all the interface variables
      // needed.
      SmallVector<Attribute, 1> interfaceVars;
      if (failed(getInterfaceVariables(op, interfaceVars))) {
        return;
      }

      builder.create<spirv::EntryPointOp>(op.getLoc(), spirv::ExecutionModel::Kernel,
                                          op, interfaceVars);

      builder.create<spirv::ExecutionModeOp>(op.getLoc(), op,
                                             spirv::ExecutionMode::SubgroupSize,
                                             threadsPerWarp);

      op->removeAttr(entryPointAttrName);
      op->removeAttr("sym_visibility");
    });
  }

  if (failed(spirv::serialize(spirvModules[0], binary)))
    return failure();
  if (failed(disassembleSPIRV(binary.data(), binary.size(), output)))
    return failure();

  return mlir::success();
}

std::string
translateTritonGPUToSPIRVIR(mlir::ModuleOp module, int computeCapability) {
  mlir::PassManager pm(module->getContext());
  applyPassManagerCLOptions(pm);
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
          /*shouldPrintBeforePass=*/nullptr,
          /*shouldPrintAfterPass=*/
          [](mlir::Pass *pass, mlir::Operation *) {
            return ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
          },
          /*printModuleScope=*/false,
          /*printAfterOnlyOnChange=*/true,
          /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  pm.addPass(createConvertTritonGPUToSPIRVPass(computeCapability));
  // Canonicalize to eliminate the remaining UnrealizedConversionCastOp
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass()); // Simplify the IR to improve readability.
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  std::string spirvModule;
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed";
    return spirvModule;
  }

  llvm::raw_string_ostream os(spirvModule);
  if (failed(translateTritonSPIRVToSPIRVIR(module, os))) {
    llvm::errs() << "Translate to SPIRV IR failed";
    return spirvModule;
  }

  return spirvModule;
}

} // namespace triton
} // namespace mlir
