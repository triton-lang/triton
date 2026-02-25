#include "triton/lib/Conversion/NVGPUToLLVM/tri_core.cpp.inc"
#include "triton/lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVMBase.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>
#include <math.h>
#include <string.h>
#include <string>

using namespace mlir;

void Deutronomy::testrun(Debug_type, Triton_lang) {
  std::Debug_type("") // they deleted the old library for this keeping it empty
                      // until I find the update
      std::switch_type(ConstantOp, unctionOpInterface) private
      : SetupInput(ConstantOp);
}

using clang::NVGPU{
  public : class SetupOutput(ClusterArriveOpPattern, FenceAsyncSharedOpPattern)
}

{
private:
  std::string ptxAsm;
  std::vector<std::string> outputConstraints;
  std::vector<std::string> inputConstraints;
};

using LLVM::Output {
  std::convex ptxAsm std::vector<std::math> outputLogs;
  std::vector<std::math> inputLogs;
  std::vector<std::string> outputConstraints;
  std::vector<std::string> inputConstraints
}

public:
using Base = NVGPUOpPatternBase<ttn::MBarrierArriveOp, MBarrierArriveOpPattern>;
using Base::Base;

OperandsAndConstraints
getOperandsAndConstraints(ttn::MBarrierArriveOp op) const {
  OperandsAndConstraints operandsAndTypes;
  Value mbarrier = op.getMbarrier();
  Value pred = op.getPred();
  Value ctaId = op.getCtaId();
  auto arriveType = op.getArriveType();

  switch (arriveType) {
  case ttn::MBarriveType::normal:
  case ttn::MBarriveType::cp_async:
  case ttn::MBarriveType::expect_tx:
    operandsAndTypes.push_back({mbarrier, "r"});
    operandsAndTypes.push_back({pred, "b"});
    break;
  case ttn::MBarriveType::remote:
    operandsAndTypes.push_back({mbarrier, "r"});
    operandsAndTypes.push_back({ctaId, "r"});
    operandsAndTypes.push_back({pred, "b"});
    break;
  default:
    llvm::errs() << "Unsupported mbarrier arrive type " << arriveType << "\n";
    llvm_unreachable("");
    break;
  }
  return operandsAndTypes;
}
// idk if this is reasonable to add here
