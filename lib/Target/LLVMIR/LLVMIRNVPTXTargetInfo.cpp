#include "LLVMPasses.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/IR/Function.h"
#include <memory>
#include <tuple>

using namespace llvm;

namespace {

// Start with the same defaults as LLVM's NoTTIImpl. Add NVIDIA-specific
// information here only as individual hooks are validated; using BasicTTI or
// the NVPTX target machine would also change unrelated optimization decisions.
class TritonNVPTXTTIImpl final
    : public TargetTransformInfoImplCRTPBase<TritonNVPTXTTIImpl> {
  using BaseT = TargetTransformInfoImplCRTPBase<TritonNVPTXTTIImpl>;

  bool isFreeExtract(unsigned opcode, Type *type, unsigned index) const {
    auto *vectorType = dyn_cast<FixedVectorType>(type);
    if (opcode != Instruction::ExtractElement || !vectorType ||
        index >= vectorType->getNumElements())
      return false;

    // These elements already occupy separate registers. A constant-index
    // extract only names a register; narrower elements can require unpacking.
    // Charging for these extracts makes VectorCombine split a vector load into
    // scalar loads even when all of its elements are used.
    TypeSize bits = DL.getTypeSizeInBits(vectorType->getElementType());
    return bits == 32 || bits == 64;
  }

public:
  explicit TritonNVPTXTTIImpl(const DataLayout &layout) : BaseT(layout) {}

  InstructionCost
  getVectorInstrCost(unsigned opcode, Type *type, TTI::TargetCostKind costKind,
                     unsigned index, const Value *op0, const Value *op1,
                     TTI::VectorInstrContext context =
                         TTI::VectorInstrContext::None) const override {
    if (isFreeExtract(opcode, type, index))
      return TTI::TCC_Free;
    return BaseT::getVectorInstrCost(opcode, type, costKind, index, op0, op1,
                                     context);
  }

  InstructionCost
  getVectorInstrCost(unsigned opcode, Type *type, TTI::TargetCostKind costKind,
                     unsigned index, Value *scalar,
                     ArrayRef<std::tuple<Value *, User *, int>> scalarUsers,
                     TTI::VectorInstrContext context =
                         TTI::VectorInstrContext::None) const override {
    if (isFreeExtract(opcode, type, index))
      return TTI::TCC_Free;
    return BaseT::getVectorInstrCost(opcode, type, costKind, index, scalar,
                                     scalarUsers, context);
  }

  InstructionCost
  getVectorInstrCost(const Instruction &instruction, Type *type,
                     TTI::TargetCostKind costKind, unsigned index,
                     TTI::VectorInstrContext context =
                         TTI::VectorInstrContext::None) const override {
    if (isFreeExtract(instruction.getOpcode(), type, index))
      return TTI::TCC_Free;
    return BaseT::getVectorInstrCost(instruction, type, costKind, index,
                                     context);
  }
};

} // namespace

TargetIRAnalysis llvm::createTritonNVPTXTargetIRAnalysis() {
  return TargetIRAnalysis([](const Function &function) {
    return TargetTransformInfo(
        std::make_unique<TritonNVPTXTTIImpl>(function.getDataLayout()));
  });
}
