#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TARGETINFO_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TARGETINFO_H_

#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "llvm/TargetParser/TargetParser.h"
#include <string>

namespace mlir::triton::AMD {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  explicit TargetInfo(std::string arch) : arch(std::move(arch)) {}

  ISAFamily getISAFamily() const { return deduceISAFamily(arch); }

  llvm::AMDGPU::GPUKind getGPUKind() const;

  int getWarpSize() const;

  int getSharedMemorySize() const;

  bool supportMaximumMinimum() const override;

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  void barrier(Location loc, RewriterBase &rewriter,
               bool isWarpSync = false) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override;
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;
  bool canUseLDSTransLoad(int bitwidth) const;

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;

  Value permute(RewriterBase &rewriter, Location loc, Value a, Value b,
                Value selector) const override;

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  ProgramIDDim axis) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op, unsigned numLaneToReduce,
                  unsigned interleave) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;

  int getSharedAddressSpace() const override;

  int getAddressSpace(Attribute addressSpace) const override;

  bool supportVectorizedAtomics() const override;

  bool supportsDirectToLdsLoadBitWidth(int bitWidth) const;

  void localLoadOpAnnotation(triton::gpu::LocalLoadOp localLoadOp,
                             Operation *llLoadOp) const override;

private:
  void printfImpl(Value formatStrStart, int formatStrByteCount, ValueRange args,
                  ArrayRef<bool> isSigned, RewriterBase &rewriter,
                  bool useStdErr) const;

  std::string arch;
};
} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TARGETINFO_H_
