#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H

#include "triton/Conversion/MLIRTypes.h"

namespace mlir::triton {
enum class ProgramIDDim : uint32_t;

class TargetInfoBase {
public:
  virtual bool supportMaximumMinimum() const = 0;

  virtual Value getClusterCTAId(RewriterBase &rewriter, Location loc) const = 0;

  virtual Value ballot(RewriterBase &rewriter, Location loc, Type type,
                       Value cmp) const = 0;

  // Insert a synchronization barrier. If isWarpSync is true, emit a warp-level
  // synchronization when supported by the backend; otherwise emit a block/CTA
  // level barrier. Backends that do not support warp-level barriers should
  // conservatively emit a block-level barrier.
  virtual void barrier(Location loc, RewriterBase &rewriter,
                       bool isWarpSync = false) const = 0;

  // Store/load a value from shared memory, either in the same CTA or, if
  // `ctaId` is non-nullopt, in another CTA in the same group.
  //
  // A target that does not support cross-CTA transfers will assert if ctaId is
  // non-nullopt.
  //
  // Assumes the address is aligned to the width of `val`.
  virtual void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                            std::optional<Value> ctaId, Value val,
                            Value pred) const = 0;
  virtual Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                            std::optional<Value> ctaId, Type elemTy, Value pred,
                            Operation *localLoadOp = nullptr) const = 0;

  void storeShared(RewriterBase &rewriter, Location loc, Value ptr, Value val,
                   Value pred) const {
    storeDShared(rewriter, loc, ptr, /*ctaId=*/std::nullopt, val, pred);
  }
  Value loadShared(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
                   Value pred) const {
    return loadDShared(rewriter, loc, ptr, /*ctaId=*/std::nullopt, elemTy,
                       pred);
  }

  virtual Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                           int i) const = 0;
  virtual Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                          int i) const = 0;
  virtual Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                           int i) const = 0;
  virtual Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                           Value i) const = 0;

  virtual Value permute(RewriterBase &rewriter, Location loc, Value a, Value b,
                        Value selector) const = 0;

  virtual Value programId(RewriterBase &rewriter, Location loc,
                          ModuleOp moduleOp, ProgramIDDim axis) const = 0;

  virtual bool warpReduce(RewriterBase &rewriter, Location loc,
                          SmallVector<Value> &acc, triton::ReduceOp op,
                          unsigned numLaneToReduce,
                          unsigned interleave) const = 0;

  virtual std::string getMulhiFuncName(Type resultElementTy) const = 0;
  // Emits LLVM code with |rewriter| to print a message following the given
  // format from the device. |formatStrStart| is the pointer to the start of
  // the format string global variable; |args| are the arguments to fill
  // placeholders in the format string.
  virtual void printf(RewriterBase &rewriter, Value formatStrStart,
                      int formatStrByteCount, ValueRange args,
                      ArrayRef<bool> isSigned = {}) const = 0;

  // Emits LLVM code with |rewriter| to print a message, particularly useful for
  // backend debug. |msg| is the message to print, |args| are the arguments to
  // fill placeholders in the |msg|.
  // NOTE: This function is used for backend debug. DO NOT DELETE.
  // Example use: targetInfo.printf(rewriter,"index: %d, value: %f", {index,
  // value});
  virtual void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                      ArrayRef<bool> isSigned = {}) const = 0;

  // Emits LLVM code with |rewriter| to perform assertion failure with the given
  // |message| from the given |func| in |file|.
  virtual void assertFail(RewriterBase &rewriter, Location loc,
                          StringRef message, StringRef file, StringRef func,
                          int line) const = 0;

  virtual int getSharedAddressSpace() const = 0;

  virtual int getAddressSpace(Attribute addressSpace) const = 0;

  virtual bool supportVectorizedAtomics() const = 0;

  virtual bool supportLdMatrix() const { return false; }
  virtual bool supportStMatrix() const { return false; }
  virtual bool isCuda() const { return false; }

  // Annotate target specific information to local load operations during
  // lowering to LLVM. `llLoadOp` is the generated LLVM load op.
  virtual void localLoadOpAnnotation(triton::gpu::LocalLoadOp localLoadOp,
                                     Operation *llLoadOp) const {}

  virtual ~TargetInfoBase() {}
};
} // namespace mlir::triton
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
