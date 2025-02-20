#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_

#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

struct MfmaIntrinsic {
  // Chooses a suitable mfma instrinsic for the given input case.
  static FailureOr<MfmaIntrinsic> selectFor(int version, unsigned mDim,
                                            unsigned nDim, unsigned inputKDim,
                                            Type aElemType, Type bElemType,
                                            bool withScale, bool useTF32);

  MfmaIntrinsic(StringRef symbol, unsigned m, unsigned n, unsigned k,
                unsigned kB, Type aET, Type bET)
      : name(symbol), mDim(m), nDim(n), kDim(k), kBase(kB), aElementType(aET),
        bElementType(bET) {}
  MfmaIntrinsic(const MfmaIntrinsic &other) = default;
  MfmaIntrinsic(MfmaIntrinsic &&other) = default;

  llvm::StringRef name;

  // m, n, and k refer to the shapes of the two operands of an mfma intrinsic:
  // Operand A has shape [m]x[k]; operand B has shape [k]x[n].
  // For mfma32 and mfma16 intrinsics, they are encoded in the instruction
  // name, i.e. mfma_DType_[m]x[n]x[k]xABType.
  unsigned mDim;
  unsigned nDim;
  unsigned kDim;

  // kBase is the number of elements each thread holds.
  unsigned kBase;

  Type aElementType;
  Type bElementType;
};
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_
