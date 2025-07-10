#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_WMMAGROUP_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_WMMAGROUP_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

struct WmmaIntrinsic {
  // Chooses a suitable wmma instrinsic for the given input case.
  static FailureOr<WmmaIntrinsic> selectFor(int version, unsigned mDim,
                                            unsigned nDim, unsigned inputKDim,
                                            Type aElemType, Type bElemType,
                                            Type dElemType);

  WmmaIntrinsic(StringRef symbol, unsigned m, unsigned n, unsigned k,
                unsigned kB, Type aET, Type bET, Type dET)
      : name(symbol), mDim(m), nDim(n), kDim(k), kBase(kB), aElementType(aET),
        bElementType(bET), dElementType(dET) {}
  WmmaIntrinsic(const WmmaIntrinsic &other) = default;
  WmmaIntrinsic(WmmaIntrinsic &&other) = default;
  WmmaIntrinsic() = default;
  WmmaIntrinsic &operator=(WmmaIntrinsic &&other) = default;

  llvm::StringRef name;

  // m, n, and k refer to the shapes of the two operands of an wmma intrinsic:
  // Operand A has shape [m]x[k]; operand B has shape [k]x[n].

  unsigned mDim;
  unsigned nDim;
  unsigned kDim;

  // kBase is the number of elements each thread holds.
  unsigned kBase;

  Type aElementType;
  Type bElementType;
  Type dElementType;
};
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_WMMAGROUP_H_
