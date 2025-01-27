#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// AMDGPU MFMA instruction selection utilities
//===----------------------------------------------------------------------===//

enum class MfmaTypeId : uint32_t {
  Fp32TyId = 0,
  Xf32TyId,
  Fp16TyId,
  Bf16TyId,
  I8TyId,
  Fp8Fp8TyId,
  Fp8Bf8TyId,
  Bf8Fp8TyId,
  Bf8Bf8TyId
};

struct MfmaInsnGroupSelectKey {
  unsigned mDim, nDim;
  MfmaTypeId elemType;
  int mfmaVersion;
};

struct MfmaInsnAttr {
  // m,n,k refer to the shapes of the two operands of mfma instructions.
  // Operand A has shape m x k. Operand B has shape k x n.
  // For mfma32 and mfma16 instructions, they are the same as
  // the dims in the instruction name, i.e. mfma_DType_mxnxkxABType
  unsigned m;
  unsigned n;
  unsigned k;
  // kBase refers to the number of elements per thread
  unsigned kBase;
  llvm::StringRef insn;
};

template <typename T>
constexpr typename std::underlying_type<T>::type cast_as_underlying(T t) {
  return static_cast<typename std::underlying_type<T>::type>(t);
}

struct MfmaInsnGroupSelectKeyInfo
    : public llvm::DenseMapInfo<MfmaInsnGroupSelectKey> {
  static inline MfmaInsnGroupSelectKey getEmptyKey() {
    return {32, 32, MfmaTypeId::Fp32TyId, 0};
  }

  static inline MfmaInsnGroupSelectKey getTombstoneKey() {
    return {32, 32, MfmaTypeId::Fp32TyId, -1};
  }

  static inline bool isEqual(const MfmaInsnGroupSelectKey &lhs,
                             const MfmaInsnGroupSelectKey &rhs) {
    return lhs.mDim == rhs.mDim && lhs.nDim == rhs.nDim &&
           lhs.elemType == rhs.elemType && lhs.mfmaVersion == rhs.mfmaVersion;
  }

  static unsigned getHashValue(const MfmaInsnGroupSelectKey &key) {
    auto dimHash = llvm::detail::combineHashValue(key.mDim, key.nDim);
    auto verHash = llvm::detail::combineHashValue(dimHash, key.mfmaVersion);
    auto elemHash = cast_as_underlying(key.elemType);
    return llvm::detail::combineHashValue(elemHash, verHash);
  }
};

class MfmaInsn {
private:
  Type elementTypeA;
  Type elementTypeB;
  MfmaInsnAttr attr;

public:
  static FailureOr<MfmaInsn> selectMfma(unsigned mDim, unsigned nDim,
                                        Type elementTypeA, Type elementTypeB,
                                        int mfmaVersion, bool allowXF32);
  MfmaInsn(Type elementTypeA, Type elementTypeB, const MfmaInsnAttr &attr);
  unsigned getKDim();
  unsigned getMDim();
  unsigned getNDim();
  StringRef getInsnName();
  unsigned getKBase();
  Type getElementTypeA();
  Type getElementTypeB();
};
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_MFMAGROUP_H_
