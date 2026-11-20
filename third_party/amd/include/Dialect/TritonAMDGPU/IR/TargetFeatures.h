#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETFEATURES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETFEATURES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace mlir::triton::amdgpu {

inline constexpr llvm::StringLiteral kWmmaRestrictedInstsFeature =
    "wmma-restricted-insts";

enum class ISAFamily {
  Unknown,
  CDNA2,
  CDNA3,
  CDNA4,
  CDNA5,
  RDNA3,
  RDNA4m,
  RDNA4,
};

class TargetFeatures {
public:
  struct LDSTransLoadParams {
    // Number of bits that each lane reads per issued instruction.
    unsigned instBitWidth;
    // Number of elements that the instruction needs to be contiguous in LDS.
    unsigned tileSize;
    // Number of leading bases in the order-preserving interleaving of register
    // and lane bases in the address layout of the full instruction tile. I.e.,
    // addr basis order:
    //   leading reg bases
    //   leading lane bases
    //   remaining reg bases
    //   remaining lane bases
    unsigned leadingRegBases;
    unsigned leadingLaneBases;
  };

  explicit TargetFeatures(std::optional<StringRef> arch);
  explicit TargetFeatures(StringRef arch);

  static TargetFeatures fromModuleOp(ModuleOp moduleOp);

  StringRef getArch() const;
  StringRef getBaseArch() const;

  ISAFamily getISAFamily() const;

  bool isCDNA() const;
  bool isRDNA() const;
  bool isCDNA3() const;
  bool isCDNA4() const;
  bool isCDNA5() const;
  bool isGFX1250Strict() const;

  int getWarpSize() const;
  bool supportsWaveId() const;
  int getSharedMemorySize() const;
  size_t getSharedMemoryPartitionSize() const;

  std::optional<LDSTransLoadParams> queryLDSTransLoadParams(int bitWidth) const;

  bool supportsDirectToLdsScatter() const;
  bool supportsDirectToLdsLoadBitWidth(int bitWidth) const;
  bool supportsDirectFromLdsStoreBitWidth(int bitWidth) const;
  bool supportsBufferLoadToLocal() const;

  bool requiresAliasInfoForAsyncOps() const;
  bool useAsyncMarks() const;

  bool supportsTDM() const;
  bool supportsMultiCTALaunch() const;
  bool supportsMulticast() const;
  unsigned getMaxMulticastMaskPopcount() const;
  bool supportsClusterLoadBitWidth(int bitWidth) const;

  bool supportsBufferAtomicRMW() const;
  bool supportsBufferAtomicFadd(Type elementType) const;
  bool supportsBufferAtomicFMinMax(Type elementType) const;
  int32_t getBufferAtomicCachePolicy(bool hasUsers) const;

  bool supportMaximumMinimum() const;
  bool supportDppBroadcast() const;
  bool supportsPermlaneSwap() const;
  bool supportsCvtPkScalePk8() const;
  bool supportsCvtPkScalePk8Upcast() const;
  bool supportsCvtPkScalePk8Block16() const;
  // Unscaled v_cvt_[pk_]{f32_fp8,f32_bf8,fp8_f32,bf8_f32} operating on FNUZ fp8
  // formats.
  bool supportsCvtFp8Fnuz() const;
  // The same unscaled instructions operating on OCP fp8 formats.
  bool supportsCvtFp8Ocp() const;
  // Scaled v_cvt_scalef32_pk_* converting two OCP fp8 values per instruction.
  bool supportsCvtScaleF32Pk() const;
  // Packed v_cvt_pk_{f16,bf16}_f32 RTNE downcasts.
  bool supportsCvtPkFp32ToFp16Bf16() const;
  ArrayRef<StringRef> getUnsupportedWmmaFeatures() const;
  bool supportsHwScaledUpcast() const;
  bool supportsHwScaledDowncast() const;

  bool supportBitwidth16Elementwise() const;
  bool supportBitwidth32Elementwise() const;

private:
  static constexpr char kTargetPrefix[] = "hip:";

  std::string arch;
  std::string baseArch;
  ISAFamily isaFamily;
};

bool isCDNA(ISAFamily isaFamily);
bool isRDNA(ISAFamily isaFamily);

} // namespace mlir::triton::amdgpu

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETFEATURES_H_
