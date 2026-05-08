#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETFEATURES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETFEATURES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace mlir::triton::amdgpu {

enum class ISAFamily {
  Unknown,
  GCN5_1,
  CDNA1,
  CDNA2,
  CDNA3,
  CDNA4,
  RDNA1,
  RDNA2,
  RDNA3,
  RDNA4,
  GFX1250,
};

class TargetFeatures {
public:
  enum class TileKind {
    Standard,         // 16x16 tile layout.
    DoubleContiguity, // 16x16 with doubled B8 contiguity requirement.
  };

  struct LDSTransLoadParams {
    // Number of lanes that cooperate in the instruction.
    unsigned numLanesInShuffleGroup;
    // Number of bits that each lane reads per issued instruction.
    unsigned instBitWidth;
    // Number of elements that the instruction needs to be contiguous in LDS.
    unsigned tileSize;
    // Distribution of base tile in the full instruction.
    TileKind tileKind;
  };

  explicit TargetFeatures(std::optional<StringRef> arch);
  explicit TargetFeatures(StringRef arch);

  static TargetFeatures fromModuleOp(ModuleOp moduleOp);

  StringRef getArch() const;

  ISAFamily getISAFamily() const;

  bool isCDNA() const;
  bool isRDNA() const;
  bool isCDNA3() const;
  bool isCDNA4() const;
  bool isGFX1250() const;

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
  bool supportsClusterLoadBitWidth(int bitWidth) const;

  bool supportsBufferAtomicRMW() const;
  bool supportsBufferAtomicFadd(Type elementType) const;
  int32_t getBufferAtomicCachePolicy(bool hasUsers) const;

  bool supportMaximumMinimum() const;
  bool supportDppBroadcast() const;
  bool supportsPermlaneSwap() const;
  bool supportsCvtPkScalePk8() const;
  bool supportsHwScaledUpcast() const;

  bool supportBitwidth16Elementwise() const;
  bool supportBitwidth32Elementwise() const;

private:
  static constexpr char kTargetPrefix[] = "hip:";

  std::string arch;
};

bool isCDNA(ISAFamily isaFamily);
bool isRDNA(ISAFamily isaFamily);

} // namespace mlir::triton::amdgpu

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETFEATURES_H_
