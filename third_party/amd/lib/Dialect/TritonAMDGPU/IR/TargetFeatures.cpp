#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::amdgpu {

namespace {
::mlir::ROCDL::TargetInfo parseTargetInfo(StringRef arch) {
  // Passes may run without a target.
  if (arch.empty())
    return {};
  FailureOr<::mlir::ROCDL::TargetInfo> info =
      ::mlir::ROCDL::TargetInfo::get(arch);
  return succeeded(info) ? *info : ::mlir::ROCDL::TargetInfo();
}

ISAFamily computeISAFamily(const ::mlir::ROCDL::TargetInfo &info);
} // namespace

TargetFeatures::TargetFeatures(std::optional<StringRef> arch)
    : TargetFeatures(arch.value_or("")) {}

TargetFeatures::TargetFeatures(StringRef arch)
    : arch(arch.str()), baseArch(arch.split("-strict").first),
      info(parseTargetInfo(arch)), isaFamily(computeISAFamily(info)) {}

TargetFeatures TargetFeatures::fromModuleOp(ModuleOp moduleOp) {
  auto targetAttr =
      moduleOp->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
  if (!targetAttr)
    return TargetFeatures(StringRef());

  StringRef targetName = targetAttr.getValue();
  assert(targetName.starts_with(kTargetPrefix) &&
         "expected target attribute to be prefixed with \"hip:\"");

  return TargetFeatures(targetName.drop_front(sizeof(kTargetPrefix) - 1));
}

StringRef TargetFeatures::getArch() const { return arch; }

StringRef TargetFeatures::getBaseArch() const { return baseArch; }

ISAFamily TargetFeatures::getISAFamily() const { return isaFamily; }

namespace {
ISAFamily computeISAFamily(const ::mlir::ROCDL::TargetInfo &info) {
  // Classify by generation and feature rather than by processor name, so that
  // generic targets and processors added to LLVM later land in the family
  // whose instructions they have. See
  // https://llvm.org/docs/AMDGPUUsage.html#processors for the grouping.
  using namespace llvm::AMDGPU;
  if (info.isGeneration(9)) {
    if (info.has(FEAT_GFX950_INSTS))
      return ISAFamily::CDNA4;
    if (info.has(FEAT_GFX940_INSTS))
      return ISAFamily::CDNA3;
    if (info.has(FEAT_GFX90A_INSTS))
      return ISAFamily::CDNA2;
    return ISAFamily::Unknown;
  }
  if (info.isGeneration(11)) {
    // gfx11.7 carries the gfx12-style 128-bit-operand WMMA instructions.
    if (info.has(FEAT_WMMA_128B_INSTS))
      return ISAFamily::RDNA4m;
    return ISAFamily::RDNA3;
  }
  if (info.isGeneration(12)) {
    if (info.has(FEAT_GFX1250_INSTS))
      return ISAFamily::CDNA5;
    return ISAFamily::RDNA4;
  }
  return ISAFamily::Unknown;
}
} // namespace

bool TargetFeatures::isCDNA() const {
  return mlir::triton::amdgpu::isCDNA(getISAFamily());
}

bool TargetFeatures::isRDNA() const {
  return mlir::triton::amdgpu::isRDNA(getISAFamily());
}

bool TargetFeatures::isCDNA3() const {
  return getISAFamily() == ISAFamily::CDNA3;
}

bool TargetFeatures::isCDNA4() const {
  return getISAFamily() == ISAFamily::CDNA4;
}

bool TargetFeatures::isCDNA5() const {
  return getISAFamily() == ISAFamily::CDNA5;
}

bool TargetFeatures::isGFX1250Strict() const {
  return getGPUKind() == llvm::AMDGPU::GK_GFX1250_STRICT;
}

int TargetFeatures::getWarpSize() const {
  return info.getWavefrontSize().value_or(32);
}

bool TargetFeatures::supportsWaveId() const {
  return getISAFamily() == ISAFamily::RDNA4 ||
         getISAFamily() == ISAFamily::CDNA5;
}

int TargetFeatures::getSharedMemorySize() const {
  return info.getMaxAddressableLocalMemorySize().value_or(64 * 1024);
}

int TargetFeatures::getSharedMemoryBanks() const {
  return info.getLDSBankCount().value_or(32);
}

size_t TargetFeatures::getSharedMemoryPartitionSize() const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA5:
    return 64 * 1024;
  default:
    // No partitioning on other targets.
    return 0;
  }
}

std::optional<TargetFeatures::LDSTransLoadParams>
TargetFeatures::queryLDSTransLoadParams(int bitWidth) const {
  struct TransConfig {
    ISAFamily isaFamily;
    int bitWidth;
    unsigned instBitWidth;
    // addr basis order:
    //   leading reg bases
    //   leading lane bases
    //   remaining reg bases
    //   remaining lane bases
    unsigned leadingRegBases;
    unsigned leadingLaneBases;
  };

  static constexpr TransConfig configs[] = {
      {ISAFamily::CDNA4, 16, 64, 0, 2}, {ISAFamily::CDNA4, 8, 64, 0, 1},
      {ISAFamily::CDNA4, 4, 64, 0, 0},  {ISAFamily::CDNA5, 16, 128, 0, 0},
      {ISAFamily::CDNA5, 8, 64, 2, 1},  {ISAFamily::CDNA5, 4, 64, 3, 1},
  };

  const auto isaFamily = getISAFamily();
  for (const auto &config : configs) {
    if (config.isaFamily == isaFamily && config.bitWidth == bitWidth) {
      return LDSTransLoadParams{
          config.instBitWidth, config.instBitWidth / config.bitWidth,
          config.leadingRegBases, config.leadingLaneBases};
    }
  }
  return std::nullopt;
}

bool TargetFeatures::supportsDirectToLdsScatter() const { return isCDNA5(); }

bool TargetFeatures::supportsDirectToLdsLoadBitWidth(int bitWidth) const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA3:
    // Disable 8 and 16 bits because they get extended to 32 bit.
    return llvm::is_contained({32, /*16, 8*/}, bitWidth);
  case ISAFamily::CDNA4:
    // Disable 8, 16, 96 bits because they get extended to 32/128 bit.
    return llvm::is_contained({128, /*96, */ 32, /*16, 8*/}, bitWidth);
  case ISAFamily::CDNA5:
    // Disable 8, 16 bits because they get extended to 32 bit and therefore
    // overwrite. 96 is not a pow2 and generally not useful in Triton.
    return llvm::is_contained({128, 64, /*96, */ 32, /*16, 8*/}, bitWidth);
  default:
    break;
  }

  return false;
}

bool TargetFeatures::supportsDirectFromLdsStoreBitWidth(int bitWidth) const {
  if (getISAFamily() == ISAFamily::CDNA5) {
    return llvm::is_contained({128, 64, 32, 8}, bitWidth);
  }
  return false;
}

bool TargetFeatures::supportsBufferLoadToLocal() const {
  return llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                            getISAFamily());
}

bool TargetFeatures::requiresAliasInfoForAsyncOps() const {
  return llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                            getISAFamily());
}

bool TargetFeatures::useAsyncMarks() const {
  return llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                            getISAFamily());
}

bool TargetFeatures::supportsTDM() const { return isCDNA5(); }

bool TargetFeatures::supportsMultiCTALaunch() const { return isCDNA5(); }

bool TargetFeatures::supportsMulticast() const {
  return isCDNA5() && !isGFX1250Strict();
}

unsigned TargetFeatures::getMaxMulticastMaskPopcount() const {
  return supportsMulticast() ? 5 : 1;
}

bool TargetFeatures::supportsClusterLoadBitWidth(int bitWidth) const {
  if (supportsMulticast()) {
    return llvm::is_contained({32, 64, 128}, bitWidth);
  }
  return false;
}

bool TargetFeatures::supportsBufferAtomicRMW() const {
  return llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4,
                             ISAFamily::RDNA3, ISAFamily::RDNA4m,
                             ISAFamily::RDNA4, ISAFamily::CDNA5},
                            getISAFamily());
}

bool TargetFeatures::supportsBufferAtomicFadd(Type elementType) const {
  auto isaFamily = getISAFamily();
  if (isaFamily == ISAFamily::CDNA3 && elementType.isBF16())
    return false;
  if (isaFamily == ISAFamily::RDNA3 && !elementType.isF32())
    return false;
  if (isaFamily == ISAFamily::RDNA4m)
    return elementType.isF32();
  if (isaFamily == ISAFamily::RDNA4 && elementType.isF64())
    return false;
  return true;
}

bool TargetFeatures::supportsBufferAtomicFMinMax(Type elementType) const {
  auto isaFamily = getISAFamily();
  if (elementType.isF32()) {
    return llvm::is_contained({ISAFamily::RDNA3, ISAFamily::RDNA4m,
                               ISAFamily::RDNA4, ISAFamily::CDNA5},
                              isaFamily);
  }
  if (elementType.isF64()) {
    return llvm::is_contained({ISAFamily::CDNA2, ISAFamily::CDNA3,
                               ISAFamily::CDNA4, ISAFamily::CDNA5},
                              isaFamily);
  }
  return false;
}

int32_t TargetFeatures::getBufferAtomicCachePolicy(bool hasUsers) const {
  const int sc0Bit = 0b1;          // TH_ATOMIC_RETURN (cpol bit 0)
  const int scopeDevBit = 0b10000; // SCOPE_DEV = 2 << 3 (cpol bits [4:3])
  int32_t aux = 0;
  if (hasUsers)
    aux |= sc0Bit;
  if (getISAFamily() == ISAFamily::CDNA5)
    aux |= scopeDevBit;
  return aux;
}

bool TargetFeatures::supportMaximumMinimum() const {
  return getISAFamily() == ISAFamily::CDNA4 ||
         getISAFamily() == ISAFamily::RDNA4m ||
         getISAFamily() == ISAFamily::RDNA4 ||
         getISAFamily() == ISAFamily::CDNA5;
}

bool TargetFeatures::supportDppBroadcast() const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return true;
  case ISAFamily::CDNA5:
    return false;
  default:
    return false;
  }
}

bool TargetFeatures::supportsPermlaneSwap() const {
  return getISAFamily() == ISAFamily::CDNA4 ||
         getISAFamily() == ISAFamily::CDNA5;
}

bool TargetFeatures::supportsCvtPkScalePk8() const { return isCDNA5(); }

bool TargetFeatures::supportsCvtPkScalePk8Upcast() const {
  return supportsCvtPkScalePk8();
}

bool TargetFeatures::supportsCvtPkScalePk8Block16() const {
  return supportsCvtPkScalePk8() && !isGFX1250Strict();
}

ArrayRef<StringRef> TargetFeatures::getUnsupportedWmmaFeatures() const {
  static const StringRef kGFX1250Strict[] = {kWmmaRestrictedInstsFeature};
  if (isGFX1250Strict())
    return kGFX1250Strict;
  return {};
}

bool TargetFeatures::supportsHwScaledUpcast() const {
  return getISAFamily() == ISAFamily::CDNA4 || supportsCvtPkScalePk8Upcast();
}

bool TargetFeatures::supportsHwScaledDowncast() const {
  return getISAFamily() == ISAFamily::CDNA4 || supportsCvtPkScalePk8();
}

bool TargetFeatures::supportBitwidth16Elementwise() const { return true; }

bool TargetFeatures::supportBitwidth32Elementwise() const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
  case ISAFamily::CDNA5:
    return true;
  default:
    return false;
  }
}

bool isCDNA(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
  case ISAFamily::CDNA5:
    return true;
  default:
    return false;
  }
}

bool isRDNA(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::RDNA3:
  case ISAFamily::RDNA4m:
  case ISAFamily::RDNA4:
    return true;
  default:
    return false;
  }
}

} // namespace mlir::triton::amdgpu
