#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

namespace mlir::triton::amdgpu {

namespace {
struct GfxArch {
  unsigned number;
  StringRef suffix;
};

std::optional<GfxArch> parseGfxArch(StringRef arch) {
  if (!arch.consume_front("gfx"))
    return std::nullopt;

  unsigned number = 0;
  bool hasDigits = false;
  while (!arch.empty() && arch.front() >= '0' && arch.front() <= '9') {
    hasDigits = true;
    number = number * 10 + (arch.front() - '0');
    arch = arch.drop_front();
  }

  if (!hasDigits)
    return std::nullopt;

  return GfxArch{number, arch};
}
} // namespace

TargetFeatures::TargetFeatures(std::optional<StringRef> arch)
    : TargetFeatures(arch.value_or("")) {}

TargetFeatures::TargetFeatures(StringRef arch) : arch(arch.str()) {}

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

ISAFamily TargetFeatures::getISAFamily() const {
  std::optional<GfxArch> gfxArch = parseGfxArch(arch);
  if (!gfxArch)
    return ISAFamily::Unknown;

  unsigned number = gfxArch->number;

  // See https://llvm.org/docs/AMDGPUUsage.html#processors for how to
  // categorize the following target gfx architectures. Parse the gfx number
  // directly here instead of depending on LLVM's target parser.
  if (number == 1250)
    return ISAFamily::GFX1250;

  // CDNA ISA cases.
  switch (number) {
  case 950:
    return ISAFamily::CDNA4;
  case 942:
    return ISAFamily::CDNA3;
  case 908:
    return ISAFamily::CDNA1;
  default:
    break;
  }
  if (number == 90 && gfxArch->suffix.starts_with("a"))
    return ISAFamily::CDNA2;
  if (number == 906)
    return ISAFamily::GCN5_1;

  // RDNA ISA cases.
  if (number >= 1200 && number <= 1201)
    return ISAFamily::RDNA4;
  if (number >= 1100 && number <= 1153)
    return ISAFamily::RDNA3;
  if (number >= 1030 && number <= 1036)
    return ISAFamily::RDNA2;
  if (number >= 1010 && number <= 1013)
    return ISAFamily::RDNA1;

  return ISAFamily::Unknown;
}

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

bool TargetFeatures::isGFX1250() const {
  return getISAFamily() == ISAFamily::GFX1250;
}

int TargetFeatures::getWarpSize() const {
  switch (getISAFamily()) {
  case ISAFamily::GCN5_1:
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return 64;
  default:
    return 32;
  }
}

bool TargetFeatures::supportsWaveId() const {
  return getISAFamily() == ISAFamily::RDNA4 ||
         getISAFamily() == ISAFamily::GFX1250;
}

int TargetFeatures::getSharedMemorySize() const {
  // Should return the maximum capacity in bytes.
  switch (getISAFamily()) {
  case ISAFamily::GFX1250:
    return 320 * 1024;
  case ISAFamily::CDNA4:
    return 160 * 1024;
  default:
    return 64 * 1024;
  }
}

size_t TargetFeatures::getSharedMemoryPartitionSize() const {
  switch (getISAFamily()) {
  case ISAFamily::GFX1250:
    return 64 * 1024;
  default:
    // No partitioning on other targets.
    return 0;
  }
}

std::optional<TargetFeatures::LDSTransLoadParams>
TargetFeatures::queryLDSTransLoadParams(int bitWidth) const {
  auto isaFamily = getISAFamily();
  // Determine LDSTrans version: V1 (CDNA4), V2 (GFX1250).
  enum { V1, V2, NONE } version = NONE;
  if (isaFamily == ISAFamily::CDNA4) {
    version = V1;
  } else if (isaFamily == ISAFamily::GFX1250) {
    version = V2;
  }

  if (version == NONE || !llvm::is_contained({16, 8, 4, 6}, bitWidth))
    return std::nullopt;

  unsigned numLanesInShuffleGroup = getWarpSize() / 4;
  unsigned instBitWidth;
  bool doubleB8Contiguity;

  switch (version) {
  case V1:
    instBitWidth = 64;
    doubleB8Contiguity = false;
    break;
  case V2:
    instBitWidth = (bitWidth == 16) ? 128 : 64;
    doubleB8Contiguity = (bitWidth == 8);
    break;
  default:
    return std::nullopt;
  }

  unsigned tileSize = instBitWidth / bitWidth;
  return LDSTransLoadParams{numLanesInShuffleGroup, instBitWidth, tileSize,
                            doubleB8Contiguity};
}

bool TargetFeatures::supportsDirectToLdsScatter() const { return isGFX1250(); }

bool TargetFeatures::supportsDirectToLdsLoadBitWidth(int bitWidth) const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA3:
    // Disable 8 and 16 bits because they get extended to 32 bit.
    return llvm::is_contained({32, /*16, 8*/}, bitWidth);
  case ISAFamily::CDNA4:
    // Disable 8, 16, 96 bits because they get extended to 32/128 bit.
    return llvm::is_contained({128, /*96, */ 32, /*16, 8*/}, bitWidth);
  case ISAFamily::GFX1250:
    // Disable 8, 16 bits because they get extended to 32 bit and therefore
    // overwrite. 96 is not a pow2 and generally not useful in Triton.
    return llvm::is_contained({128, 64, /*96, */ 32, /*16, 8*/}, bitWidth);
  default:
    break;
  }

  return false;
}

bool TargetFeatures::supportsDirectFromLdsStoreBitWidth(int bitWidth) const {
  if (getISAFamily() == ISAFamily::GFX1250) {
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

bool TargetFeatures::supportsTDM() const { return isGFX1250(); }

bool TargetFeatures::supportsMultiCTALaunch() const { return isGFX1250(); }

bool TargetFeatures::supportsClusterLoadBitWidth(int bitWidth) const {
  if (getISAFamily() == ISAFamily::GFX1250) {
    return llvm::is_contained({32, 64, 128}, bitWidth);
  }
  return false;
}

bool TargetFeatures::supportsBufferAtomicRMW() const {
  return llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4,
                             ISAFamily::RDNA4, ISAFamily::GFX1250},
                            getISAFamily());
}

bool TargetFeatures::supportsBufferAtomicFadd(Type elementType) const {
  auto isaFamily = getISAFamily();
  if (isaFamily == ISAFamily::CDNA3 && elementType.isBF16())
    return false;
  if (isaFamily == ISAFamily::RDNA4 && elementType.isF64())
    return false;
  return true;
}

int32_t TargetFeatures::getBufferAtomicCachePolicy(bool hasUsers) const {
  const int sc0Bit = 0b1;          // TH_ATOMIC_RETURN (cpol bit 0)
  const int scopeDevBit = 0b10000; // SCOPE_DEV = 2 << 3 (cpol bits [4:3])
  int32_t aux = 0;
  if (hasUsers)
    aux |= sc0Bit;
  if (getISAFamily() == ISAFamily::GFX1250)
    aux |= scopeDevBit;
  return aux;
}

bool TargetFeatures::supportMaximumMinimum() const {
  return getISAFamily() == ISAFamily::CDNA4 ||
         getISAFamily() == ISAFamily::GFX1250;
}

bool TargetFeatures::supportDppBroadcast() const {
  switch (getISAFamily()) {
  case ISAFamily::GCN5_1:
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return true;
  case ISAFamily::GFX1250:
    return false;
  default:
    return false;
  }
}

bool TargetFeatures::supportsPermlaneSwap() const {
  return getISAFamily() == ISAFamily::CDNA4 ||
         getISAFamily() == ISAFamily::GFX1250;
}

bool TargetFeatures::supportsCvtPkScalePk8() const { return isGFX1250(); }

bool TargetFeatures::supportsHwScaledUpcast() const {
  return getISAFamily() == ISAFamily::CDNA4 ||
         getISAFamily() == ISAFamily::GFX1250;
}

bool TargetFeatures::supportBitwidth16Elementwise() const { return true; }

bool TargetFeatures::supportBitwidth32Elementwise() const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
  case ISAFamily::GFX1250:
    return true;
  default:
    return false;
  }
}

bool isCDNA(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
  case ISAFamily::GFX1250:
    return true;
  default:
    return false;
  }
}

bool isRDNA(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::RDNA1:
  case ISAFamily::RDNA2:
  case ISAFamily::RDNA3:
  case ISAFamily::RDNA4:
    return true;
  default:
    return false;
  }
}

} // namespace mlir::triton::amdgpu
