// SPDX-License-Identifier: Apache-2.0

#ifndef TRITON_GAUDI_RUNTIME_LAUNCH_ABI_H_
#define TRITON_GAUDI_RUNTIME_LAUNCH_ABI_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace triton::gaudi {

inline constexpr std::uint32_t kLaunchParamsMagic = 0x31475452U;
inline constexpr std::uint16_t kLaunchAbiMajor = 1;
inline constexpr std::uint16_t kLaunchAbiMinor = 8;
inline constexpr std::size_t kArtifactHashChars = 64;
inline constexpr std::size_t kMaxIndexSpaceRank = 5;
inline constexpr std::size_t kMaxScalarParams = 32;
inline constexpr char kKernelGuid[] = "triton_gaudi2_v1";

enum class KernelKind : std::uint32_t {
  Elementwise = 0,
  FusedAddRmsNorm = 1,
  SiluAndMul = 2,
  GdnDecodePacked = 3,
  GdnDecodeConvPacked = 4,
  GdnQkConvPacked = 5,
  GdnDecodeValueConvPacked = 6,
  DynamicQuant = 7,
};

// This is copied byte-for-byte through Synapse nodeParams.  Keep the layout
// standard and fixed-width; scalar_params contains the bit representation that
// InstantiateTpcKernel forwards to DeviceKernel::scalarParams.
struct LaunchParamsV1 {
  std::uint32_t magic{kLaunchParamsMagic};
  std::uint16_t abi_major{kLaunchAbiMajor};
  std::uint16_t abi_minor{kLaunchAbiMinor};
  std::uint16_t input_count{0};
  std::uint16_t output_count{0};
  std::uint16_t scalar_count{0};
  std::uint16_t index_space_rank{1};
  std::uint32_t block_size{0};
  std::uint32_t logical_size{0};
  std::uint32_t tensor_dtype{0};
  KernelKind kernel_kind{KernelKind::Elementwise};
  std::array<std::uint64_t, kMaxIndexSpaceRank> grid{};
  std::array<std::uint32_t, kMaxScalarParams> scalar_params{};
  std::array<char, kArtifactHashChars + 1> artifact_hash{};
};

static_assert(std::is_standard_layout_v<LaunchParamsV1>);
static_assert(std::is_trivially_copyable_v<LaunchParamsV1>);

} // namespace triton::gaudi

#endif // TRITON_GAUDI_RUNTIME_LAUNCH_ABI_H_
