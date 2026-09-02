// SPDX-License-Identifier: Apache-2.0

#include "launch_abi.h"

#include <habanalabs/tpc_kernel_lib_interface.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using triton::gaudi::LaunchParamsV1;

constexpr std::uintmax_t kMaxElfBytes = 64ULL * 1024ULL * 1024ULL;

#if defined(__GNUC__)
#define TRITON_GAUDI_PERF_EXPORT __attribute__((visibility("default")))
#else
#define TRITON_GAUDI_PERF_EXPORT
#endif

bool isLowerHexHash(const std::array<char, triton::gaudi::kArtifactHashChars + 1>& hash) {
  if (hash.back() != '\0') {
    return false;
  }
  return std::all_of(hash.begin(), hash.end() - 1, [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

const LaunchParamsV1* getLaunchParams(const tpc_lib_api::HabanaKernelParams* params) {
  if (params == nullptr || params->nodeParams.nodeParams == nullptr ||
      params->nodeParams.nodeParamsSize != sizeof(LaunchParamsV1)) {
    return nullptr;
  }
  const auto* launch = static_cast<const LaunchParamsV1*>(params->nodeParams.nodeParams);
  if (launch->magic != triton::gaudi::kLaunchParamsMagic ||
      launch->abi_major != triton::gaudi::kLaunchAbiMajor ||
      launch->abi_minor > triton::gaudi::kLaunchAbiMinor ||
      launch->index_space_rank == 0 ||
      launch->index_space_rank > triton::gaudi::kMaxIndexSpaceRank ||
      launch->scalar_count > triton::gaudi::kMaxScalarParams ||
      (launch->kernel_kind != triton::gaudi::KernelKind::Elementwise &&
       launch->kernel_kind != triton::gaudi::KernelKind::FusedAddRmsNorm &&
       launch->kernel_kind != triton::gaudi::KernelKind::SiluAndMul &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnDecodePacked &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnDecodeConvPacked &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnQkConvPacked &&
       launch->kernel_kind !=
           triton::gaudi::KernelKind::GdnDecodeValueConvPacked) ||
      !isLowerHexHash(launch->artifact_hash)) {
    return nullptr;
  }
  return launch;
}

fs::path artifactPath(const LaunchParamsV1& launch) {
  const char* directory = std::getenv("TRITON_GAUDI_ARTIFACT_DIR");
  if (directory == nullptr || *directory == '\0') {
    return {};
  }
  return fs::path(directory) / (std::string(launch.artifact_hash.data()) + ".elf");
}

tpc_lib_api::GlueCodeReturn loadElf(
    const LaunchParamsV1& launch,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  const fs::path path = artifactPath(launch);
  std::error_code error;
  if (path.empty() || !fs::is_regular_file(path, error) || fs::is_symlink(path, error)) {
    return tpc_lib_api::GLUE_FAILED;
  }
  const auto file_size = fs::file_size(path, error);
  if (error || file_size < 4 || file_size > kMaxElfBytes ||
      file_size > std::numeric_limits<std::uint32_t>::max()) {
    return tpc_lib_api::GLUE_FAILED;
  }

  const auto required = static_cast<std::uint32_t>(file_size);
  const auto available = instance->kernel.elfSize;
  instance->kernel.elfSize = required;
  if (instance->kernel.kernelElf == nullptr || available < required) {
    return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
  }

  std::ifstream input(path, std::ios::binary);
  if (!input.read(static_cast<char*>(instance->kernel.kernelElf), required)) {
    return tpc_lib_api::GLUE_FAILED;
  }
  const auto* elf = static_cast<const unsigned char*>(instance->kernel.kernelElf);
  if (!(elf[0] == 0x7f && elf[1] == 'E' && elf[2] == 'L' && elf[3] == 'F')) {
    return tpc_lib_api::GLUE_FAILED;
  }
  return tpc_lib_api::GLUE_SUCCESS;
}

void setAccessPattern(
    tpc_lib_api::TensorAccessPattern& pattern,
    const tpc_lib_api::Tensor& tensor,
    std::uint32_t elements_per_program,
    bool broadcast = false) {
  std::memset(&pattern, 0, sizeof(pattern));
  pattern.mapping[0].indexSpaceDim = 0;
  pattern.mapping[0].a = broadcast
      ? 0.0F
      : static_cast<float>(elements_per_program);
  pattern.mapping[0].start_b = 0.0F;
  pattern.mapping[0].end_b = static_cast<float>(elements_per_program - 1);
  pattern.mapping[0].allRequired = broadcast;
  for (std::uint32_t dimension = 1; dimension < tensor.geometry.dims; ++dimension) {
    pattern.mapping[dimension].indexSpaceDim = 0;
    pattern.mapping[dimension].a = 0.0F;
    pattern.mapping[dimension].start_b = 0.0F;
    pattern.mapping[dimension].end_b =
        static_cast<float>(tensor.geometry.maxSizes[dimension] - 1);
    pattern.mapping[dimension].allRequired = true;
  }
}

void setSiluAndMulAccessPattern(
    tpc_lib_api::TensorAccessPattern& pattern,
    std::uint32_t chunk_size,
    std::uint32_t n_cols,
    bool input) {
  std::memset(&pattern, 0, sizeof(pattern));
  pattern.mapping[0].indexSpaceDim = 0;
  pattern.mapping[0].a = static_cast<float>(chunk_size);
  pattern.mapping[0].start_b = 0.0F;
  pattern.mapping[0].end_b = static_cast<float>(
      (input ? n_cols : 0) + chunk_size - 1);
  pattern.mapping[1].indexSpaceDim = 1;
  pattern.mapping[1].a = 1.0F;
  pattern.mapping[1].start_b = 0.0F;
  pattern.mapping[1].end_b = 0.0F;
}

void mapDimension(
    tpc_lib_api::TensorAccessPattern& pattern,
    unsigned tensor_dimension,
    unsigned index_dimension,
    float a,
    float start_b,
    float end_b,
    bool all_required = false) {
  pattern.mapping[tensor_dimension].indexSpaceDim = index_dimension;
  pattern.mapping[tensor_dimension].a = a;
  pattern.mapping[tensor_dimension].start_b = start_b;
  pattern.mapping[tensor_dimension].end_b = end_b;
  pattern.mapping[tensor_dimension].allRequired = all_required;
}

void setGdnDecodeAccessPatterns(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance,
    std::uint32_t value_tile) {
  for (std::uint16_t index = 0; index < 7; ++index) {
    std::memset(
        &instance->inputTensorAccessPattern[index],
        0,
        sizeof(instance->inputTensorAccessPattern[index]));
  }
  std::memset(
      &instance->outputTensorAccessPattern[0],
      0,
      sizeof(instance->outputTensorAccessPattern[0]));

  auto& state = instance->inputTensorAccessPattern[0];
  mapDimension(state, 0, 0, 0, 0, 127, true);
  mapDimension(state, 1, 0, value_tile, 0, value_tile - 1);
  mapDimension(state, 2, 1, 1, 0, 0);
  mapDimension(
      state,
      3,
      2,
      0,
      0,
      static_cast<float>(params->inputTensors[0].geometry.maxSizes[3] - 1),
      true);

  auto& packed = instance->inputTensorAccessPattern[1];
  mapDimension(packed, 0, 1, 0, 0, 10239, true);
  mapDimension(packed, 1, 2, 1, 0, 0);

  for (unsigned input = 2; input <= 3; ++input) {
    auto& gate = instance->inputTensorAccessPattern[input];
    mapDimension(gate, 0, 1, 1, 0, 0);
    mapDimension(gate, 1, 2, 1, 0, 0);
  }
  for (unsigned input = 4; input <= 5; ++input) {
    mapDimension(
        instance->inputTensorAccessPattern[input], 0, 1, 1, 0, 0);
  }
  mapDimension(instance->inputTensorAccessPattern[6], 0, 2, 1, 0, 0);

  auto& output = instance->outputTensorAccessPattern[0];
  mapDimension(output, 0, 0, value_tile, 0, value_tile - 1);
  mapDimension(output, 1, 1, 1, 0, 0);
  mapDimension(output, 2, 2, 1, 0, 0);
}

void setGdnDecodeConvAccessPatterns(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  for (std::uint16_t index = 0; index < 9; ++index) {
    std::memset(
        &instance->inputTensorAccessPattern[index],
        0,
        sizeof(instance->inputTensorAccessPattern[index]));
  }
  std::memset(
      &instance->outputTensorAccessPattern[0],
      0,
      sizeof(instance->outputTensorAccessPattern[0]));

  auto& conv_state = instance->inputTensorAccessPattern[0];
  mapDimension(conv_state, 0, 0, 128, 0, 4479);
  mapDimension(conv_state, 1, 0, 0, 0, 2, true);
  mapDimension(
      conv_state,
      2,
      1,
      0,
      0,
      static_cast<float>(params->inputTensors[0].geometry.maxSizes[2] - 1),
      true);

  auto& recurrent_state = instance->inputTensorAccessPattern[1];
  mapDimension(recurrent_state, 0, 0, 0, 0, 127, true);
  mapDimension(recurrent_state, 1, 0, 0, 0, 127, true);
  mapDimension(recurrent_state, 2, 0, 3, 0, 2);
  mapDimension(
      recurrent_state,
      3,
      1,
      0,
      0,
      static_cast<float>(params->inputTensors[1].geometry.maxSizes[3] - 1),
      true);

  auto& packed = instance->inputTensorAccessPattern[2];
  mapDimension(packed, 0, 0, 128, 0, 4479);
  mapDimension(packed, 1, 1, 1, 0, 0);
  for (unsigned input = 3; input <= 4; ++input) {
    auto& gate = instance->inputTensorAccessPattern[input];
    mapDimension(gate, 0, 0, 3, 0, 2);
    mapDimension(gate, 1, 1, 1, 0, 0);
  }
  for (unsigned input = 5; input <= 6; ++input) {
    mapDimension(instance->inputTensorAccessPattern[input], 0, 0, 3, 0, 2);
  }
  mapDimension(instance->inputTensorAccessPattern[7], 0, 1, 1, 0, 0);
  auto& weight = instance->inputTensorAccessPattern[8];
  mapDimension(weight, 0, 0, 128, 0, 4479);
  mapDimension(weight, 1, 0, 0, 0, 3, true);

  auto& output = instance->outputTensorAccessPattern[0];
  mapDimension(output, 0, 0, 0, 0, 127, true);
  mapDimension(output, 1, 0, 3, 0, 2);
  mapDimension(output, 2, 1, 1, 0, 0);
}

void setGdnQkConvAccessPatterns(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  for (std::uint16_t index = 0; index < 4; ++index) {
    std::memset(
        &instance->inputTensorAccessPattern[index],
        0,
        sizeof(instance->inputTensorAccessPattern[index]));
  }
  std::memset(
      &instance->outputTensorAccessPattern[0],
      0,
      sizeof(instance->outputTensorAccessPattern[0]));
  auto& conv_state = instance->inputTensorAccessPattern[0];
  mapDimension(conv_state, 0, 0, 128, 0, 127);
  mapDimension(conv_state, 1, 0, 0, 0, 2, true);
  mapDimension(
      conv_state,
      2,
      1,
      0,
      0,
      static_cast<float>(params->inputTensors[0].geometry.maxSizes[2] - 1),
      true);
  auto& packed = instance->inputTensorAccessPattern[1];
  mapDimension(packed, 0, 0, 128, 0, 127);
  mapDimension(packed, 1, 1, 1, 0, 0);
  mapDimension(instance->inputTensorAccessPattern[2], 0, 1, 1, 0, 0);
  auto& weight = instance->inputTensorAccessPattern[3];
  mapDimension(weight, 0, 0, 128, 0, 127);
  mapDimension(weight, 1, 0, 0, 0, 3, true);
  auto& output = instance->outputTensorAccessPattern[0];
  mapDimension(output, 0, 0, 128, 0, 127);
  mapDimension(output, 1, 1, 1, 0, 0);
}

void setGdnValueConvAccessPatterns(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance,
    std::uint32_t value_tile) {
  for (std::uint16_t index = 0; index < 10; ++index) {
    std::memset(
        &instance->inputTensorAccessPattern[index],
        0,
        sizeof(instance->inputTensorAccessPattern[index]));
  }
  std::memset(
      &instance->outputTensorAccessPattern[0],
      0,
      sizeof(instance->outputTensorAccessPattern[0]));
  auto& conv_state = instance->inputTensorAccessPattern[0];
  mapDimension(conv_state, 0, 0, 0, 4096, 10239, true);
  mapDimension(conv_state, 1, 0, 0, 0, 2, true);
  mapDimension(
      conv_state,
      2,
      2,
      0,
      0,
      static_cast<float>(params->inputTensors[0].geometry.maxSizes[2] - 1),
      true);
  auto& state = instance->inputTensorAccessPattern[1];
  mapDimension(state, 0, 0, 0, 0, 127, true);
  mapDimension(state, 1, 0, value_tile, 0, value_tile - 1);
  mapDimension(state, 2, 1, 1, 0, 0);
  mapDimension(
      state,
      3,
      2,
      0,
      0,
      static_cast<float>(params->inputTensors[1].geometry.maxSizes[3] - 1),
      true);
  auto& qk = instance->inputTensorAccessPattern[2];
  mapDimension(qk, 0, 1, 0, 0, 4095, true);
  mapDimension(qk, 1, 2, 1, 0, 0);
  auto& packed = instance->inputTensorAccessPattern[3];
  mapDimension(packed, 0, 0, 0, 4096, 10239, true);
  mapDimension(packed, 1, 2, 1, 0, 0);
  for (unsigned input = 4; input <= 5; ++input) {
    auto& gate = instance->inputTensorAccessPattern[input];
    mapDimension(gate, 0, 1, 1, 0, 0);
    mapDimension(gate, 1, 2, 1, 0, 0);
  }
  for (unsigned input = 6; input <= 7; ++input) {
    mapDimension(instance->inputTensorAccessPattern[input], 0, 1, 1, 0, 0);
  }
  mapDimension(instance->inputTensorAccessPattern[8], 0, 2, 1, 0, 0);
  auto& weight = instance->inputTensorAccessPattern[9];
  mapDimension(weight, 0, 0, 0, 4096, 10239, true);
  mapDimension(weight, 1, 0, 0, 0, 3, true);
  auto& output = instance->outputTensorAccessPattern[0];
  mapDimension(output, 0, 0, value_tile, 0, value_tile - 1);
  mapDimension(output, 1, 1, 1, 0, 0);
  mapDimension(output, 2, 2, 1, 0, 0);
}

bool hasGeometry(
    const tpc_lib_api::Tensor& tensor,
    tpc_lib_api::TensorDataType dtype,
    std::initializer_list<std::uint64_t> sizes) {
  if (tensor.geometry.dataType != dtype ||
      tensor.geometry.dims != sizes.size()) {
    return false;
  }
  std::size_t dimension = 0;
  for (const auto size : sizes) {
    if (tensor.geometry.maxSizes[dimension++] != size) {
      return false;
    }
  }
  return true;
}

} // namespace

extern "C" TRITON_GAUDI_PERF_EXPORT tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId device_id,
    std::uint32_t* kernel_count,
    tpc_lib_api::GuidInfo* guids) {
  if (kernel_count == nullptr) {
    return tpc_lib_api::GLUE_FAILED;
  }
  if (device_id != tpc_lib_api::DEVICE_ID_GAUDI2) {
    *kernel_count = 0;
    return tpc_lib_api::GLUE_SUCCESS;
  }
  if (guids != nullptr) {
    std::memset(&guids[0], 0, sizeof(guids[0]));
    std::memcpy(
        guids[0].name,
        triton::gaudi::kKernelGuid,
        sizeof(triton::gaudi::kKernelGuid));
  }
  *kernel_count = 1;
  return tpc_lib_api::GLUE_SUCCESS;
}

extern "C" TRITON_GAUDI_PERF_EXPORT std::uint64_t GetLibVersion() {
  // Bump when the fixed GUID or LaunchParams contract changes so Synapse's
  // recipe coherency checks cannot reuse an incompatible perf library.
  return 0x545249544F4E0007ULL;
}

extern "C" TRITON_GAUDI_PERF_EXPORT tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params == nullptr || instance == nullptr ||
      std::strcmp(params->guid.name, triton::gaudi::kKernelGuid) != 0) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  const LaunchParamsV1* launch = getLaunchParams(params);
  if (launch == nullptr || launch->block_size == 0) {
    return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
  }
  if (params->inputTensorNr != launch->input_count) {
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != launch->output_count) {
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  std::uint32_t elements_per_program = launch->block_size;
  if (launch->kernel_kind == triton::gaudi::KernelKind::FusedAddRmsNorm) {
    if (launch->input_count != 3 || launch->output_count != 2 ||
        launch->scalar_count != 1) {
      return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    elements_per_program = launch->logical_size;
    if (elements_per_program == 0 ||
        elements_per_program > launch->block_size) {
      return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
  } else if (launch->kernel_kind == triton::gaudi::KernelKind::SiluAndMul) {
    if (launch->input_count != 1 || launch->output_count != 1 ||
        launch->scalar_count != 0) {
      return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    elements_per_program = launch->logical_size;
    if (elements_per_program == 0 || launch->index_space_rank != 2 ||
        launch->block_size < 128 || launch->block_size > 1024 ||
        (launch->block_size & (launch->block_size - 1)) != 0 ||
        launch->grid[0] !=
            (elements_per_program + launch->block_size - 1) / launch->block_size ||
        launch->grid[1] == 0 ||
        elements_per_program > std::numeric_limits<std::uint32_t>::max() / 2) {
      return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    if (params->inputTensors[0].geometry.dims != 2 ||
        params->inputTensors[0].geometry.maxSizes[0] != 2 * elements_per_program ||
        params->inputTensors[0].geometry.maxSizes[1] != launch->grid[1]) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (params->outputTensors[0].geometry.dims != 2 ||
        params->outputTensors[0].geometry.maxSizes[0] != elements_per_program ||
        params->outputTensors[0].geometry.maxSizes[1] != launch->grid[1]) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
  } else if (
      launch->kernel_kind == triton::gaudi::KernelKind::GdnDecodePacked) {
    const auto batch = launch->grid[2];
    const auto state_slots = launch->scalar_params[0];
    if (launch->input_count != 7 || launch->output_count != 1 ||
        launch->scalar_count != 1 || launch->index_space_rank != 3 ||
        launch->logical_size != 128 ||
        (launch->block_size != 16 && launch->block_size != 32 &&
         launch->block_size != 64 && launch->block_size != 128) ||
        launch->grid[0] != 128 / launch->block_size ||
        launch->grid[1] != 48 || batch == 0 || state_slots == 0) {
      return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    const auto f32 = static_cast<tpc_lib_api::TensorDataType>(1U << 12);
    const auto bf16 = static_cast<tpc_lib_api::TensorDataType>(1U << 8);
    const auto i32 = static_cast<tpc_lib_api::TensorDataType>(1U << 10);
    if (!hasGeometry(params->inputTensors[0], f32, {128, 128, 48, state_slots}) ||
        !hasGeometry(params->inputTensors[1], bf16, {10240, batch}) ||
        !hasGeometry(params->inputTensors[2], bf16, {48, batch}) ||
        !hasGeometry(params->inputTensors[3], bf16, {48, batch}) ||
        !hasGeometry(params->inputTensors[4], f32, {48}) ||
        !hasGeometry(params->inputTensors[5], f32, {48}) ||
        !hasGeometry(params->inputTensors[6], i32, {batch})) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!hasGeometry(params->outputTensors[0], bf16, {128, 48, batch})) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
  } else if (
      launch->kernel_kind ==
      triton::gaudi::KernelKind::GdnDecodeConvPacked) {
    const auto batch = launch->grid[1];
    const auto conv_slots = launch->scalar_params[0];
    const auto state_slots = launch->scalar_params[1];
    if (launch->input_count != 9 || launch->output_count != 1 ||
        launch->scalar_count != 2 || launch->index_space_rank != 2 ||
        launch->logical_size != 128 || launch->block_size != 128 ||
        launch->grid[0] != 16 || batch == 0 || conv_slots == 0 ||
        state_slots == 0) {
      return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    const auto f32 = static_cast<tpc_lib_api::TensorDataType>(1U << 12);
    const auto bf16 = static_cast<tpc_lib_api::TensorDataType>(1U << 8);
    const auto i32 = static_cast<tpc_lib_api::TensorDataType>(1U << 10);
    if (!hasGeometry(params->inputTensors[0], bf16, {10240, 3, conv_slots}) ||
        !hasGeometry(
            params->inputTensors[1], f32, {128, 128, 48, state_slots}) ||
        !hasGeometry(params->inputTensors[2], bf16, {10240, batch}) ||
        !hasGeometry(params->inputTensors[3], bf16, {48, batch}) ||
        !hasGeometry(params->inputTensors[4], bf16, {48, batch}) ||
        !hasGeometry(params->inputTensors[5], f32, {48}) ||
        !hasGeometry(params->inputTensors[6], f32, {48}) ||
        !hasGeometry(params->inputTensors[7], i32, {batch}) ||
        !hasGeometry(params->inputTensors[8], bf16, {10240, 4})) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!hasGeometry(params->outputTensors[0], bf16, {128, 48, batch})) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
  } else if (
      launch->kernel_kind == triton::gaudi::KernelKind::GdnQkConvPacked) {
    const auto batch = launch->grid[1];
    const auto conv_slots = launch->scalar_params[0];
    if (launch->input_count != 4 || launch->output_count != 1 ||
        launch->scalar_count != 1 || launch->index_space_rank != 2 ||
        launch->logical_size != 4096 ||
        (launch->block_size != 128 && launch->block_size != 256 &&
         launch->block_size != 512) ||
        launch->grid[0] != 4096 / launch->block_size || batch == 0 ||
        conv_slots == 0) {
      return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    const auto bf16 = static_cast<tpc_lib_api::TensorDataType>(1U << 8);
    const auto i32 = static_cast<tpc_lib_api::TensorDataType>(1U << 10);
    if (!hasGeometry(params->inputTensors[0], bf16, {10240, 3, conv_slots}) ||
        !hasGeometry(params->inputTensors[1], bf16, {10240, batch}) ||
        !hasGeometry(params->inputTensors[2], i32, {batch}) ||
        !hasGeometry(params->inputTensors[3], bf16, {10240, 4})) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!hasGeometry(params->outputTensors[0], bf16, {4096, batch})) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
  } else if (
      launch->kernel_kind ==
      triton::gaudi::KernelKind::GdnDecodeValueConvPacked) {
    const auto batch = launch->grid[2];
    const auto conv_slots = launch->scalar_params[0];
    const auto state_slots = launch->scalar_params[1];
    if (launch->input_count != 10 || launch->output_count != 1 ||
        launch->scalar_count != 2 || launch->index_space_rank != 3 ||
        launch->logical_size != 128 ||
        (launch->block_size != 16 && launch->block_size != 32 &&
         launch->block_size != 64 && launch->block_size != 128) ||
        launch->grid[0] != 128 / launch->block_size ||
        launch->grid[1] != 48 || batch == 0 || conv_slots == 0 ||
        state_slots == 0) {
      return tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    const auto f32 = static_cast<tpc_lib_api::TensorDataType>(1U << 12);
    const auto bf16 = static_cast<tpc_lib_api::TensorDataType>(1U << 8);
    const auto i32 = static_cast<tpc_lib_api::TensorDataType>(1U << 10);
    if (!hasGeometry(params->inputTensors[0], bf16, {10240, 3, conv_slots}) ||
        !hasGeometry(
            params->inputTensors[1], f32, {128, 128, 48, state_slots}) ||
        !hasGeometry(params->inputTensors[2], bf16, {4096, batch}) ||
        !hasGeometry(params->inputTensors[3], bf16, {10240, batch}) ||
        !hasGeometry(params->inputTensors[4], bf16, {48, batch}) ||
        !hasGeometry(params->inputTensors[5], bf16, {48, batch}) ||
        !hasGeometry(params->inputTensors[6], f32, {48}) ||
        !hasGeometry(params->inputTensors[7], f32, {48}) ||
        !hasGeometry(params->inputTensors[8], i32, {batch}) ||
        !hasGeometry(params->inputTensors[9], bf16, {10240, 4})) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (!hasGeometry(params->outputTensors[0], bf16, {128, 48, batch})) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
  }

  instance->indexSpaceRank = launch->index_space_rank;
  for (std::uint16_t dimension = 0; dimension < launch->index_space_rank; ++dimension) {
    instance->indexSpaceGeometry[dimension] = launch->grid[dimension];
  }
  if (launch->kernel_kind == triton::gaudi::KernelKind::GdnDecodePacked) {
    setGdnDecodeAccessPatterns(params, instance, launch->block_size);
  } else if (
      launch->kernel_kind ==
      triton::gaudi::KernelKind::GdnDecodeConvPacked) {
    setGdnDecodeConvAccessPatterns(params, instance);
  } else if (
      launch->kernel_kind == triton::gaudi::KernelKind::GdnQkConvPacked) {
    setGdnQkConvAccessPatterns(params, instance);
  } else if (
      launch->kernel_kind ==
      triton::gaudi::KernelKind::GdnDecodeValueConvPacked) {
    setGdnValueConvAccessPatterns(params, instance, launch->block_size);
  }
  for (std::uint16_t index = 0;
       index < launch->input_count &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnDecodePacked &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnDecodeConvPacked &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnQkConvPacked &&
       launch->kernel_kind !=
           triton::gaudi::KernelKind::GdnDecodeValueConvPacked;
       ++index) {
    if (params->inputTensors[index].geometry.dataType !=
        static_cast<tpc_lib_api::TensorDataType>(launch->tensor_dtype)) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (launch->kernel_kind == triton::gaudi::KernelKind::SiluAndMul) {
      setSiluAndMulAccessPattern(
          instance->inputTensorAccessPattern[index],
          launch->block_size,
          elements_per_program,
          true);
    } else {
      setAccessPattern(
          instance->inputTensorAccessPattern[index],
          params->inputTensors[index],
          elements_per_program,
          launch->kernel_kind == triton::gaudi::KernelKind::FusedAddRmsNorm &&
              index == 2);
    }
  }
  for (std::uint16_t index = 0;
       index < launch->output_count &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnDecodePacked &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnDecodeConvPacked &&
       launch->kernel_kind != triton::gaudi::KernelKind::GdnQkConvPacked &&
       launch->kernel_kind !=
           triton::gaudi::KernelKind::GdnDecodeValueConvPacked;
       ++index) {
    if (params->outputTensors[index].geometry.dataType !=
        static_cast<tpc_lib_api::TensorDataType>(launch->tensor_dtype)) {
      return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (launch->kernel_kind == triton::gaudi::KernelKind::SiluAndMul) {
      setSiluAndMulAccessPattern(
          instance->outputTensorAccessPattern[index],
          launch->block_size,
          elements_per_program,
          false);
    } else {
      setAccessPattern(
          instance->outputTensorAccessPattern[index],
          params->outputTensors[index],
          elements_per_program);
    }
  }

  instance->kernel.paramsNr = launch->scalar_count;
  std::copy_n(
      launch->scalar_params.begin(),
      launch->scalar_count,
      instance->kernel.scalarParams);
  return loadElf(*launch, instance);
}

extern "C" TRITON_GAUDI_PERF_EXPORT tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId,
    const tpc_lib_api::ShapeInferenceParams*,
    tpc_lib_api::ShapeInferenceOutput*) {
  // GaudiKernelArtifactV1 currently describes static, shape-preserving
  // elementwise kernels. Synapse already owns the concrete output geometry,
  // but requires every external perf library to expose this entry point.
  return tpc_lib_api::GLUE_SUCCESS;
}

#undef TRITON_GAUDI_PERF_EXPORT
