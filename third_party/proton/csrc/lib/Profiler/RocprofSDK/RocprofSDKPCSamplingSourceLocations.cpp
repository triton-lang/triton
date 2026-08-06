#include "Profiler/RocprofSDK/RocprofSDKPCSampling.h"

#if PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING

#include "Utility/String.h"

#if PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE
#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"
#endif

#include <charconv>
#include <iostream>
#include <system_error>
#include <utility>

namespace proton {

namespace {

// Code-object images are copied so rocprofiler-sdk can resolve sampled PCs.
// Cap the copy to avoid unbounded memory use.
constexpr uint64_t MaxCodeObjectImageSize = 256 * 1024 * 1024;

} // namespace

RocprofSDKPCSampling::MetadataState::MetadataState() = default;

RocprofSDKPCSampling::MetadataState::~MetadataState() = default;

std::optional<RocprofSDKPCSampling::SourceLocation>
RocprofSDKPCSampling::parseSourceLocationComment(
    const std::string &comment, const std::string &fallbackFunction) {
  auto separator = comment.find(" -> ");
  auto fileLine = trim(comment.substr(0, separator));
  if (fileLine.empty())
    return std::nullopt;

  auto lastColon = fileLine.rfind(':');
  if (lastColon == std::string::npos)
    return std::nullopt;

  SourceLocation location;
  location.file = fileLine.substr(0, lastColon);
  auto line = fileLine.substr(lastColon + 1);
  auto begin = line.data();
  auto end = begin + line.size();
  auto [ptr, ec] = std::from_chars(begin, end, location.line);
  if (ec != std::errc{} || ptr != end)
    return std::nullopt;
  if (location.file.empty() || location.line == 0)
    return std::nullopt;
  location.function = fallbackFunction;
  return location;
}

void RocprofSDKPCSampling::recordCodeObjectLoad(
    const rocprofiler_callback_tracing_code_object_load_data_t &load,
    bool pcSamplingModeEnabled) {
  if (load.code_object_id == ROCPROFILER_CODE_OBJECT_ID_NONE)
    return;

  CodeObjectInfo info;
  info.codeObjectId = load.code_object_id;
  info.loadSize = load.load_size;
  info.loadDelta = load.load_delta;

#if PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE
  if (pcSamplingModeEnabled) {
    if (load.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
      if (load.memory_base != 0 && load.memory_size != 0 &&
          load.memory_size <= MaxCodeObjectImageSize) {
        const auto *base = reinterpret_cast<const char *>(load.memory_base);
        info.image.assign(base, base + load.memory_size);
      }
    }
  }
#endif

  metadataState.withLock([&](MetadataState &state) {
    replaceCodeObjectLocked(state, std::move(info));
  });
}

void RocprofSDKPCSampling::replaceCodeObjectLocked(MetadataState &state,
                                                   CodeObjectInfo info) {
  const uint64_t codeObjectId = info.codeObjectId;
  auto oldInfo = state.codeObjects.find(codeObjectId);
  if (oldInfo != state.codeObjects.end())
    removeSourceLocationDecoderLocked(state, oldInfo->second);
  state.codeObjects.insert_or_assign(codeObjectId, std::move(info));
  clearSourceLocationCacheLocked(state, codeObjectId);
}

void RocprofSDKPCSampling::removeSourceLocationDecoderLocked(
    MetadataState &state, const CodeObjectInfo &info) {
#if PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE
  if (state.sourceLocationTranslator && info.decoderRegistered) {
    try {
      state.sourceLocationTranslator->removeDecoder(
          info.codeObjectId, static_cast<uint64_t>(info.loadDelta));
    } catch (...) {
      // Removal is best-effort because the decoder may have failed to register.
    }
  }
#else
  (void)info;
#endif
}

void RocprofSDKPCSampling::recordCodeObjectUnload(uint64_t codeObjectId) {
  if (codeObjectId == ROCPROFILER_CODE_OBJECT_ID_NONE)
    return;

  metadataState.withLock([&](MetadataState &state) {
    auto info = state.codeObjects.find(codeObjectId);
    if (info != state.codeObjects.end())
      info->second.unloaded = true;
    clearSourceLocationCacheLocked(state, codeObjectId);
  });
  if (!pcSamplingStarted)
    tryReleaseCodeObject(codeObjectId);
}

std::optional<RocprofSDKPCSampling::SourceLocation>
RocprofSDKPCSampling::resolveSourceLocationLocked(
    MetadataState &state, uint64_t codeObjectId, uint64_t pcOffset,
    const PCSamplingTarget &target) {
#if !PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE
  (void)state;
  (void)pcOffset;
  (void)target;
  if (codeObjectId == ROCPROFILER_CODE_OBJECT_ID_NONE)
    return std::nullopt;
  return std::nullopt;
#else
  if (codeObjectId == ROCPROFILER_CODE_OBJECT_ID_NONE) {
    return std::nullopt;
  }

  SourceLocationKey key{codeObjectId, pcOffset};
  auto cached = state.sourceLocationCache.find(key);
  if (cached != state.sourceLocationCache.end())
    return cached->second;

  std::optional<SourceLocation> resolved;
  if (ensureSourceLocationDecoderLocked(state, codeObjectId) &&
      state.sourceLocationTranslator) {
    try {
      auto inst = state.sourceLocationTranslator->get(codeObjectId, pcOffset);
      if (inst && !inst->comment.empty())
        resolved = parseSourceLocationComment(inst->comment, target.kernelName);
    } catch (...) {
      // Fall back to kernel-level attribution if translation fails.
    }
  }
  state.sourceLocationCache.insert_or_assign(key, resolved);
  return resolved;
#endif
}

bool RocprofSDKPCSampling::ensureSourceLocationDecoderLocked(
    MetadataState &state, uint64_t codeObjectId) {
#if !PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE
  (void)state;
  (void)codeObjectId;
  return false;
#else
  auto codeObject = state.codeObjects.find(codeObjectId);
  if (codeObject == state.codeObjects.end())
    return false;
  auto &info = codeObject->second;
  if (info.decoderRegistered)
    return true;
  if (info.image.empty())
    return false;
  if (!state.sourceLocationTranslator) {
    using ::rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;
    state.sourceLocationTranslator =
        std::make_unique<CodeobjAddressTranslate>();
  }
  try {
    state.sourceLocationTranslator->addDecoder(
        info.image.data(), info.image.size(), info.codeObjectId,
        static_cast<uint64_t>(info.loadDelta), info.loadSize);
    info.decoderRegistered = true;
    return true;
  } catch (...) {
    // A decoder failure should only disable source attribution for this
    // object.
    return false;
  }
#endif
}

void RocprofSDKPCSampling::clearSourceLocationCacheLocked(
    MetadataState &state, uint64_t codeObjectId) {
  for (auto it = state.sourceLocationCache.begin();
       it != state.sourceLocationCache.end();) {
    if (it->first.codeObjectId == codeObjectId) {
      it = state.sourceLocationCache.erase(it);
    } else {
      ++it;
    }
  }
}

void RocprofSDKPCSampling::tryReleaseCodeObject(uint64_t codeObjectId) {
  metadataState.withLock([&](MetadataState &metadata) {
    samplingState.withLock([&](SamplingState &sampling) {
      if (sampling.pendingCodeObjectIds.count(codeObjectId) > 0 ||
          sampling.flushingCodeObjectIds.count(codeObjectId) > 0)
        return;

      auto info = metadata.codeObjects.find(codeObjectId);
      if (info == metadata.codeObjects.end() || !info->second.unloaded)
        return;
      removeSourceLocationDecoderLocked(metadata, info->second);
      clearSourceLocationCacheLocked(metadata, codeObjectId);
      metadata.codeObjects.erase(info);
    });
  });
}

} // namespace proton

#endif
