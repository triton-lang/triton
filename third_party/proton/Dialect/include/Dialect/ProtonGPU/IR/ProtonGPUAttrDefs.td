#ifndef PROTONGPU_ATTR_DEFS
#define PROTONGPU_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"

def GranularityAttr : I32EnumAttr<
  "Granularity", "The granularity of the profiling metric",
  [
    I32EnumAttrCase<"THREAD", 0, "thread">,
    I32EnumAttrCase<"WARP", 1, "warp">,
    I32EnumAttrCase<"WARP_2", 2, "warp_2">,
    I32EnumAttrCase<"WARP_4", 3, "warp_4">,
    I32EnumAttrCase<"WARP_8", 4, "warp_8">,
    I32EnumAttrCase<"CTA", 5, "cta">,
    I32EnumAttrCase<"WARP_GROUP", 6, "warp_group">,
    I32EnumAttrCase<"WARP_GROUP_2", 7, "warp_group_2">,
    I32EnumAttrCase<"WARP_GROUP_4", 8, "warp_group_4">,
    I32EnumAttrCase<"WARP_GROUP_8", 9, "warp_group_8">,
  ]> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let description = [{
    The granularity can be per CTA, per warp, or per warp group.
    The following granularity levels are supported:
    - THREAD: Metrics are recorded per thread.
    - CTA: Metrics are recorded per CTA.
    - WARP: Metrics are recorded per warp.
    - WARP_2, WARP_4, WARP_8: Metrics are recorded for every 2, 4, or 8 warps, respectively.
    - WARP_GROUP: Metrics are recorded per warp group.
    - WARP_GROUP_2, WARP_GROUP_4, WARP_GROUP_8: Metrics are recorded for every 2, 4, or 8 warp groups, respectively.
  }];
}

def BufferStrategyAttr : I32EnumAttr<
  "BufferStrategy", "The strategy for buffer management",
  [
    I32EnumAttrCase<"CIRCULAR", 0, "circular">,
    I32EnumAttrCase<"FLUSH", 1, "flush">,
  ]> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let description = [{
    The following buffer management strategies are supported:
    - CIRCULAR: Circular buffer management strategy. Out of space is handled by overwriting the oldest data.
    - FLUSH: Flush buffer management strategy. Once the GPU buffer is full, data is flushed to the host.
  }];
}

def BufferTypeAttr : I32EnumAttr<
  "BufferType", "The type of internal buffer to be used",
  [
    I32EnumAttrCase<"SHARED", 1, "shared">,
    I32EnumAttrCase<"GLOBAL", 2, "global">,
  ]> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let description = [{
    The following buffer types are supported:
    - SHARED: Shared memory buffer type.
    - GLOBAL: Profiling data get stored directly in global memory, but may be cached in L2/L1.
  }];
}

def PTG_GlobalMemorySpace : AttrDef<ProtonGPU_Dialect, "GlobalMemorySpace"> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let mnemonic = "global_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to global memory.
  }];
}

#endif // PROTONGPU_ATTR_DEFS
