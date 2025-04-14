#ifndef PROTONGPU_ATTR_DEFS
#define PROTONGPU_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"

def GranularityAttr : I32EnumAttr<
  "Granularity", "The granularity of the profiling metric",
  [
    I32EnumAttrCase<"CTA", 0, "cta">,
    I32EnumAttrCase<"WARP", 1, "warp">,
    I32EnumAttrCase<"WARP_2", 2, "warp_2">,
    I32EnumAttrCase<"WARP_4", 3, "warp_4">,
    I32EnumAttrCase<"WARP_8", 4, "warp_8">,
    I32EnumAttrCase<"WARP_GROUP", 5, "warp_group">,
    I32EnumAttrCase<"WARP_GROUP_2", 6, "warp_group_2">,
    I32EnumAttrCase<"WARP_GROUP_4", 7, "warp_group_4">,
    I32EnumAttrCase<"WARP_GROUP_8", 8, "warp_group_8">,
  ]> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let description = [{
    The granularity can be per CTA, per warp, or per warp group.
    The following granularity levels are supported:
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
  "BufferType", "The type of buffer to be used",
  [
    I32EnumAttrCase<"DEFAULT", 0, "default">,
    I32EnumAttrCase<"SHARED", 1, "shared">,
    I32EnumAttrCase<"GLOBAL", 2, "global">,
    I32EnumAttrCase<"LOCAL", 3, "local">,
  ]> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let description = [{
    The following buffer types are supported:
    - DEFAULT: Default buffer type.
    - SHARED: Shared buffer type.
    - GLOBAL: Global buffer type.
    - LOCAL: Local buffer type.
  }];
}


def PTG_StackMemorySpace : AttrDef<ProtonGPU_Dialect, "StackMemorySpace"> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let mnemonic = "stack_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to stack memory.
  }];
}

def PTG_HeapMemorySpace : AttrDef<ProtonGPU_Dialect, "HeapMemorySpace"> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let mnemonic = "heap_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to heap memory.
  }];
}

#endif // PROTONGPU_ATTR_DEFS
