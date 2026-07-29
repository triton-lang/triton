#ifndef PROTON_ATTR_DEFS
#define PROTON_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

def MetricTypeAttr : I32EnumAttr<
  "MetricType", "The type of metric to be profiled",
  [
    I32EnumAttrCase<"CYCLE", 0, "cycle">,
  ]> {
  let cppNamespace = "::mlir::triton::proton";
  let description = [{
    Attribute to indicate the metric to be profiled.
    The following metrics are supported:
    - CYCLE: Cycle count metric.
  }];
}

def MetricValueTypeAttr : I32EnumAttr<
  "MetricValueType", "The scalar type of an in-kernel metric",
  [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"BOOL", 1, "bool">,
    I32EnumAttrCase<"I8", 2, "i8">,
    I32EnumAttrCase<"I16", 3, "i16">,
    I32EnumAttrCase<"I32", 4, "i32">,
    I32EnumAttrCase<"U8", 5, "u8">,
    I32EnumAttrCase<"U16", 6, "u16">,
    I32EnumAttrCase<"U32", 7, "u32">,
    I32EnumAttrCase<"F16", 8, "f16">,
    I32EnumAttrCase<"BF16", 9, "bf16">,
    I32EnumAttrCase<"F32", 10, "f32">
  ]> {
  let cppNamespace = "::mlir::triton::proton";
}

def SamplingStrategyAttr : I32EnumAttr<
  "SamplingStrategy", "The strategy for sampling the profiling data",
  [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"SELECTIVE", 1, "selective">,
  ]> {
  let cppNamespace = "::mlir::triton::proton";
  let description = [{
    Attribute to indicate the sampling strategy for profiling.
    The following sampling strategies are supported:
    - NONE: No sampling.
    - SELECTIVE: Manually select a couple of instances to profile.
  }];
}

def ModeAttr : I32EnumAttr<
  "Mode", "The mode of profiling",
  [
    I32EnumAttrCase<"DEFAULT", 0, "default">,
    I32EnumAttrCase<"MMA", 1, "mma">,
  ]> {
  let cppNamespace = "::mlir::triton::proton";
  let description = [{
    Attribute to indicate the mode of profiling, which specifies passes and instructions to monitor.
  }];
}

#endif // PROTON_ATTR_DEFS
