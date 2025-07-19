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
