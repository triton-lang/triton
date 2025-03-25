//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITONXPU_INTERFACES
#define TRITONXPU_INTERFACES


include "mlir/Interfaces/SideEffectInterfaces.td" // MemoryEffects
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike

def SameOperandsEncoding : NativeOpTrait<"SameOperandsEncoding">;

#endif // TRITONXPU_INTERFACES
