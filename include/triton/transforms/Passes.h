#ifndef TRITON_TRANSFORMS_PASSES_H_
#define TRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

std::unique_ptr<Pass> createCombineOpsPass();

// // Registration
// #define GEN_PASS_REGISTRATION
// #include 

#endif // TRITON_TRANSFORMS_PASSES_H_
