#ifndef TRITON_CONVERSION_PASSDETAIL_H
#define TRITON_CONVERSION_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir{
namespace triton{

#define GEN_PASS_CLASSES
#include "triton/Conversion/Passes.h.inc"

}
}

#endif
