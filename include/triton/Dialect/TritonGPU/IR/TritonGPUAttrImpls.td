//===----------------------------------------------------------------------===//
// Aggregated attr definitions (including CGA) for implementation emission.
// This file exists to generate AttrDefs.cpp.inc once, without duplicating
// CGAEncodingAttr while still making CGA available before LayoutEncodingTrait.
//===----------------------------------------------------------------------===//

#ifndef TRITONGPU_ATTRIMPLS_TD
#define TRITONGPU_ATTRIMPLS_TD

include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

#endif // TRITONGPU_ATTRIMPLS_TD
