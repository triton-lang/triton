//===----------------------------------------------------------------------===//
// Aggregated attr definitions (including CTA) for implementation emission.
// This file exists to generate AttrDefs.cpp.inc once, without duplicating
// CTAEncodingAttr while still making CTA available before LayoutEncodingTrait.
//===----------------------------------------------------------------------===//

#ifndef TRITONGPU_ATTRIMPLS_TD
#define TRITONGPU_ATTRIMPLS_TD

include "triton/Dialect/TritonGPU/IR/CTAEncodingAttr.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

#endif // TRITONGPU_ATTRIMPLS_TD
