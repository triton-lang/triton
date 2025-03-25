//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITONXPU_TYPES
#define TRITONXPU_TYPES

include "triton/Dialect/Triton/IR/TritonTypes.td" // For TT_FloatLike


def TTX_Vector : FixedVectorOf<[TT_FloatLike, TT_IntLike]>;
def TTX_TensorVector: AnyTypeOf<[TT_Tensor, TensorOf<[TTX_Vector]>]>;
def TTX_VectorLike: AnyTypeOf<[TTX_Vector, TensorOf<[TTX_Vector]>]>;
def TTX_Type: AnyTypeOf<[TT_Type, TTX_VectorLike]>;

#endif // TRITONXPU_TYPES
