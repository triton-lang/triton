#pragma once
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/Gluon/IR/Dialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/Gluon/IR/GluonAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/Gluon/IR/Ops.h.inc"
