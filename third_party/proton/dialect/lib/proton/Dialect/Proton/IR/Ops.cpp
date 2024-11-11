/*
 * Copyright (c) 2024 Meta Platforms Corporation & Affiliates. All rights
 * reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "proton/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define GET_OP_CLASSES
#include "proton/Dialect/Proton/IR/Ops.cpp.inc"
#include "proton/Dialect/Proton/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace proton {

// -- RecordOp --
void RecordOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult RecordOp::verify() { return success(); }

ParseResult RecordOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  int idx = 0;
  std::string tag, metric, granularity;
  mlir::IntegerAttr rid;

  auto parseAttr = [&]() {
    switch (idx++) {
    case 0:
      return parser.parseAttribute(rid, mlir::IntegerType::get(ctx, 32));
    case 1:
      return parser.parseString(&tag);
    case 2:
      return parser.parseOptionalString(&metric);
    case 3:
      return parser.parseOptionalString(&granularity);
    }
    return ParseResult(failure());
  };

  if (parser.parseLess() || parser.parseCommaSeparatedList(parseAttr) ||
      parser.parseGreater())
    return failure();

  bool isStart;
  if (tag == "start")
    isStart = true;
  else if (tag == "end")
    isStart = false;
  else
    return failure();
  result.addAttribute("isStart", BoolAttr::get(ctx, isStart));

  result.addAttribute("regionId", rid);

  MetricAttr metricAttr;
  if (!metric.empty()) {
    if (metric == "cycle")
      metricAttr = MetricAttr::get(ctx, Metric::CYCLE);
    else if (metric == "invalid")
      metricAttr = MetricAttr::get(ctx, Metric::INVALID);
    else
      return failure();
    result.addAttribute("metric", metricAttr);
  }

  GranularityAttr granularityAttr;
  if (!granularity.empty()) {
    if (granularity == "warpgroup")
      granularityAttr =
          GranularityAttr::get(ctx, Granularity::WARPGROUP);
    else if (granularity == "warp")
      granularityAttr =
          GranularityAttr::get(ctx, Granularity::WARP);
    else
      return failure();
    result.addAttribute("granularity", granularityAttr);
  }

  return success();
}

void RecordOp::print(OpAsmPrinter &printer) {
  Operation *op = getOperation();
  printer << " <";
  printer << getRegionId() << ", ";

  if (getIsStart()) {
    printer << "\"start\", ";
  } else {
    printer << "\"end\", ";
  }

  if (getMetric() == Metric::CYCLE) {
    printer << "\"cycle\", ";
  } else {
    printer << "\"invalid\", ";
  }

  if (getGranularity() == Granularity::WARP) {
    printer << "\"warp\"";
  } else {
    printer << "\"warpgroup\"";
  }

  printer << ">";
}

} // namespace proton
} // namespace triton
} // namespace mlir
