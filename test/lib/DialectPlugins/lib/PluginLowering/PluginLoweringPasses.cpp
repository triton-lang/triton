//===- PluginLoweringPasses.cpp - PluginLowering passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PluginLowering/PluginLoweringPasses.h"

namespace mlir::pluginlowering {
#define GEN_PASS_DEF_PLUGINLOWERINGSWITCHBARFOO
#include "PluginLowering/PluginLoweringPasses.h.inc"

namespace {
class PluginLoweringSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class PluginLoweringSwitchBarFoo
    : public impl::PluginLoweringSwitchBarFooBase<PluginLoweringSwitchBarFoo> {
public:
  using impl::PluginLoweringSwitchBarFooBase<
      PluginLoweringSwitchBarFoo>::PluginLoweringSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PluginLoweringSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::pluginlowering
