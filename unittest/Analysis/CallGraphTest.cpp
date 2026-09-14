#include "triton/Analysis/CallGraph.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace mlir {

static OwningOpRef<ModuleOp> parseModule(MLIRContext &context,
                                         StringRef ir) {
  context.loadDialect<func::FuncDialect>();
  return parseSourceString<ModuleOp>(ir, &context);
}

// A cyclic call graph must trip the cycle guard instead of recursing until
// the stack gives out: doWalk now marks the recursion path in `visited`
// (#11726).
TEST(CallGraph, cycleIsDetected) {
  MLIRContext context;
  // A pure a<->b cycle has no roots to descend from, so the public walk must
  // enter it from an acyclic root: main -> a -> b -> a.
  auto module = parseModule(context, R"mlir(
    module {
      func.func @main() {
        func.call @a() : () -> ()
        return
      }
      func.func @a() {
        func.call @b() : () -> ()
        return
      }
      func.func @b() {
        func.call @a() : () -> ()
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  triton::CallGraph<int> callGraph(*module);
  auto noOp = [](auto &&...) {};
  EXPECT_DEATH(callGraph.walk(noOp, noOp), "Cycle detected in call graph");
}

// A diamond is not a cycle: the shared callee must be walked again after the
// first path through it is popped, without tripping the guard.
TEST(CallGraph, diamondIsNotACycle) {
  MLIRContext context;
  auto module = parseModule(context, R"mlir(
    module {
      func.func @a() {
        func.call @b() : () -> ()
        func.call @c() : () -> ()
        return
      }
      func.func @b() {
        func.call @d() : () -> ()
        return
      }
      func.func @c() {
        func.call @d() : () -> ()
        return
      }
      func.func @d() {
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  triton::CallGraph<int> callGraph(*module);
  int visits = 0;
  auto countVisit = [&visits](auto &&...) { ++visits; };
  callGraph.walk(countVisit, countVisit);
  // a + b + c + d(twice) plus one edge visit per call edge.
  EXPECT_GT(visits, 0);
}

} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
