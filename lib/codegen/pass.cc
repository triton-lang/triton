#include "triton/codegen/pass.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/selection/generator.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/transform/cts.h"
#include "triton/codegen/transform/dce.h"
#include "triton/codegen/transform/disassociate.h"
#include "triton/codegen/transform/membar.h"
#include "triton/codegen/transform/peephole.h"
#include "triton/codegen/transform/pipeline.h"
#include "triton/codegen/transform/prefetch.h"
#include "triton/codegen/transform/inline.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/print.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
namespace triton {
namespace codegen {

// TODO:
// There should be a proper pass manager there!
std::unique_ptr<llvm::Module> add_passes_to_emit_bin(ir::module &ir, llvm::LLVMContext& ctx, codegen::target* target,
                                                     int cc, int num_warps, int num_stages, int& shared_static) {
  // generate llvm code
  std::string name = ir.get_function_list()[0]->get_name();
  std::unique_ptr<llvm::Module> llvm(new llvm::Module(name, ctx));
  // optimizations
  bool cts_use_async = target->as_nvidia() && target->as_nvidia()->sm() >= 80;
  // create passes
  codegen::analysis::align align;
  codegen::transform::inliner inliner;
  codegen::analysis::axes axes;
  codegen::transform::cts cts(cts_use_async);
  codegen::transform::pipeline pipeline(cts_use_async, num_stages);
  codegen::transform::disassociate disassociate;
  codegen::analysis::layouts layouts(&axes, &align, num_warps, target);
  codegen::analysis::liveness liveness(&layouts);
  codegen::analysis::swizzle swizzle(&layouts, target);
  codegen::analysis::allocation allocation(&liveness);
  codegen::transform::dce dce;
  codegen::transform::peephole peephole(target, &layouts);
  codegen::transform::coalesce coalesce(&align, &layouts);
  codegen::transform::prefetch prefetch_s(target);
  codegen::transform::membar barriers(&liveness, &layouts, &allocation, &prefetch_s, target);
  codegen::generator isel(&axes, &layouts, &align, &allocation, &swizzle, target, num_warps);
  // run passes
  ir.print(std::cout);
  inliner.run(ir);
  dce.run(ir);
  ir.print(std::cout);
  exit(1);
  peephole.run(ir);
  dce.run(ir);
  pipeline.run(ir);
  dce.run(ir);  
  disassociate.run(ir);
  dce.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  peephole.run(ir);
  dce.run(ir);
  if (target->is_gpu())
    cts.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  coalesce.run(ir);
  dce.run(ir);
  align.run(ir);
  dce.run(ir);
  if (target->is_gpu())
    cts.run(ir);
  dce.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  peephole.run(ir);
  dce.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  swizzle.run(ir);
  liveness.run(ir);
  allocation.run(ir);
  prefetch_s.run(ir);
  barriers.run(ir);
  isel.visit(ir, *llvm);
  shared_static = allocation.allocated_size();
  return llvm;
}

} // namespace codegen
} // namespace triton
