#include "triton/codegen/pass.h"

#include "llvm/Pass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
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
#include "triton/codegen/transform/inline.h"
#include "triton/codegen/transform/membar.h"
#include "triton/codegen/transform/peephole.h"
#include "triton/codegen/transform/pipeline.h"
#include "triton/codegen/transform/prefetch.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/print.h"

namespace triton {
namespace codegen {

static void link_extern_libs(const ExternLibMap& user_extern_lib_map,
                             const ExternLibMap& target_extern_lib_map,
                             ir::module& ir, llvm::LLVMContext& ctx,
                             std::unique_ptr<llvm::Module>& llvm) {
  for (const auto& iter : target_extern_lib_map) {
    auto &lib_name = iter.first;
    if (user_extern_lib_map.count(lib_name) != 0 &&
        user_extern_lib_map.at(lib_name)->path() != "") {
      // If the user specified a path for this library, use it.
      user_extern_lib_map.at(lib_name)->install(ctx, llvm);
    } else {
      // Otherwise, use the default path.
      iter.second->install(ctx, llvm);
    }
  }

  std::set<llvm::StringRef> function_names;
  for (auto& func : ir.get_function_list()) {
    function_names.insert(func->get_name());
  }
  llvm::legacy::PassManager pass;
  pass.add(llvm::createInternalizePass([&](const llvm::GlobalValue& v) -> bool {
    if (function_names.count(v.getName()) != 0) {
      // Preserve global functions
      return true;
    }
    // Internalize all device functions
    return false;
  }));

  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(*llvm);

  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;
  builder.SizeLevel = 0;
  builder.populateModulePassManager(pass);

  pass.run(*llvm);
}

// TODO:
// There should be a proper pass manager there!
std::unique_ptr<llvm::Module> add_passes_to_emit_bin(
    ir::module& ir, llvm::LLVMContext& ctx, codegen::target* target,
    int num_warps, int num_stages, int& shared_static,
    const ExternLibMap& extern_lib_map) {
  // generate llvm code
  std::string name = ir.get_function_list()[0]->get_name();
  std::unique_ptr<llvm::Module> llvm(new llvm::Module(name, ctx));
  // optimizations
  bool has_sm80 = target->as_nvidia() && target->as_nvidia()->sm() >= 80;
  // create passes
  codegen::analysis::align align;
  codegen::transform::inliner inliner;
  codegen::analysis::axes axes;
  codegen::transform::pipeline pipeline(has_sm80, num_stages);
  codegen::transform::disassociate disassociate;
  codegen::analysis::layouts layouts(&axes, &align, num_warps, target);
  codegen::transform::cts cts(&layouts, has_sm80);
  codegen::analysis::liveness liveness(&layouts);
  codegen::analysis::swizzle swizzle(&layouts, target);
  codegen::analysis::allocation allocation(&liveness);
  codegen::transform::dce dce;
  codegen::transform::peephole peephole(target, &layouts);
  codegen::transform::coalesce coalesce(&align, &layouts, has_sm80);
  codegen::transform::prefetch prefetch_s(target);
  codegen::transform::membar barriers(&liveness, &layouts, &allocation,
                                      &prefetch_s, target);
  codegen::generator isel(&axes, &layouts, &align, &allocation, &swizzle,
                          target, num_warps);
  // run passes
  inliner.run(ir);
  dce.run(ir);
  peephole.run(ir);
  dce.run(ir);
  pipeline.run(ir);
  dce.run(ir);
  // ir.print(std::cout);
  disassociate.run(ir);
  dce.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  peephole.run(ir);
  dce.run(ir);
  if (target->is_gpu()) cts.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  coalesce.run(ir);
  dce.run(ir);
  align.run(ir);
  dce.run(ir);
  if (target->is_gpu()) cts.run(ir);
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
  // std::cout << "---" << std::endl;
  // ir.print(std::cout);
  // std::cout << "---" << std::endl;
  // ir.print(std::cout);
  liveness.run(ir);
  allocation.run(ir);
  prefetch_s.run(ir);
  barriers.run(ir);
  // exit(1);
  // ir.print(std::cout);
  isel.visit(ir, *llvm);
  shared_static = allocation.allocated_size();
  if (target->as_nvidia() && target->as_nvidia()->sm() < 70) {
    // sm < 70 (Pascal) has little shared memory resource.
    // Instead of having "Error: Invalid argument" on launching a kernel, let's throw an error here.
    if (shared_static >= 65536) {
      throw std::runtime_error("Device does not support shared memory of " + std::to_string(shared_static) + "bytes");
    }
  }

  if (isel.get_extern_lib_map().size() > 0) {
    // If there's any extern lib calls,
    // we need to link them in.
    link_extern_libs(extern_lib_map, isel.get_extern_lib_map(), ir, ctx, llvm);
  }

  return llvm;
}

}  // namespace codegen
}  // namespace triton
