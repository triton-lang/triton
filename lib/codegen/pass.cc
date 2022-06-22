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
#include "triton/driver/llvm.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/print.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"

namespace triton {
namespace codegen {

static std::unique_ptr<llvm::Module> link_libdevice(
    ir::module& ir, llvm::LLVMContext& ctx,
    std::unique_ptr<llvm::Module> llvm) {
  std::string module_path = driver::path_to_libdevice();
  llvm::SMDiagnostic err;
  auto libdevice_mod = llvm::parseIRFile(module_path, err, ctx);
  if (!libdevice_mod) {
    throw std::runtime_error("Failed to load libdevice at " + module_path);
  }

  // Set triple and data layout of libdevice to match the target module
  if (llvm::Linker::linkModules(*llvm, std::move(libdevice_mod))) {
    throw std::runtime_error("Failed to link libdevice at " + module_path);
  }

  // Internalize all device functions
  std::set<llvm::StringRef> function_names;
  for (auto& func : ir.get_function_list()) {
    function_names.insert(func->get_name());
  }
  llvm::legacy::PassManager pass;
  pass.add(llvm::createInternalizePass([&](const llvm::GlobalValue& v) -> bool {
    if (function_names.count(v.getName()) != 0) {
      return true;
    }
    return false;
  }));

  // Add nvvm reflect flags to llvm module
  // (i32 4 indicates that the value set here overrides the value in another
  // module we link with. See the LangRef <LangRef.html#module-flags-metadata>
  // for details.)
  llvm::Type* I32 = llvm::Type::getInt32Ty(ctx);
  llvm::Metadata* md_four =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(I32, 4));
  llvm::Metadata* md_name = llvm::MDString::get(ctx, "nvvm-reflect-ftz");
  llvm::Metadata* md_one =
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(I32, 1));
  llvm::MDNode* reflect = llvm::MDNode::get(ctx, {md_four, md_name, md_one});
  llvm->addModuleFlag(reflect);

  // Cleanup unused functions caused by reflection
  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;
  builder.SizeLevel = 0;
  builder.populateModulePassManager(pass);

  pass.run(*llvm);
  return llvm;
}

// TODO:
// There should be a proper pass manager there!
std::unique_ptr<llvm::Module> add_passes_to_emit_bin(
    ir::module& ir, llvm::LLVMContext& ctx, codegen::target* target, int cc,
    int num_warps, int num_stages, int& shared_static) {
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
  inliner.run(ir);
  dce.run(ir);
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
  // ir.print(std::cout);
  isel.visit(ir, *llvm);
  shared_static = allocation.allocated_size();
  if (isel.has_libdevice_functions()) {
    llvm = link_libdevice(ir, ctx, std::move(llvm));
  }
  return llvm;
}


} // namespace codegen
} // namespace triton
