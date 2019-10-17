#include <numeric>
#include "triton/codegen/selection/selection.h"
#include "triton/codegen/selection/generator.h"
#include "triton/ir/module.h"

namespace triton{
namespace codegen{

using namespace llvm;

void selection::run(ir::module &src, Module &dst) {
  generator gen(&dst, a_axes_, tgt_, layouts_, alignment_, alloc_, num_warps_ );
  for(ir::alloc_const *x: src.allocs())
    gen.visit_alloc_const(x);
  for(ir::function *fn: src.get_function_list())
    gen.visit_function(fn);
}

}
}
