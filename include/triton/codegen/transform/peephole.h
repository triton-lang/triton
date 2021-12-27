#ifndef TDL_INCLUDE_CODEGEN_OPTIMIZE_TRANS_H
#define TDL_INCLUDE_CODEGEN_OPTIMIZE_TRANS_H

#include "triton/codegen/target.h"

namespace triton {

namespace ir {
  class module;
  class value;
  class instruction;
  class trans_inst;
  class builder;
  class constant_int;
  class dot_inst;
}

namespace codegen{
namespace analysis{
class layouts;
}

namespace transform{

class peephole {
private:
//  bool rewrite_cts_cfs(ir::instruction *value, ir::builder &builder);
  bool rewrite_trans_phi(ir::instruction* value, ir::builder &builder);
  bool rewrite_dot_fp32(ir::dot_inst *dot, ir::builder& builder, bool trans_a, bool trans_b, ir::value *A, ir::value *B, ir::value *D);
  bool rewrite_dot_hmma(ir::dot_inst *dot, ir::builder& builder, bool trans_a, bool trans_b, ir::value *A, ir::value *B, ir::value *D);
  bool rewrite_dot(ir::instruction *value, ir::builder& builder);
  bool rewrite_mult(ir::instruction *value, ir::builder& builder);
  bool rewrite_insert_extract(ir::instruction *value, ir::builder& builder);


  bool rewrite_unit_red(ir::instruction *value, ir::builder& builder);
  bool rewrite_gep_ptr_min_off_plus_off(ir::instruction *value, ir::builder& builder);
  bool rewrite_select_masked_load(ir::instruction *value, ir::builder& builder);
  bool rewrite_load_to_shared(ir::instruction *value, ir::builder& builder);
  bool rewrite_cvt_layout(ir::instruction *value, ir::builder& builder);
 
public:
  peephole(target* tgt, analysis::layouts* layouts): tgt_(tgt), layouts_(layouts) {}
  void run(ir::module &mod);

private:
  target* tgt_;
  analysis::layouts* layouts_;
};


}
}
}

#endif
