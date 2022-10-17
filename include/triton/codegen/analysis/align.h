#ifndef TDL_INCLUDE_CODEGEN_ALIGNMENT_INFO_PASS_H
#define TDL_INCLUDE_CODEGEN_ALIGNMENT_INFO_PASS_H

#include <map>
#include <vector>

namespace triton {

namespace ir {
  class value;
  class module;
  class phi_node;
  class splat_inst;
  class cast_inst;
  class cmp_inst;
  class reshape_inst;
  class dequantize_inst;
  class broadcast_inst;
  class binary_operator;
  class getelementptr_inst;
}

namespace codegen{
namespace analysis{

class align {
private:
  struct cst_info {
    unsigned num_cst;
    unsigned value;
  };
  // helpers
  std::vector<unsigned> get_shapes(ir::value *v);
  // populate is_constant
  std::vector<cst_info> populate_is_constant_phi(ir::phi_node* x);
  std::vector<cst_info> populate_is_constant_splat(ir::splat_inst* x);
  std::vector<cst_info> populate_is_constant_reshape(ir::reshape_inst* x);
  std::vector<cst_info> populate_is_constant_dequantize(ir::dequantize_inst* x);
  std::vector<cst_info> populate_is_constant_broadcast(ir::broadcast_inst* x);
  std::vector<cst_info> populate_is_constant_binop(ir::binary_operator* x);
  std::vector<cst_info> populate_is_constant_cmp(ir::cmp_inst* x);
  std::vector<cst_info> populate_is_constant_gep(ir::getelementptr_inst* x);
  std::vector<cst_info> populate_is_constant_default(ir::value* v);
  std::vector<cst_info> populate_is_constant(ir::value *v);
  // populate max_contiguous
  std::vector<unsigned> populate_max_contiguous_phi(ir::phi_node* x);
  std::vector<unsigned> populate_max_contiguous_splat(ir::splat_inst* x);
  std::vector<unsigned> populate_max_contiguous_reshape(ir::reshape_inst* x);
  std::vector<unsigned> populate_max_contiguous_dequantize(ir::dequantize_inst* x);
  std::vector<unsigned> populate_max_contiguous_broadcast(ir::broadcast_inst* x);
  std::vector<unsigned> populate_max_contiguous_binop(ir::binary_operator* x);
  std::vector<unsigned> populate_max_contiguous_gep(ir::getelementptr_inst* x);
  std::vector<unsigned> populate_max_contiguous_cast(ir::cast_inst* x);
  std::vector<unsigned> populate_max_contiguous_default(ir::value* v);
  std::vector<unsigned> populate_max_contiguous(ir::value *v);
  // populate starting_multiple
  std::vector<unsigned> populate_starting_multiple_phi(ir::phi_node* x);
  std::vector<unsigned> populate_starting_multiple_splat(ir::splat_inst* x);
  std::vector<unsigned> populate_starting_multiple_reshape(ir::reshape_inst* x);
  std::vector<unsigned> populate_starting_multiple_dequantize(ir::dequantize_inst* x);
  std::vector<unsigned> populate_starting_multiple_broadcast(ir::broadcast_inst* x);
  std::vector<unsigned> populate_starting_multiple_binop(ir::binary_operator* x);
  std::vector<unsigned> populate_starting_multiple_gep(ir::getelementptr_inst* x);
  std::vector<unsigned> populate_starting_multiple_cast(ir::cast_inst* x);
  std::vector<unsigned> populate_starting_multiple_default(ir::value* v);
  std::vector<unsigned> populate_starting_multiple(ir::value *v);
  // populate all maps
  void populate(ir::value *v);

public:
  void run(ir::module &mod);
  unsigned get(ir::value* v, unsigned ax) const;
  std::vector<unsigned> contiguous(ir::value* v) const;
  std::vector<cst_info> get_cst_info(ir::value* v) const;

private:
  std::map<ir::value*, std::vector<cst_info>> is_constant_;
  std::map<ir::value*, std::vector<unsigned>> max_contiguous_;
  std::map<ir::value*, std::vector<unsigned>> starting_multiple_;
};


}
}
}

#endif
