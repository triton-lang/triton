#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/transform/cts.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"
#include <iostream>

namespace triton {
namespace codegen{
namespace transform{


bool cts::is_shmem_op(ir::instruction* i, int op) {
  if(i->get_id() == ir::INST_DOT){
    // std::cout << i << " " << i->get_operand(0) << " " << layouts_->has(i->get_operand(0)) << " " << layouts_->has(i->get_operand(1)) << std::endl;
    // FP16 MMA layout can be kept in register for the LHS
    // Anything else has to be in shared memory
    // ir::value* lhs = i->get_operand(0);
    // if(op == 0){
      // i->print(std::cout);
    //   std::cout << layouts_->has(lhs) << std::endl;
    //   analysis::mma_layout* mma_lhs = layouts_->get(lhs)->to_mma();
    //   bool is_lhs_shmem = !(mma_lhs && lhs->get_type()->get_primitive_size_in_bits() == 16);
    //   // return is_lhs_shmem;
    // }
    return op == 0 || op == 1;
  }
  if(i->get_id() == ir::INST_COPY_FROM_SHARED)
    return op==0;
  if(i->get_id() == ir::INST_TRANS)
    return op==0;
  return false;
}

bool cts::is_shmem_res(ir::value* v){
  ir::instruction* i = dynamic_cast<ir::instruction*>(v);
  if(!i)
    return false;
  if(i->get_id() == ir::INST_TRANS)
    return true;
  if(i->get_id() == ir::INST_COPY_TO_SHARED)
    return true;
  if(i->get_id() == ir::INST_MASKED_LOAD_ASYNC)
    return true;
  return false;
}


// run pass on module
void cts::add_copy(ir::instruction *parent, ir::value *x, ir::builder &builder, bool to_shared, std::map<ir::value*, ir::value*>& copies) {
  auto *i = dynamic_cast<ir::instruction*>(x);
  // not an instruction
  if(!i) {
    builder.set_insert_point(parent);
    ir::value *copy;
    if(to_shared)
      copy = builder.create_copy_to_shared(x);
    else
      copy = builder.create_copy_from_shared(x);
    parent->replace_uses_of_with(x, copy);
    return;
  }
  // phi node
  if(auto* phi = dynamic_cast<ir::phi_node*>(x)) {
    for(unsigned i = 0; i < phi->get_num_incoming(); ++i)
      add_copy(phi, phi->get_incoming_value(i), builder, to_shared, copies);
    return;
  }
  // already in shared memory
  if(to_shared && is_shmem_res(i))
    return;
  // copy
  builder.set_insert_point_after(i);
  ir::value *copy;
  if(to_shared){
    copy = builder.create_copy_to_shared(x);
  }
  else
    copy = builder.create_copy_from_shared(x);
  copies.insert({x, copy});
  parent->replace_uses_of_with(x, copies.at(x));
}

void cts::run(ir::module &mod) {
  // Precompute where copies should be added
  std::set<ir::value*> shmem_ops;
  std::set<ir::value*> shmem_res;
  ir::for_each_instruction(mod, [&](ir::instruction* i) {
    if(i->get_id() == ir::INST_DOT){
      ir::value* lhs = i->get_operand(0);
      ir::type* ty = lhs->get_type()->get_scalar_ty();
      analysis::mma_layout* mma_lhs = layouts_->get(lhs)->to_mma();
      // TODO: V100
      bool is_lhs_shmem = !(mma_lhs && ty->get_primitive_size_in_bits() == 16);
      if(is_lhs_shmem)
        shmem_ops.insert(lhs);
      shmem_ops.insert(i->get_operand(1));
    }
      // std::cout << i << " " << i->get_operand(0) << " " << layouts_->has(i->get_operand(0)) << " " << layouts_->has(i->get_operand(1)) << std::endl;
      // FP16 MMA layout can be kept in register for the LHS
      // Anything else has to be in shared memory
      // if(op == 0){
        // i->print(std::cout);
      //   std::cout << layouts_->has(lhs) << std::endl;
      //   // return is_lhs_shmem;
      // }
    if(i->get_id() == ir::INST_COPY_FROM_SHARED)
      shmem_ops.insert(i->get_operand(0));
    if(i->get_id() == ir::INST_TRANS)
      shmem_ops.insert(i->get_operand(0));
    if(i->get_id() == ir::INST_TRANS ||
       i->get_id() == ir::INST_COPY_TO_SHARED ||
       i->get_id() == ir::INST_MASKED_LOAD_ASYNC)
      shmem_res.insert(i);
  });

  // Add shared copies
  std::map<ir::value*, ir::value*> copies;
  ir::builder &builder = mod.get_builder();
  ir::for_each_instruction(mod, [&](ir::instruction* i) {
    size_t num_op = i->get_num_operands();
    for(size_t k = 0; k < num_op; k++){
      ir::value* op = i->get_operand(k);
      // copy to shared operands
      bool is_shmem_op = shmem_ops.find(op) != shmem_ops.end();
      if(is_shmem_op)
        add_copy(i, op, builder, true, copies);
    }
  });
}


}
}
}