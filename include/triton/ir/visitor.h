#pragma once

#ifndef _TRITON_IR_VISITOR_H_
#define _TRITON_IR_VISITOR_H_


namespace triton{
namespace ir{

class value;

class instruction;

class call_inst;

class phi_node;
class binary_operator;
class getelementptr_inst;

class icmp_inst;
class fcmp_inst;
class cast_inst;
class trunc_inst;
class z_ext_inst;
class s_ext_inst;
class fp_trunc_inst;
class fp_ext_inst;
class ui_to_fp_inst;
class si_to_fp_inst;
class fp_to_ui_inst;
class fp_to_si_inst;
class ptr_to_int_inst;
class int_to_ptr_inst;
class bit_cast_inst;
class addr_space_cast_inst;

class return_inst;
class cond_branch_inst;
class uncond_branch_inst;


class unmasked_load_inst;
class masked_load_inst;
class unmasked_store_inst;
class masked_store_inst;

class extract_value_inst;
class insert_value_inst;

class retile_inst;
class reshape_inst;
class splat_inst;
class cat_inst;
class broadcast_inst;
class downcast_inst;

class umulhi_inst;
class exp_inst;
class cos_inst;
class sin_inst;
class log_inst;

class get_program_id_inst;
class get_num_programs_inst;
class atomic_inst;
class atomic_cas_inst;
class atomic_rmw_inst;
class dot_inst;
class trans_inst;
class sqrt_inst;
class reduce_inst;
class select_inst;

class cvt_layout_inst;
class copy_to_shared_inst;
class copy_from_shared_inst;
class masked_load_async_inst;
class barrier_inst;
class async_wait_inst;
class make_range_dyn;
class make_range;
class prefetch_s_inst;

class make_range_sta;
class undef_value;
class constant_int;
class constant_fp;
class global_value;
class global_object;
class alloc_const;

class constant_fp;
class undef_value;
class constant_int;
class constant_fp;
class global_value;
class global_object;
class alloc_const;

class function;

class basic_block;

class argument;

class visitor {
public:
  virtual ~visitor() {}

  virtual void visit_value(ir::value*);
  virtual void visit_call_inst(ir::call_inst*) = 0;

  virtual void visit_basic_block(basic_block*) = 0;
  virtual void visit_argument(argument*) = 0;
  virtual void visit_phi_node(phi_node*) = 0;
  virtual void visit_binary_operator(binary_operator*) = 0;
  virtual void visit_getelementptr_inst(getelementptr_inst*) = 0;

  virtual void visit_icmp_inst(icmp_inst*) = 0;
  virtual void visit_fcmp_inst(fcmp_inst*) = 0;
  virtual void visit_cast_inst(cast_inst*) = 0;

  virtual void visit_return_inst(return_inst*) = 0;
  virtual void visit_cond_branch_inst(cond_branch_inst*) = 0;
  virtual void visit_uncond_branch_inst(uncond_branch_inst*) = 0;


  virtual void visit_unmasked_load_inst(unmasked_load_inst*) = 0;
  virtual void visit_masked_load_inst(masked_load_inst*) = 0;
  virtual void visit_unmasked_store_inst(unmasked_store_inst*) = 0;
  virtual void visit_masked_store_inst(masked_store_inst*) = 0;

  virtual void visit_umulhi_inst(umulhi_inst*) = 0;
  virtual void visit_exp_inst(exp_inst*) = 0;
  virtual void visit_cos_inst(cos_inst*) = 0;
  virtual void visit_sin_inst(sin_inst*) = 0;
  virtual void visit_log_inst(log_inst*) = 0;

  virtual void visit_extract_value_inst(extract_value_inst*) = 0;
  virtual void visit_insert_value_inst(insert_value_inst*) = 0;

  virtual void visit_reshape_inst(reshape_inst*) = 0;
  virtual void visit_splat_inst(splat_inst*) = 0;
  virtual void visit_cat_inst(cat_inst*) = 0;
  virtual void visit_broadcast_inst(broadcast_inst*) = 0;
  virtual void visit_downcast_inst(downcast_inst*) = 0;

  virtual void visit_get_program_id_inst(get_program_id_inst*) = 0;
  virtual void visit_get_num_programs_inst(get_num_programs_inst*) = 0;
  virtual void visit_atomic_cas_inst(atomic_cas_inst*) = 0;
  virtual void visit_atomic_rmw_inst(atomic_rmw_inst*) = 0;
  virtual void visit_dot_inst(dot_inst*) = 0;
  virtual void visit_trans_inst(trans_inst*) = 0;
  virtual void visit_sqrt_inst(sqrt_inst*) = 0;
  virtual void visit_reduce_inst(reduce_inst*) = 0;
  virtual void visit_select_inst(select_inst*) = 0;

  virtual void visit_cvt_layout_inst(cvt_layout_inst*) = 0;
  virtual void visit_copy_to_shared_inst(copy_to_shared_inst*) = 0;
  virtual void visit_copy_from_shared_inst(copy_from_shared_inst*) = 0;


  virtual void visit_masked_load_async_inst(masked_load_async_inst*)= 0;
  virtual void visit_barrier_inst(barrier_inst*) = 0;
  virtual void visit_async_wait_inst(async_wait_inst*) = 0;
  virtual void visit_make_range(make_range*) = 0;
  virtual void visit_prefetch_s_inst(prefetch_s_inst*) = 0;
  virtual void visit_function(function*) = 0;

  virtual void visit_undef_value(undef_value*) = 0;
  virtual void visit_constant_int(constant_int*) = 0;
  virtual void visit_constant_fp(constant_fp*) = 0;
  virtual void visit_alloc_const(alloc_const*) = 0;
};

}
}

#endif
