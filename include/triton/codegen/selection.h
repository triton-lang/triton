#ifndef TDL_INCLUDE_CODEGEN_SELECTION_H
#define TDL_INCLUDE_CODEGEN_SELECTION_H

#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "triton/ir/visitor.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/transform/cts.h"


namespace llvm{
  class Type;
  class Value;
  class Instruction;
  class Constant;
  class LLVMContext;
  class Module;
  class ConstantFolder;
  class IRBuilderDefaultInserter;
  template <typename T, typename Inserter>
  class IRBuilder;
  class ArrayType;
  class Function;
}


// typedefs
namespace triton{
namespace codegen{
  typedef llvm::IRBuilder<llvm::ConstantFolder,
                          llvm::IRBuilderDefaultInserter> Builder;
  typedef llvm::LLVMContext LLVMContext;
  typedef llvm::Type Type;
  typedef llvm::Value Value;
  typedef llvm::Module Module;
  typedef llvm::Instruction Instruction;
  typedef llvm::Constant Constant;
  typedef llvm::ArrayType ArrayType;
  typedef llvm::Function Function;
}
}

namespace triton{
namespace codegen{

namespace analysis{
class liveness;
class tiles;
class align;
class allocation;
class cts;
class axes;
class layout;
}

namespace transform{
class coalesce;
}

class target;

typedef std::vector<Value*> indices_t;

struct distributed_axis {
  int contiguous;
  std::vector<Value*> values;
  Value* thread_id;
};

class tile {
protected:
  typedef std::vector<unsigned> shapes_t;

public:
  tile(Type *ty, const shapes_t &shapes): ty_(ty), shapes_(shapes){ }
  virtual void set_value(indices_t idx, Value *v) = 0;
  virtual Value* get_value(indices_t idx) = 0;
  Type *get_ty() const { return ty_; }
  shapes_t get_shapes() const { return shapes_; }

protected:
  Type *ty_;
  shapes_t shapes_;
};

class shared_tile: public tile {
private:
  void extract_constant(Value *arg, Value *&non_cst, Value *&cst);
  void extract_constant(const indices_t &arg_idx, indices_t &non_cst_idx, indices_t &cst_idx);


public:
  shared_tile(Type* ty, const shapes_t &shapes, const std::vector<int> &order, Value* ptr, Builder &builder, Value* offset = nullptr, const std::vector<int>& perm = {});
  void set_vector_size(unsigned vector_size);
  void set_return_mode(bool return_vector);
  void set_value(indices_t, Value *);
  Value* get_ptr_to(indices_t idx);
  Value* get_value(indices_t idx);
  Value* get_pointer() { return ptr_; }
  Value* get_offset() { return offset_; }
  const std::vector<int>& get_perm() { return perm_; }
  const std::vector<int>& get_order() { return order_; }
  static Value* shared_offset(Builder& builder, const shapes_t& shapes, const std::vector<int>& perm, const std::vector<int>& order, indices_t idx);

private:
  Value *ptr_;
  bool return_vector_;
  Builder &builder_;
  Value *offset_;
  std::map<indices_t, Value*> ptr_cache_;
  unsigned vector_size_;
  std::vector<int> order_;
  std::vector<int> perm_;
};

// Distribtued tile
class distributed_tile: public tile{
  typedef std::vector<distributed_axis> axes_t;
  typedef std::vector<indices_t> ordered_indices_vec_t;
  typedef std::map<indices_t, unsigned> indices_map_t;
  typedef std::map<indices_t, Value*> values_map_t;

private:
  void init_indices();
  Type *make_vector_ty(Type *ty, size_t vector_size);

public:
  distributed_tile(Type *ty, const shapes_t& shapes, const std::vector<int>& order, const axes_t &axes, Builder &builder, bool vectorize);
  void set_value(indices_t idx, Value *v);
  Value* get_value(indices_t idx);
  const std::vector<int>& get_order() { return order_; }
  unsigned get_linear_index(indices_t idx);
  indices_t get_ordered_indices(unsigned id);
  void for_each(std::function<void(indices_t)> fn);
  const distributed_axis &axis(unsigned dim) { return axes_.at(dim); }

private:
  axes_t axes_;
  std::vector<int> order_;
  indices_map_t indices_;
  values_map_t values_;
  ordered_indices_vec_t ordered_indices_;
  size_t vector_size_;
  Builder &builder_;
};

class machine_layout_t {
public:
  virtual tile* create(ir::value *v) = 0;
};

class machine_layout_shared_t: public machine_layout_t {
public:
  machine_layout_shared_t(Module *mod, Builder *builder, target *tgt, analysis::allocation* alloc, Value *&sh_mem_ptr, analysis::layout_t* layout,
                          std::map<ir::value *, Value *>& vmap,
                          std::map<ir::value *, tile *>& tmap);

  tile* create(ir::value *v);

  Module *mod_;
  Builder *builder_;
  target *tgt_;
  analysis::allocation* alloc_;
  Value *&sh_mem_ptr_;
  analysis::layout_t* layout_;
  std::map<ir::value *, Value *>& vmap_;
  std::map<ir::value *, tile *>& tmap_;

  Value *offset_;
  Value *ptr_;
  Value *pre_ptr_;
  Value *next_ptr_;

};

class machine_layout_distributed_t: public machine_layout_t {
public:
  machine_layout_distributed_t(Module *mod, Builder *builder, target *tgt, Type *ty,
                               analysis::axes *a_axes, std::map<unsigned, distributed_axis>& axes,
                               analysis::layout_t* layout);

  tile* create(ir::value *v);
  Module *mod_;
  Builder *builder_;
  target *tgt_;
  Type *ty_;
  analysis::axes *a_axes_;
  std::map<unsigned, distributed_axis>& axes_;
  analysis::layout_t* layout_;
};


class machine_layout_hmma_884_t: public machine_layout_distributed_t {
public:
  machine_layout_hmma_884_t(Module *mod, Builder *builder,
                            target *tgt, Type *ty,
                            analysis::axes *a_axes, std::map<unsigned, distributed_axis>& axes,
                            analysis::layout_hmma_884_t* layout);
  Value *offset_a_i_, *offset_a_k_;
  Value *offset_b_j_, *offset_b_k_;
  unsigned pack_size_0_;
  unsigned pack_size_1_;
  unsigned num_packs_0_;
  unsigned num_packs_1_;
};

class machine_layout_scanline_t: public machine_layout_distributed_t {
public:
  machine_layout_scanline_t(Module *mod, Builder *builder,
                            target *tgt, Type *ty,
                            analysis::axes *a_axes, std::map<unsigned, distributed_axis>& axes,
                            analysis::layout_scanline_t* layout);
};

class generator: public ir::visitor, public analysis::layout_visitor {
private:
  void for_each(ir::value *x, const std::function<void(indices_t)>& fn);
  Value* get_value(ir::value *x, const indices_t& idx);
  void set_value(ir::value *x, const indices_t& idx, Value* v);

  void visit_hmma_dot(ir::dot_inst*, shared_tile *TA, shared_tile *TB, distributed_tile *TD, unsigned NK);
  void visit_scanline_dot(ir::dot_inst*, shared_tile *TA, shared_tile *TB, distributed_tile *TD, unsigned NK, Type *c_ty, Function *f_mul_add);
  void visit_outer_dot(ir::dot_inst*, distributed_tile *TA, distributed_tile *TB, distributed_tile *TD, unsigned NK,
                       Type *c_ty, Function *f_mul_add);

  void finalize_shared_layout(analysis::layout_shared_t*);
  void finalize_function(ir::function*);
  void finalize_phi_node(ir::phi_node*);

public:
  generator(Module *dst,
            analysis::axes *a_axes,
            target *tgt,
            analysis::layout *layouts,
            analysis::align *alignment,
            analysis::allocation *alloc,
            unsigned num_warps);

  void visit_value(ir::value* v);

  void visit_phi_node(ir::phi_node*);
  void visit_binary_operator(ir::binary_operator*);
  void visit_getelementptr_inst(ir::getelementptr_inst*);

  void visit_icmp_inst(ir::icmp_inst*);
  void visit_fcmp_inst(ir::fcmp_inst*);
  void visit_cast_inst(ir::cast_inst*);

  void visit_return_inst(ir::return_inst*);
  void visit_cond_branch_inst(ir::cond_branch_inst*);
  void visit_uncond_branch_inst(ir::uncond_branch_inst*);


  void visit_unmasked_load_inst(ir::unmasked_load_inst*);
  void visit_masked_load_inst(ir::masked_load_inst*);
  void visit_unmasked_store_inst(ir::unmasked_store_inst*);
  void visit_masked_store_inst(ir::masked_store_inst*);

  void visit_reshape_inst(ir::reshape_inst*);
  void visit_splat_inst(ir::splat_inst*);
  void visit_broadcast_inst(ir::broadcast_inst*);
  void visit_downcast_inst(ir::downcast_inst*);

  void visit_get_program_id_inst(ir::get_program_id_inst*);
  void visit_get_num_program_inst(ir::get_num_program_inst*);
  void visit_atomic_cas_inst(ir::atomic_cas_inst*);
  void visit_atomic_exch_inst(ir::atomic_exch_inst*);
  void visit_atomic_add_inst(ir::atomic_add_inst*);
  void visit_dot_inst(ir::dot_inst*);
  void visit_trans_inst(ir::trans_inst*);
  void visit_sqrt_inst(ir::sqrt_inst*);
  void visit_reduce_inst(ir::reduce_inst*);
  void visit_select_inst(ir::select_inst*);

  void visit_copy_to_shared_inst(ir::copy_to_shared_inst*);
  void visit_copy_from_shared_inst(ir::copy_from_shared_inst*);
  void visit_barrier_inst(ir::barrier_inst*);
  void visit_make_range_dyn(ir::make_range_dyn*);
  void visit_make_range(ir::make_range*);

  void visit_make_range_sta(ir::make_range_sta*);
  void visit_undef_value(ir::undef_value*);
  void visit_constant_int(ir::constant_int*);
  void visit_constant_fp(ir::constant_fp*);
  void visit_alloc_const(ir::alloc_const*);

  void visit_function(ir::function*);
  void visit_basic_block(ir::basic_block*);
  void visit_argument(ir::argument*);

  void visit_layout_hmma_884(analysis::layout_hmma_884_t*);
  void visit_layout_scanline(analysis::layout_scanline_t*);
  void visit_layout_shared(analysis::layout_shared_t*);

private:
  LLVMContext *ctx_;
  std::unique_ptr<Builder> builder_;
  Module *mod_;

  std::map<const analysis::layout_t*, machine_layout_t*> machine_layouts_;
  analysis::axes *a_axes_;
  std::map<unsigned, distributed_axis> axes_;
  std::map<ir::value *, Value *> vmap_;
  std::map<ir::value *, tile *> tmap_;
  target *tgt_;
  analysis::layout *layouts_;
  analysis::align *alignment_;
  analysis::allocation *alloc_;
  Value *sh_mem_ptr_;
  unsigned num_warps_;

  std::set<ir::value*> seen_;
};


// Selection pass
class selection{
  typedef std::map<ir::value *, Value *> vmap_t;
  typedef std::map<ir::value *, tile *> tmap_t;

public:
  selection(analysis::liveness* liveness, analysis::allocation *alloc,
            analysis::align *alignment, analysis::axes *axes,
            analysis::layout *layouts, target *tgt, unsigned num_warps)
    : liveness_(liveness), alloc_(alloc),
      alignment_(alignment), a_axes_(axes), layouts_(layouts),
      tgt_(tgt), num_warps_(num_warps){ }

  void run(ir::module &src, Module &dst);

private:
  analysis::liveness *liveness_;
  analysis::allocation *alloc_;
  analysis::axes *a_axes_;
  analysis::layout *layouts_;
  analysis::align *alignment_;
  target *tgt_;
  unsigned num_warps_;
};

}
}

#endif
