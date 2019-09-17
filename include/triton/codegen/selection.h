#ifndef TDL_INCLUDE_CODEGEN_SELECTION_H
#define TDL_INCLUDE_CODEGEN_SELECTION_H

#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "triton/codegen/analysis/meminfo.h"


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
class tiles;
class align;
class memalloc;
class meminfo;
class axes;
class layout;
}

namespace transform{
class coalesce;
}

class target;

typedef std::vector<Value*> indices_t;

struct distributed_axis {
  size_t contiguous;
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
  shared_tile(Type* ty, const shapes_t &shapes, Value* ptr, Builder &builder, Value* offset = nullptr);
  void set_vector_size(unsigned vector_size);
  void set_return_mode(bool return_vector);
  void set_value(indices_t, Value *);
  Value* get_ptr_to(indices_t idx);
  Value* get_value(indices_t idx);
  Value* get_pointer() { return ptr_; }
  Value* get_offset() { return offset_; }
  static Value* shared_offset(Builder& builder, const shapes_t& shapes, indices_t idx);

private:
  Value *ptr_;
  bool return_vector_;
  Builder &builder_;
  Value *offset_;
  std::map<indices_t, Value*> ptr_cache_;
  unsigned vector_size_;
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


// Selection pass
class selection{
  typedef std::map<ir::value *, Value *> vmap_t;
  typedef std::map<ir::value *, tile *> tmap_t;

private:
  // utils
  Type *make_vector_ty(Type *ty, size_t vector_size);
  std::vector<unsigned> extract_shapes(ir::value *v);

  // LLVM conversions
  Type*        llvm_type(ir::type *ty, LLVMContext &ctx);
  Value*       llvm_value(ir::value *v, Builder &builder);
  Instruction* llvm_inst(ir::instruction *inst, std::function<Value*(ir::value*)> value, Builder &builder);
  Constant*    llvm_constant(ir::constant *cst, LLVMContext &ctx);
  Value*       llvm_alloc_const(ir::alloc_const *v, Module *module, Builder &builder);
  ArrayType*   llvm_linearized_tile_type(ir::type *ty, LLVMContext &ctx);

  // grid construction
  void create_grids(std::vector<ir::value *> &grids,
                    std::map<unsigned, ir::value *> &references,
                    ir::function *fn);
  void create_shared_tile(ir::value *v, Builder &builder, Value *sh_mem_ptr);
  void create_distributed_tile(ir::value *v, Builder &builder);
  void create_tile(ir::value *v, Builder &builder, std::set<ir::value *> &seen, Value *sh_mem_ptr);
  void init_strided_scan_axes(ir::value *i, Builder &builder, Value *u_thread_id, Value *u_warp_id);
  void init_hmma_axes(ir::value *i, Builder &builder, Value *u_thread_id, Value *u_warp_id);
  void init_axes(ir::value *i, Builder &builder, Value *u_thread_id, Value *u_warp_id);
  void init_grids(ir::function *fn, Builder &builder, Value *sh_mem_ptr);

  // lower scalar instruction
  void lower_instruction(ir::instruction *src, Builder &builder);
  // lower tile instruction
  void lower_masked_store(ir::masked_store_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_store(ir::store_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_downcast(ir::downcast_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_reduce(ir::reduce_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_dynamic_program_idx(ir::make_range_dyn *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_reshape(ir::reshape_inst* x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_splat(ir::splat_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_broadcast(ir::broadcast_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_vectorize(ir::vectorize_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_copy_to_shared(ir::copy_to_shared_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_trans(ir::trans_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  // matrix multiply
  void lower_hmma_dot(ir::dot_inst *x, LLVMContext &ctx, Function *fn, Builder &builder,
                      distributed_tile *TC, shared_tile *TA, shared_tile *TB, distributed_tile *TD, unsigned NK);
  void lower_scanline_dot(ir::dot_inst *x, LLVMContext &ctx, Function *fn, Builder &builder,
                        distributed_tile *TC, shared_tile *TA, shared_tile *TB, distributed_tile *TD, unsigned NK,
                        Type *c_ty, Function *f_mul_add);
  void lower_outer_dot(ir::dot_inst *x, LLVMContext &ctx, Function *fn, Builder &builder,
                        distributed_tile *TC, distributed_tile *TA, distributed_tile *TB, distributed_tile *TD,
                        Type *c_ty, Function *f_mul_add);
  void lower_dot(ir::dot_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  // load
  void lower_masked_load(ir::masked_load_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_load(ir::load_inst *x, LLVMContext &ctx, Function *fn, Builder &builder);
  // element-wise
  void lower_elementwise(ir::instruction *x, LLVMContext &ctx, Function *fn, Builder &builder);
  void lower_tile_instruction(ir::instruction *src, Builder &builder);



public:
  selection(analysis::memalloc *alloc, analysis::tiles *tiles, analysis::meminfo *buffer_info,
            analysis::align *alignment, analysis::axes *axes, analysis::layout *layouts,
            transform::coalesce* reorder, target *tgt, unsigned num_warps)
    : alloc_(alloc), tiles_(tiles), buffer_info_(buffer_info),
      alignment_(alignment), a_axes_(axes), layouts_(layouts),
      reorder_(reorder), tgt_(tgt), num_warps_(num_warps){ }

  void run(ir::module &src, Module &dst);

private:
  vmap_t vmap_;
  tmap_t tmap_;
  analysis::memalloc *alloc_;
  analysis::tiles *tiles_;
  analysis::axes *a_axes_;
  analysis::layout *layouts_;
  analysis::meminfo *buffer_info_;
  analysis::align *alignment_;
  transform::coalesce *reorder_;
  target *tgt_;
  std::map<unsigned, distributed_axis> axes_;
  Value *sh_mem_ptr_;
  Value *offset_a_i_, *offset_a_k_;
  Value *offset_b_j_, *offset_b_k_;
  unsigned num_packs_0_, num_packs_1_;
  unsigned pack_size_0_, pack_size_1_;
  unsigned num_warps_;
};

}
}

#endif
