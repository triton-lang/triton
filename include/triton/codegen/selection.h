#ifndef TDL_INCLUDE_CODEGEN_SELECTION_H
#define TDL_INCLUDE_CODEGEN_SELECTION_H

#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "triton/codegen/shmem_info.h"


namespace llvm{
  class Type;
  class Value;
  class Instruction;
  class Constant;
  class LLVMContext;
}

namespace triton{
namespace codegen{

class shmem_allocation;
class tune;
class shmem_info;
class target;

typedef std::vector<llvm::Value*> indices_t;

struct distributed_axis {
  size_t contiguous;
  std::vector<llvm::Value*> values;
};

class tile {
protected:
  typedef std::vector<unsigned> shapes_t;

public:
  tile(llvm::Type *ty, const shapes_t &shapes): ty_(ty), shapes_(shapes){ }
  virtual void set_value(indices_t idx, llvm::Value *v) = 0;
  virtual llvm::Value* get_value(indices_t idx) = 0;

protected:
  llvm::Type *ty_;
  shapes_t shapes_;
};

class shared_tile: public tile {
private:
  void extract_constant(llvm::Value *arg, llvm::Value *&non_cst, llvm::Value *&cst);
  void extract_constant(const indices_t &arg_idx, indices_t &non_cst_idx, indices_t &cst_idx);

  llvm::Value* shared_offset(indices_t idx);

public:
  shared_tile(llvm::Type* ty, const shapes_t &shapes, llvm::Value* ptr, llvm::IRBuilder<> &builder, llvm::Value* offset = nullptr);
  void set_vector_size(unsigned vector_size);
  void set_value(indices_t, llvm::Value *);
  llvm::Value* get_value(indices_t idx);
  llvm::Value* get_pointer() { return ptr_; }
  llvm::Value* get_offset() { return offset_; }

private:
  llvm::Value *ptr_;
  llvm::Value *offset_;
  llvm::IRBuilder<> &builder_;
  std::map<indices_t, llvm::Value*> ptr_cache_;
  unsigned vector_size_;
};

class distributed_tile: public tile{
  typedef std::vector<distributed_axis> axes_t;
  typedef std::vector<indices_t> ordered_indices_vec_t;
  typedef std::map<indices_t, unsigned> indices_map_t;
  typedef std::map<indices_t, llvm::Value*> values_map_t;

private:
  void init_indices();
  llvm::Type *make_vector_ty(llvm::Type *ty, size_t vector_size);

public:
  distributed_tile(llvm::Type *ty, const shapes_t& shapes, const axes_t &axes, llvm::IRBuilder<> &builder, bool vectorize);
  void set_value(indices_t idx, llvm::Value *v);
  llvm::Value* get_value(indices_t idx);
  unsigned get_linear_index(indices_t idx);
  void for_each(std::function<void(indices_t)> fn);
  const distributed_axis &axis(unsigned dim) { return axes_.at(dim); }

private:
  axes_t axes_;
  indices_map_t indices_;
  values_map_t values_;
  ordered_indices_vec_t ordered_indices_;
  size_t vector_size_;
  llvm::IRBuilder<> &builder_;
};


class selection{
  typedef std::map<ir::value *, llvm::Value *> vmap_t;
  typedef std::map<ir::value *, tile *> tmap_t;
  typedef std::map<std::pair<tile*, indices_t>, llvm::BasicBlock*> pmap_t;

private:
  // utils
  llvm::Type *make_vector_ty(llvm::Type *ty, size_t vector_size);
  std::vector<unsigned> extract_shapes(ir::value *v);

  // LLVM conversions
  llvm::Type*        llvm_type(ir::type *ty, llvm::LLVMContext &ctx);
  llvm::Value*       llvm_value(ir::value *v, llvm::IRBuilder<> &builder);
  llvm::Instruction* llvm_inst(ir::instruction *inst, std::function<llvm::Value*(ir::value*)> value, llvm::IRBuilder<> &builder);
  llvm::Constant*    llvm_constant(ir::constant *cst, llvm::LLVMContext &ctx);
  llvm::Value*       llvm_alloc_const(ir::alloc_const *v, llvm::Module *module, llvm::IRBuilder<> &builder);
  llvm::ArrayType*   llvm_linearized_tile_type(ir::type *ty, llvm::LLVMContext &ctx);

  // grid construction
  void create_grids(std::vector<ir::value *> &grids,
                    std::map<ir::metaparameter *, ir::value *> &references,
                    ir::function *fn);
  void create_tile(ir::value *v, llvm::IRBuilder<> &builder, const std::map<ir::metaparameter *, ir::value *> &references, std::set<ir::value *> &seen, llvm::Value *sh_mem_ptr);
  void init_axes(ir::value *i, llvm::IRBuilder<> &builder, llvm::Value *u_thread_id, llvm::Value *u_warp_id);
  void init_grids(ir::function *fn, llvm::IRBuilder<> &builder, llvm::Value *sh_mem_ptr);

  // lowering
  void lower_instruction(ir::instruction *src, llvm::IRBuilder<> &builder);
  void lower_tile_instruction(ir::instruction *src, llvm::IRBuilder<> &builder);

public:
  selection(shmem_allocation *alloc, tune *params, shmem_info *buffer_info, target *tgt)
    : alloc_(alloc), params_(params), buffer_info_(buffer_info), tgt_(tgt){ }

  void run(ir::module &src, llvm::Module &dst);

private:
  vmap_t vmap_;
  tmap_t tmap_;
  pmap_t pmap_;
  pmap_t last_block_;
  shmem_allocation *alloc_;
  tune *params_;
  target *tgt_;
  shmem_info *buffer_info_;
  std::map<ir::metaparameter*, distributed_axis> axes_;
  llvm::Value *sh_mem_ptr_;
};

}
}

#endif
