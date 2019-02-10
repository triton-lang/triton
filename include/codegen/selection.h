#ifndef TDL_INCLUDE_CODEGEN_SELECTION_H
#define TDL_INCLUDE_CODEGEN_SELECTION_H

#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "ir/context.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/type.h"


namespace llvm{
  class Type;
  class Value;
  class Instruction;
  class Constant;
  class LLVMContext;
}

namespace tdl{
namespace codegen{

class allocation;
class tune;
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
  shared_tile(llvm::Type* ty, const shapes_t &shapes, llvm::Value* ptr, llvm::IRBuilder<> &builder);
  void set_value(indices_t, llvm::Value *);
  llvm::Value* get_value(indices_t idx);

private:
  llvm::Value *ptr_;
  llvm::IRBuilder<> &builder_;
  std::map<indices_t, llvm::Value*> ptr_cache_;
};

class distributed_tile: public tile{
  typedef std::vector<distributed_axis> axes_t;
  typedef std::map<indices_t, unsigned> indices_map_t;
  typedef std::vector<llvm::Value*> values_t;

private:
  void init_indices();

public:
  distributed_tile(llvm::Type *ty, const shapes_t& shapes, const axes_t &axes, llvm::IRBuilder<> &builder);
  void set_vectorized_iteration() { vectorized_ = true; }
  void unset_vectorized_iteration() { vectorized_ = false; }
  void set_value(indices_t idx, llvm::Value *v);
  llvm::Value* get_value(indices_t idx);
  void for_each(std::function<void(indices_t)> fn);

private:
  axes_t axes_;
  indices_map_t indices_;
  values_t values_;
  size_t vector_size_;
  llvm::IRBuilder<> &builder_;
  bool vectorized_;
};


class selection{
  typedef std::map<ir::value *, llvm::Value *> vmap_t;
  typedef std::map<ir::value *, tile *> tmap_t;

private:
  // utils
  llvm::Type *make_vector_ty(llvm::Type *ty, size_t vector_size);

  // LLVM conversions
  llvm::Type*        llvm_type(ir::type *ty, llvm::LLVMContext &ctx);
  llvm::Value*       llvm_value(ir::value *v, llvm::IRBuilder<> &builder);
  llvm::Instruction* llvm_inst(ir::instruction *inst, std::function<llvm::Value*(ir::value*)> value, llvm::IRBuilder<> &builder);
  llvm::Constant*    llvm_constant(ir::constant *cst, llvm::LLVMContext &ctx);

  // grid construction
  void create_grids(std::vector<ir::value *> &grids,
                    std::map<unsigned *, ir::value *> &references,
                    ir::function *fn);
  void create_tile(ir::value *v, llvm::IRBuilder<> &builder, const std::map<unsigned *, ir::value *> &references, std::set<ir::value *> &seen, llvm::Value *sh_mem_ptr);
  void init_axes(ir::value *i, llvm::IRBuilder<> &builder, llvm::Value *u_thread_id, llvm::Value *u_warp_id);
  void init_grids(ir::function *fn, llvm::IRBuilder<> &builder, llvm::Value *sh_mem_ptr);

  // lowering
  void lower_instruction(ir::instruction *src, llvm::IRBuilder<> &builder);
  void lower_tile_instruction(ir::instruction *src, llvm::IRBuilder<> &builder);

public:
  selection(allocation *alloc, tune *params): alloc_(alloc), params_(params){ }
  void run(ir::module &src, llvm::Module &dst);

private:
  vmap_t vmap_;
  tmap_t tmap_;
  allocation *alloc_;
  tune *params_;
  std::map<unsigned*, distributed_axis> axes_;
};

}
}

#endif
