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
  std::vector<llvm::Value*> values;
};

class tile {
protected:
  typedef std::vector<unsigned> shapes_t;

public:
  tile(llvm::Type *ty, const shapes_t &shapes): shapes_(shapes){ }
  virtual void set_value(indices_t idx, llvm::Value *v) = 0;
  virtual llvm::Value* get_value(indices_t idx) = 0;

private:
  llvm::Type *ty_;
  shapes_t shapes_;
};

class shared_tile: public tile {
public:
  using tile::tile;
  void set_value(indices_t idx, llvm::Value *v) { }
  llvm::Value* get_value(indices_t idx) { return nullptr; }
};

class distributed_tile: public tile{
  typedef std::vector<distributed_axis> axes_t;
  typedef std::map<indices_t, unsigned> indices_map_t;
  typedef std::vector<llvm::Value*> values_t;

private:
  void init_indices();

public:
  distributed_tile(llvm::Type *ty, const shapes_t& shapes, const axes_t &axes);
  void set_value(indices_t idx, llvm::Value *v);
  llvm::Value* get_value(indices_t idx);
  void for_each(std::function<void(indices_t)> fn);

private:
  axes_t axes_;
  indices_map_t indices_;
  values_t values_;
};


class selection{
  typedef std::map<ir::value *, llvm::Value *> vmap_t;
  typedef std::map<ir::value *, tile *> tmap_t;

private:
  // LLVM conversions
  llvm::Type*        llvm_type(ir::type *ty, llvm::LLVMContext &ctx);
  llvm::Value*       llvm_value(ir::value *v, llvm::IRBuilder<> &builder);
  llvm::Instruction* llvm_inst(ir::instruction *inst, std::function<llvm::Value*(ir::value*)> value, llvm::IRBuilder<> &builder);
  llvm::Constant*    llvm_constant(ir::constant *cst, llvm::LLVMContext &ctx);

  // grid construction
  void create_grids(std::vector<ir::value *> &grids,
                    std::map<unsigned *, ir::value *> &references,
                    ir::function *fn);
  void create_tile(ir::value *v, llvm::IRBuilder<> &builder, const std::map<unsigned *, ir::value *> &references, std::set<ir::value *> &seen);
  void init_axes(ir::value *i, llvm::IRBuilder<> &builder, llvm::Value *u_thread_id, llvm::Value *u_warp_id);
  void init_grids(ir::function *fn, llvm::IRBuilder<> &builder);

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
  std::map<ir::value*, std::vector<distributed_axis>> axes_;
};

}
}

#endif
