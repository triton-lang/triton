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

struct distributed_axis {
  std::vector<llvm::Value*> values;
};

class tile {
protected:
  typedef std::vector<unsigned> shapes_t;

public:
  tile(const shapes_t &shapes): shapes_(shapes){ }

private:
  shapes_t shapes_;
};

class shared_tile: public tile {
public:
  using tile::tile;

};

class distributed_tile: public tile{
  typedef std::vector<distributed_axis> axes_t;

public:
  distributed_tile(const shapes_t& shapes, const axes_t &axes)
    : tile(shapes), axes_(axes) {}

private:
  axes_t axes_;
};


class selection{
  typedef std::map<ir::value *, llvm::Value *> vmap_t;
  typedef std::map<ir::basic_block *, llvm::BasicBlock *> bmap_t;
  typedef std::map<ir::value *, tile *> tmap_t;

private:
  // LLVM conversions
  llvm::Type*        llvm_type(ir::type *ty, llvm::LLVMContext &ctx);
  llvm::Value*       llvm_value(ir::value *v,llvm:: LLVMContext &ctx);
  llvm::Instruction* llvm_inst(ir::instruction *inst, llvm::LLVMContext &ctx);
  llvm::Constant*    llvm_constant(ir::constant *cst, llvm::LLVMContext &ctx);

  // grid construction
  void create_grids(std::vector<ir::instruction*> &grids,
                    std::map<unsigned*, ir::instruction*> &references,
                    ir::function *fn);
  void init_axes(ir::instruction *i, llvm::IRBuilder<> &builder, llvm::Value *u_thread_id, llvm::Value *u_warp_id);
  void init_grids(ir::function *fn, llvm::IRBuilder<> &builder);

  // lowering
  void lower_instruction(ir::instruction *src, llvm::IRBuilder<> &builder);
  void lower_tile_instruction(ir::instruction *src, llvm::IRBuilder<> &builder);

public:
  selection(allocation *alloc, tune *params): alloc_(alloc), params_(params){ }
  void run(ir::module &src, llvm::Module &dst);

private:
  vmap_t vmap_;
  bmap_t bmap_;
  tmap_t tmap_;
  allocation *alloc_;
  tune *params_;
  std::map<ir::instruction*, std::vector<distributed_axis>> axes_;
};

}
}

#endif
