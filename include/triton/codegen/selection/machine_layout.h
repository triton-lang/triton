#pragma once

#ifndef _TRITON_SELECTION_MACHINE_LAYOUT_H_
#define _TRITON_SELECTION_MACHINE_LAYOUT_H_

#include <map>
#include "triton/codegen/analysis/layout.h"

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

namespace triton{

namespace ir{
class value;
}

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

class distributed_axis;
class machine_layout_t;
class tile;
class shared_tile;
class distributed_tile;
class target;

}
}

namespace triton{
namespace codegen{


class machine_layout_t {
public:
  virtual tile* create(ir::value *v) = 0;
};

class machine_layout_shared_t: public machine_layout_t {
public:
  machine_layout_shared_t(Module *mod, Builder *builder, target *tgt, analysis::allocation* alloc, Value *&sh_mem_ptr,
                          analysis::layout_shared_t* layout,
                          std::map<ir::value *, Value *>& vmap,
                          std::map<ir::value *, tile *>& tmap);

  tile* create(ir::value *v);

  Module *mod_;
  Builder *builder_;
  target *tgt_;
  analysis::allocation* alloc_;
  Value *&sh_mem_ptr_;
  analysis::layout_shared_t* layout_;
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

}
}

#endif
