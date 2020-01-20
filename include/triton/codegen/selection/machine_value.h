#pragma once

#ifndef _TRITON_SELECTION_MACHINE_VALUE_H_
#define _TRITON_SELECTION_MACHINE_VALUE_H_

#include <vector>
#include <map>
#include <functional>

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

class distributed_axis;
class machine_layout_t;
class tile;
class shared_tile;
class distributed_tile;
class target;
typedef std::vector<Value*> indices_t;

}
}

namespace triton{
namespace codegen{

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

public:
  distributed_tile(Type *ty, const shapes_t& shapes, const std::vector<int>& order, const axes_t &axes, Builder &builder);
  void set_value(indices_t idx, Value *v);
  Value* get_value(indices_t idx);
  const std::vector<int>& get_order() { return order_; }
  unsigned get_linear_index(indices_t idx);
  indices_t get_ordered_indices(unsigned id);
  void for_each(std::function<void(indices_t)> fn, int start = 0, int end = -1);
  void for_each(std::function<void(indices_t)> fn, std::vector<int> start, std::vector<int> size);

  const distributed_axis &axis(unsigned dim) { return axes_.at(dim); }
private:
  axes_t axes_;
  std::vector<int> order_;
  indices_map_t indices_;
  values_map_t values_;
  ordered_indices_vec_t ordered_indices_;
  Builder &builder_;
};

}
}

#endif
