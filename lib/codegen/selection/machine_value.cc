#include <numeric>
#include <iostream>
#include "llvm/IR/IRBuilder.h"
#include "triton/codegen/selection/machine_value.h"

namespace triton{
namespace codegen{

using namespace llvm;

/* Distributed Tile */
void distributed_tile::init_indices() {
  std::vector<size_t> id(axes_.size(), 0);
  // build
  size_t k = 0;
  while(true) {
    indices_t current;
    for(size_t d = 0; d < id.size(); d++)
      current.push_back(axes_[d].values[id[d]]);
    size_t sz = indices_.size();
    indices_[current] = sz;
    values_[current] = nullptr;
    ordered_indices_.push_back(current);
    id[order_[0]]++;
    while(id[order_[k]] == axes_[order_[k]].values.size()){
      if(k == id.size() - 1)
        return;
      id[order_[k++]] = 0;
      id[order_[k]]++;
    }
    k = 0;
  }
}


distributed_tile::distributed_tile(Type *ty, const shapes_t &shapes, const std::vector<int>& order, const axes_t &axes, llvm::IRBuilder<> &builder)
    : tile(ty, shapes), axes_(axes), order_(order), builder_(builder) {
  init_indices();
}

void distributed_tile::set_value(indices_t idx, Value *x) {
  assert(x->getType() == ty_ && "cannot set a value of different type");
  Value *&result = values_[idx];
  assert(!result && "value cannot be set twice");
  result = x;
}

Value* distributed_tile::get_value(indices_t idx) {
  Value *result = values_.at(idx);
  assert(result && "value has not been set");
  return result;
}

unsigned distributed_tile::get_linear_index(indices_t idx) {
  return indices_[idx];
}

indices_t distributed_tile::get_ordered_indices(unsigned id) {
  return ordered_indices_.at(id);
}


void distributed_tile::for_each(std::function<void (indices_t)> fn, int start, int end) {
  if(end < 0)
    end = ordered_indices_.size() + end + 1;
  for(unsigned i = start; i < end; i++)
    fn(ordered_indices_[i]);
}

void distributed_tile::for_each(std::function<void(indices_t)> fn, std::vector<int> starts, std::vector<int> sizes){
  int rank = sizes.size();
  int len = 1;
  for(int s: sizes)
    len *= s;

  for(int i = 0; i < len; i++){
    indices_t idx(rank);
    int current = i;
    for(int k = 0; k < rank; k++){
      idx[k] = axes_[k].values.at(starts[k] + current % sizes[k]);
      current = current / sizes[k];
    }
    fn(idx);
  }
}


/* Shared Tile */
void shared_tile::extract_constant(Value *arg, Value *&non_cst, Value *&cst) {
  BinaryOperator *bin_op = dyn_cast<BinaryOperator>(arg);
  Constant *_0 = ConstantInt::get(Type::getInt32Ty(arg->getContext()), 0);
  if(dyn_cast<Constant>(arg)){
    cst = arg;
    non_cst = _0;
    return;
  }
  if(!bin_op || bin_op->getOpcode() != llvm::BinaryOperator::Add){
    non_cst = arg;
    cst = _0;
    return;
  }
  Constant *cst_lhs = dyn_cast<Constant>(bin_op->getOperand(0));
  Constant *cst_rhs = dyn_cast<Constant>(bin_op->getOperand(1));
  if(cst_lhs && cst_rhs){
    cst = arg;
    non_cst = _0;
  }
  else if(cst_lhs){
    cst = cst_lhs;
    non_cst = bin_op->getOperand(1);
  }
  else if(cst_rhs){
    cst = cst_rhs;
    non_cst = bin_op->getOperand(0);
  }
  else{
    non_cst = arg;
    cst = _0;
  }
}

void shared_tile::extract_constant(const indices_t &arg_idx, indices_t &non_cst_idx, indices_t &cst_idx) {
  non_cst_idx.clear();
  cst_idx.clear();
  for(Value *idx: arg_idx){
    Value *non_cst, *cst;
    extract_constant(idx, non_cst, cst);
    non_cst_idx.push_back(non_cst);
    cst_idx.push_back(cst);
  }
}


Value* shared_tile::shared_offset(llvm::IRBuilder<> &builder, const shapes_t& shapes,
                                  const std::vector<int>& perm, const std::vector<int>& order,
                                  indices_t idx) {
  // strides
  std::vector<Value*> strides(shapes.size(), builder.getInt32(0));
  strides[order[0]] = builder.getInt32(1);
  for(size_t i = 1; i < idx.size(); i++)
    strides[order[i]] = builder.CreateMul(strides[order[i-1]], builder.getInt32(shapes[order[i-1]]));
  // result
  Value *result = builder.getInt32(0);
  for(size_t i = 0; i < idx.size(); i++)
    result = builder.CreateAdd(result, builder.CreateMul(idx[perm[i]], strides[i]));
  return result;
}

shared_tile::shared_tile(Type *ty, const shapes_t &shapes, const std::vector<int>& order, Value *ptr, llvm::IRBuilder<> &builder, Value *offset, const std::vector<int>& perm):
  tile(ty, shapes), order_(order), ptr_(ptr), builder_(builder), offset_(offset), vector_size_(1), perm_(perm){
  return_vector_ = false;
  if(perm_.empty()){
    perm_.resize(shapes.size());
    std::iota(perm_.begin(), perm_.end(), 0);
  }
}

Value* shared_tile::get_ptr_to(indices_t idx){
  indices_t non_cst_idx, cst_idx;
  extract_constant(idx, non_cst_idx, cst_idx);
  Value *&base_ptr = ptr_cache_[non_cst_idx];
  unsigned vector_size = vector_size_;
  Type *ty = ty_;
  if(ty->isHalfTy() && (vector_size % 2 == 0)){
    ty = IntegerType::get(ty->getContext(), 32);
    vector_size = vector_size / 2;
  }
  if(base_ptr == nullptr){
    base_ptr = builder_.CreateGEP(ptr_, shared_offset(builder_, shapes_, perm_, order_, non_cst_idx));
    if(vector_size_ > 1){
      Type *vec_ty = VectorType::get(ty, vector_size);
      Type *vec_ptr_ty = PointerType::get(vec_ty, base_ptr->getType()->getPointerAddressSpace());
      base_ptr = builder_.CreateBitCast(base_ptr, vec_ptr_ty);
    }
  }
  Value *offset = shared_offset(builder_, shapes_, perm_, order_, cst_idx);
  return builder_.CreateGEP(base_ptr, offset);
}

void shared_tile::set_value(indices_t idx, Value *value) {
  Value *ptr = builder_.CreateGEP(ptr_, shared_offset(builder_, shapes_, perm_, order_, idx));
  unsigned addr_space = ptr->getType()->getPointerAddressSpace();
  ptr = builder_.CreateBitCast(ptr, value->getType()->getPointerTo(addr_space));
  builder_.CreateStore(value, ptr);
}

void shared_tile::set_vector_size(unsigned vector_size) {
  vector_size_ = vector_size;
}

void shared_tile::set_return_mode(bool return_vector){
  return_vector_ = return_vector;
}


Value* shared_tile::get_value(indices_t idx) {
  indices_t non_cst_idx, cst_idx;
  extract_constant(idx, non_cst_idx, cst_idx);
  Value *&base_ptr = ptr_cache_[non_cst_idx];
  unsigned vector_size = vector_size_;
  Type *ty = ty_;
  if(ty->isHalfTy() && (vector_size % 2 == 0)){
    ty = IntegerType::get(ty->getContext(), 32);
    vector_size = vector_size / 2;
  }
  if(base_ptr == nullptr){
    base_ptr = builder_.CreateGEP(ptr_, shared_offset(builder_, shapes_, perm_, order_, non_cst_idx));
    if(vector_size_ > 1){
      Type *vec_ty = VectorType::get(ty, vector_size);
      Type *vec_ptr_ty = PointerType::get(vec_ty, base_ptr->getType()->getPointerAddressSpace());
      base_ptr = builder_.CreateBitCast(base_ptr, vec_ptr_ty);
    }
  }
  Value *offset = shared_offset(builder_, shapes_, perm_, order_, cst_idx);
  Value *div = offset;
  if(vector_size_ > 1)
    div = builder_.CreateUDiv(offset, builder_.getInt32(vector_size_));
  Value *ptr = builder_.CreateGEP(base_ptr, div);
  Value *result = builder_.CreateLoad(ptr);
  if(return_vector_ == false && vector_size_ > 1) {
    Value *rem = builder_.CreateURem(offset, builder_.getInt32(vector_size_));
    result = builder_.CreateExtractElement(result, rem);
  }
  return result;
}



}
}
