#include <numeric>
#include "triton/codegen/selection/machine_layout.h"
#include "triton/codegen/selection/machine_value.h"
#include "triton/codegen/selection/generator.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/target.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"
#include "llvm/IR/IRBuilder.h"

namespace triton{
namespace codegen{

using namespace llvm;

inline Type *llvm_type(ir::type *ty, LLVMContext &ctx) {
  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    Type *return_ty = llvm_type(tt->get_return_ty(), ctx);
    std::vector<Type*> param_tys;
    std::transform(tt->params_begin(), tt->params_end(), std::back_inserter(param_tys),
                   [&ctx](ir::type* t){ return llvm_type(t, ctx);});
    return FunctionType::get(return_ty, param_tys, false);
  }
  // pointer
  if(ty->is_pointer_ty()){
    Type *elt_ty = llvm_type(ty->get_pointer_element_ty(), ctx);
    unsigned addr_space = ty->get_pointer_address_space();
    return PointerType::get(elt_ty, addr_space);
  }
  // integer
  if(ty->is_integer_ty()){
    unsigned bitwidth = ty->get_integer_bitwidth();
    return IntegerType::get(ctx, bitwidth);
  }
  // primitive types
  switch(ty->get_type_id()){
    case ir::type::VoidTyID:      return Type::getVoidTy(ctx);
    case ir::type::HalfTyID:      return Type::getHalfTy(ctx);
    case ir::type::FloatTyID:     return Type::getFloatTy(ctx);
    case ir::type::DoubleTyID:    return Type::getDoubleTy(ctx);
    case ir::type::X86_FP80TyID:  return Type::getX86_FP80Ty(ctx);
    case ir::type::PPC_FP128TyID: return Type::getPPC_FP128Ty(ctx);
    case ir::type::LabelTyID:     return Type::getLabelTy(ctx);
    case ir::type::MetadataTyID:  return Type::getMetadataTy(ctx);
    case ir::type::TokenTyID:     return Type::getTokenTy(ctx);
    default: break;
  }
  // unknown type
  throw std::runtime_error("unknown conversion from ir::type to Type");
}

// Grid construction
inline std::vector<Value*> delinearize(Value *trailing, const std::vector<int>& order, std::vector<int> &shapes, IRBuilder<> &builder){
  size_t dim = shapes.size();
  std::vector<Value*> result(dim);
  for(unsigned k = 0; k < dim - 1; k++){
    Constant *dim_k = builder.getInt32(shapes[order[k]]);
    Value *rem = builder.CreateURem(trailing, dim_k);
    trailing = builder.CreateUDiv(trailing, dim_k);
    result[order[k]] = rem;
  }
  result[order[dim - 1]] = trailing;
  return result;
}

inline int32_t ceil(int32_t num, int32_t div){
  return (num + div - 1)/div;
}



machine_shared_layout::machine_shared_layout(Module *mod, Builder *builder, target *tgt, analysis::allocation* alloc,
                                                 Value *&sh_mem_ptr, analysis::shared_layout *layout,
                                                 std::map<ir::value *, Value *>& vmap,
                                                 std::map<ir::value *, tile *>& tmap)
  : mod_(mod), builder_(builder), tgt_(tgt), alloc_(alloc), sh_mem_ptr_(sh_mem_ptr), layout_(layout), vmap_(vmap), tmap_(tmap) {

  Type* ty = llvm_type(layout_->get_type(), builder_->getContext());
  PointerType *ptr_ty = ty->getPointerTo(sh_mem_ptr_->getType()->getPointerAddressSpace());
  // double-buffered
  if(layout_->get_double_buffer()) {
    BasicBlock *current = builder_->GetInsertBlock();
    auto info = *layout_->get_double_buffer();
    ir::phi_node *phi = info.phi;
    BasicBlock *parent = (BasicBlock*)vmap_.at((ir::value*)(phi->get_parent()));
    if(parent->empty())
      builder_->SetInsertPoint(parent);
    else
      builder_->SetInsertPoint(&*parent->getFirstNonPHI());
    // create pointers
    ptr_ = builder_->CreatePHI(ptr_ty, 2);
    pre_ptr_ = builder_->CreateGEP(sh_mem_ptr_, builder_->getInt32(alloc_->offset(layout_)));
    pre_ptr_ = builder_->CreateBitCast(pre_ptr_, ptr_->getType());
    offset_ = builder_->CreatePHI(builder_->getInt32Ty(), 2);
    next_ptr_ = builder_->CreateGEP(ptr_, offset_, "next_ptr");
    builder_->SetInsertPoint(current);
  }
  else{
    size_t offset = alloc_->offset(layout_);
    ptr_ = builder_->CreateGEP(sh_mem_ptr_, builder_->getInt32(offset));
    ptr_ = builder_->CreateBitCast(ptr_, ptr_ty);
  }
}


tile* machine_shared_layout::create(ir::value *v) {
  Type* ty = llvm_type(layout_->get_type(), builder_->getContext());
  auto double_buffer = layout_->get_double_buffer();
  // offset
  Value *offset = nullptr;
  if(double_buffer && v == double_buffer->phi)
    offset = offset_;
  // base pointer
  Value *ptr = ptr_;
  if(double_buffer && v == double_buffer->latch)
    ptr = next_ptr_;
  else if(double_buffer && v == double_buffer->first)
    ptr = pre_ptr_;
  // create tile
  return new shared_tile(ty, layout_->get_shape(), layout_->get_order(), ptr, *builder_, offset);
}

machine_distributed_layout::machine_distributed_layout(Module *mod, Builder *builder, target *tgt,
                             analysis::axes *a_axes, std::map<unsigned, distributed_axis>& axes,
                             analysis::data_layout *layout)
  : mod_(mod), builder_(builder), tgt_(tgt), a_axes_(a_axes), axes_(axes), layout_(layout) {

}

tile *machine_distributed_layout::create(ir::value *v) {
  Type *ty = llvm_type(v->get_type()->get_scalar_ty(), builder_->getContext());
  const auto &shapes = v->get_type()->get_tile_shapes();
  size_t rank = shapes.size();
  std::vector<distributed_axis> axes(rank);
  std::vector<int> order(rank);
  // compute axes
  for(size_t d = 0; d < shapes.size(); d++){
    if(shapes[d] > 1){
      unsigned x = a_axes_->get(v, d);
      axes[d] = axes_.at(x);
    }
    else{
      axes[d].contiguous = 1;
      axes[d].values = {builder_->getInt32(0)};
    }
  }
  // compute order
  std::iota(order.begin(), order.end(), 0);
  auto cmp = [&](int x, int y) {
    unsigned axx = a_axes_->get(v, x);
    unsigned axy = a_axes_->get(v, y);
    size_t posx = layout_->find_axis(axx);
    size_t posy = layout_->find_axis(axy);
    if(posx < rank && posy < rank)
      return layout_->get_order(posx) < layout_->get_order(posy);
    return false;
  };
  std::sort(order.begin(), order.end(), cmp);
  return new distributed_tile(ty, shapes, order, axes, *builder_);
}

machine_mma_layout::machine_mma_layout(Module *mod, Builder *builder,
                          target *tgt, analysis::axes *a_axes,
                          std::map<unsigned, distributed_axis>& axes,
                          analysis::mma_layout* L, analysis::data_layout *layout_a, analysis::data_layout *layout_b)
  : machine_distributed_layout(mod, builder, tgt, a_axes, axes, L) {

  const auto& shape = L->get_shape();
  Value *_1 = builder_->getInt32(1);
  Value *_2 = builder_->getInt32(2);
  Value *_3 = builder_->getInt32(3);
  Value *_4 = builder_->getInt32(4);
  Value *_8 = builder_->getInt32(8);
  Value *_16 = builder_->getInt32(16);
  Value *_32 = builder_->getInt32(32);
  int cc = tgt_->as_nvidia()->sm();
  std::vector<Value*> idx_m;
  std::vector<Value*> idx_n;
  std::vector<Value*> idx_z;
  //
  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value *lane = builder_->CreateURem(thread, _32);
  Value *warp = builder_->CreateUDiv(thread, _32);
  /* lane offset */
  if(cc < 80){
    auto ord_a = layout_a->get_order();
    auto ord_b = layout_b->get_order();
    bool is_a_row = ord_a[0] != 0;
    bool is_b_row = ord_b[0] != 0;
    /* warp offset */
    Value *warp_0 = builder_->CreateURem(warp, builder_->getInt32(L->wpt(0)));
    Value *warp_12 = builder_->CreateUDiv(warp, builder_->getInt32(L->wpt(0)));
    Value *warp_1 = builder_->CreateURem(warp_12, builder_->getInt32(L->wpt(1)));
    Value *off_warp_m = builder_->CreateMul(warp_0, builder_->getInt32(L->spw(0)));
    Value *off_warp_n = builder_->CreateMul(warp_1, builder_->getInt32(L->spw(1)));
    // Quad offset
    Value *off_quad_m = builder_->CreateMul(builder_->CreateUDiv(builder_->CreateAnd(lane, _16), _4), builder_->getInt32(L->fpw(0)));
    Value *off_quad_n = builder_->CreateMul(builder_->CreateUDiv(builder_->CreateAnd(lane, _16), _4), builder_->getInt32(L->fpw(1)));
    // Pair offset
    Value *off_pair_m = builder_->CreateUDiv(builder_->CreateURem(lane, _16), _4);
    off_pair_m = builder_->CreateURem(off_pair_m, builder_->getInt32(L->fpw(0)));
    off_pair_m = builder_->CreateMul(off_pair_m, builder_->getInt32(4));
    Value *off_pair_n = builder_->CreateUDiv(builder_->CreateURem(lane, _16), _4);
    off_pair_n = builder_->CreateUDiv(off_pair_n, builder_->getInt32(L->fpw(0)));
    off_pair_n = builder_->CreateURem(off_pair_n, builder_->getInt32(L->fpw(1)));
    off_pair_n = builder_->CreateMul(off_pair_n, builder_->getInt32(4));
    // scale
    off_pair_m = builder_->CreateMul(off_pair_m, builder_->getInt32(L->rep(0)/2));
    off_quad_m = builder_->CreateMul(off_quad_m, builder_->getInt32(L->rep(0)/2));
    off_pair_n = builder_->CreateMul(off_pair_n, builder_->getInt32(L->rep(1)/2));
    off_quad_n = builder_->CreateMul(off_quad_n, builder_->getInt32(L->rep(1)/2));
    // Quad pair offset
    Value *off_lane_m = builder_->CreateAdd(off_pair_m, off_quad_m);
    Value *off_lane_n = builder_->CreateAdd(off_pair_n, off_quad_n);
    // a offset
    offset_a_m_ = builder_->CreateAdd(off_warp_m, off_lane_m);
    offset_a_k_ = builder_->CreateAnd(lane, _3);
    // b offsets
    offset_b_n_ = builder_->CreateAdd(off_warp_n, off_lane_n);
    offset_b_k_ = builder_->CreateAnd(lane, _3);
    // i indices
    Value *offset_c_m = builder_->CreateAdd(builder_->CreateAnd(lane, _1), offset_a_m_);
    for(unsigned m = 0; m < shape[0]; m+=L->spt(0))
    for(unsigned mm = 0; mm < L->rep(0); mm++)
      idx_m.push_back(builder_->CreateAdd(offset_c_m, builder_->getInt32(m + mm*2)));
    // j indices
    Value *offset_c_n = builder_->CreateAdd(builder_->CreateAnd(lane, _2), builder_->CreateAdd(off_warp_n, off_pair_n));
    for(unsigned n = 0; n < shape[1]; n+=L->spt(1))
    for(unsigned nn = 0; nn < L->rep(1); nn++){
      idx_n.push_back(builder_->CreateAdd(offset_c_n, builder_->getInt32(n + nn/2*4 + (nn%2)*2*L->fpw(1)*L->rep(1))));
      idx_n.push_back(builder_->CreateAdd(offset_c_n, builder_->getInt32(n + nn/2*4 + (nn%2)*2*L->fpw(1)*L->rep(1) + 1)));
    }
    if(is_a_row){
      offset_a_m_ = builder_->CreateAdd(offset_a_m_, builder_->CreateURem(thread, builder_->getInt32(4)));
      offset_a_k_ = builder_->getInt32(0);
    }
    if(!is_b_row){
      offset_b_n_ = builder_->CreateAdd(offset_b_n_, builder_->CreateURem(thread, builder_->getInt32(4)));
      offset_b_k_ = builder_->getInt32(0);
    }
    /* axes */
    axes_[L->get_axis(0)] = distributed_axis{1, idx_m, warp_0};
    axes_[L->get_axis(1)] = distributed_axis{1, idx_n, warp_1};
  }
  else{
    /* warp offset */
    Value *warp_0 = builder_->CreateURem(warp, builder_->getInt32(L->wpt(0)));
    Value *warp_12 = builder_->CreateUDiv(warp, builder_->getInt32(L->wpt(0)));
    Value *warp_1 = builder_->CreateURem(warp_12, builder_->getInt32(L->wpt(1)));
    Value *off_warp_m = builder_->CreateMul(warp_0, builder_->getInt32(L->spw(0)));
    Value *off_warp_n = builder_->CreateMul(warp_1, builder_->getInt32(L->spw(1)));
    Value *off_lane_m = builder_->CreateURem(lane, _16);
    Value *off_lane_n = builder_->CreateURem(lane, _8);
    /* offsets */
    // a offset
    offset_a_m_ = builder_->CreateAdd(off_warp_m, off_lane_m);
    offset_a_k_ = builder_->getInt32(0);
    // b offsets
    offset_b_n_ = builder_->CreateAdd(off_warp_n, off_lane_n);
    offset_b_k_ = builder_->getInt32(0);
    // c offset
    Value *off_c_m = builder_->CreateAdd(builder_->CreateUDiv(lane, _4), off_warp_m);
    Value *off_c_n = builder_->CreateAdd(builder_->CreateMul(_2, builder_->CreateURem(lane, _4)), off_warp_n);
    for(unsigned m = 0; m < shape[0]; m+=L->spt(0)){
      idx_m.push_back(builder_->CreateAdd(off_c_m, builder_->getInt32(m)));
      idx_m.push_back(builder_->CreateAdd(off_c_m, builder_->getInt32(m + 8)));
    }
    for(unsigned n = 0; n < shape[1]; n+=L->spt(1)){
      idx_n.push_back(builder_->CreateAdd(off_c_n, builder_->getInt32(n)));
      idx_n.push_back(builder_->CreateAdd(off_c_n, builder_->getInt32(n + 1)));
    }
    /* axes */
    axes_[L->get_axis(0)] = distributed_axis{1, idx_m, warp_0};
    axes_[L->get_axis(1)] = distributed_axis{1, idx_n, warp_1};
  }


}


machine_scanline_layout::machine_scanline_layout(Module *mod, Builder *builder,
                                                     target *tgt,
                                                     analysis::axes *a_axes, std::map<unsigned, distributed_axis> &axes,
                                                     analysis::scanline_layout* layout)
  : machine_distributed_layout(mod, builder, tgt, a_axes, axes, layout) {

  Value *warp_size = builder_->getInt32(32);
  Value* u_thread_id_0 = tgt_->get_local_id(mod_, *builder_, 0);
  Value *u_thread_id = builder_->CreateURem(u_thread_id_0, warp_size);
  Value *u_warp_id = builder_->CreateUDiv(u_thread_id_0, warp_size);

  auto order = layout->get_order();
  const auto& shape = layout->get_shape();
  Value* full_thread_id = builder_->CreateAdd(builder_->CreateMul(u_warp_id, builder_->getInt32(32)), u_thread_id);
  // Delinearize
  size_t dim = shape.size();
  std::vector<Value*> thread_id(dim);
  for(unsigned k = 0; k < dim - 1; k++){
    Constant *dim_k = builder_->getInt32(layout->mts(order[k]));
    Value *rem = builder_->CreateURem(full_thread_id, dim_k);
    full_thread_id = builder_->CreateUDiv(full_thread_id, dim_k);
    thread_id[order[k]] = rem;
  }
  thread_id[order[dim - 1]] = full_thread_id;
  // Create axes
  for(unsigned k = 0; k < dim; k++) {
    int nts = layout->nts(k);
    int mts = layout->mts(k);
    std::string str_k = std::to_string(k);
    Value *contiguous_k = builder_->getInt32(nts);
    Value *scaled_thread_id = builder_->CreateMul(thread_id[k], contiguous_k);
    unsigned per_block  = nts * mts;
    unsigned per_thread = nts * shape[k] / per_block;
    std::vector<Value*> idx_list(per_thread);
    for(unsigned n = 0 ; n < per_thread; n++){
      unsigned offset = n / nts * per_block + n % nts;
      idx_list[n] = builder_->CreateAdd(scaled_thread_id, builder_->getInt32(offset), "idx_" + str_k + "_" + std::to_string(n));
    }
    axes_[layout->get_axis(k)] = distributed_axis{nts, idx_list, thread_id[k]};
  }
}


}
}
