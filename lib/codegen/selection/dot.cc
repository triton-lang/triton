#include "common.h"



namespace triton::codegen {

using namespace llvm;

// namespace {
/**
 * \brief Code Generation for `mma.884` (V100)
 */
//TODO: clean-up
void generator::visit_mma884(ir::dot_inst* C, ir::value *A, ir::value *B, ir::value *D, unsigned NK) {
  // shapes
  auto shape_c = C->get_type()->get_block_shapes();
  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  // order
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();
  // layouts
  analysis::mma_layout*    layout_c = layouts_->get(C)->to_mma();
  analysis::shared_layout* layout_a = layouts_->get(A)->to_shared();
  analysis::shared_layout* layout_b = layouts_->get(B)->to_shared();
  // vectorization
  int vec_a = swizzle_->get_vec(layout_a);
  int vec_b = swizzle_->get_vec(layout_b);
  // strides
  bool is_a_row = ord_a[0] != 0;
  bool is_b_row = ord_b[0] != 0;
  int stride_am = is_a_row ? shape_a[1] : 1;
  int stride_ak = is_a_row ? 1 : shape_a[0];
  int stride_a0 = is_a_row ? stride_ak : stride_am;
  int stride_a1 = is_a_row ? stride_am : stride_ak;
  int stride_bn = is_b_row ? 1 : shape_b[0];
  int stride_bk = is_b_row ? shape_b[1] : 1;
  int stride_b0 = is_b_row ? stride_bn : stride_bk;
  int stride_b1 = is_b_row ? stride_bk : stride_bn;
  int stride_rep_m = layout_c->wpt(0) * layout_c->fpw(0) * 8;
  int stride_rep_n = layout_c->wpt(1) * layout_c->fpw(1) * 8;
  int stride_rep_k = 1;
  // swizzling
  int per_phase_a = swizzle_->get_per_phase(layout_a);
  int max_phase_a = swizzle_->get_max_phase(layout_a);
  int step_a0   = is_a_row ? stride_rep_k : stride_rep_m;
  int num_ptr_a = std::max(2 * per_phase_a * max_phase_a / step_a0, 1);
  int per_phase_b = swizzle_->get_per_phase(layout_b);
  int max_phase_b = swizzle_->get_max_phase(layout_b);
  int step_b0   = is_b_row ? stride_rep_n : stride_rep_k;
  int num_ptr_b = std::max(2 * per_phase_b * max_phase_b / step_b0, 1);

  /* --------------------------------- */
  /* --- pre-compute pointer lanes --- */
  /* --------------------------------- */
  BasicBlock* curr_bb = builder_->GetInsertBlock();
  BasicBlock* entry = &curr_bb->getParent()->getEntryBlock();
  if(entry != curr_bb)
    builder_->SetInsertPoint(entry->getTerminator());
  Value* off_a0 = is_a_row ? offset_a_k_[layout_c] : offset_a_m_[layout_c];
  Value* off_a1 = is_a_row ? offset_a_m_[layout_c] : offset_a_k_[layout_c];
  Value* phase_a = urem(udiv(off_a1, i32(per_phase_a)), i32(max_phase_a));
  std::vector<Value*> off_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++){
    Value* off_a0i = add(off_a0, i32(i*(is_a_row?4:stride_rep_m)));
    off_a0i = exact_udiv(off_a0i, i32(vec_a));
    off_a0i = xor_(off_a0i, phase_a);
    off_a0i = mul(off_a0i, i32(vec_a));
    off_a[i] = add(mul(off_a0i, i32(stride_a0)), mul(off_a1, i32(stride_a1)));
  }
  Value* off_b0 = is_b_row ? offset_b_n_[layout_c] : offset_b_k_[layout_c];
  Value* off_b1 = is_b_row ? offset_b_k_[layout_c] : offset_b_n_[layout_c];
  Value* phase_b = urem(udiv(off_b1, i32(per_phase_b)), i32(max_phase_b));
  std::vector<Value*> off_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++){
    Value* off_b0i = add(off_b0, i32(i*(is_b_row?stride_rep_n:4)));
    off_b0i = udiv(off_b0i, i32(vec_b));
    off_b0i = xor_(off_b0i, phase_b);
    off_b0i = mul(off_b0i, i32(vec_b));
    off_b[i] = add(mul(off_b0i, i32(stride_b0)), mul(off_b1, i32(stride_b1)));
  }
  builder_->SetInsertPoint(curr_bb);

  /* --------------------------------- */
  /* ---       MMA intrinsic       --- */
  /* --------------------------------- */
  Type *f16x2_ty = vec_ty(f16_ty, 2);
  Type *ret_ty = StructType::get(*ctx_, {f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty});
  std::vector<Type*> arg_ty = {f16x2_ty, f16x2_ty, f16x2_ty, f16x2_ty,
                               f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty};
  InlineAsm *mma = InlineAsm::get(FunctionType::get(ret_ty, arg_ty, false),
                                             " mma.sync.aligned.m8n8k4."
                                             + std::string(is_a_row ? "row" : "col")
                                             + "."
                                             + std::string(is_b_row ? "row" : "col")
                                             + ".f32.f16.f16.f32 "
                                             "{$0, $1, $2, $3, $4, $5, $6, $7}, "
                                             "{$8, $9}, "
                                             "{$10, $11}, "
                                             "{$0, $1, $2, $3, $4, $5, $6, $7};", "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,0,1,2,3,4,5,6,7", false);


  std::vector<Value*> ptr_a(num_ptr_a);
  std::vector<Value*> ptr_b(num_ptr_b);
  std::map<std::pair<int, int>, std::pair<Value*, Value*>> has, hbs;
  for(int i = 0; i < num_ptr_a; i++)
    ptr_a[i] = gep(shmems_[A], off_a[i]);
  for(int i = 0; i < num_ptr_b; i++)
    ptr_b[i] = gep(shmems_[B], off_b[i]);


  // initialize accumulators
  std::vector<Value*> acc;
  for(indices_t idx: idxs_.at(C))
    acc.push_back(vals_[D][idx]);

  unsigned num_m = layout_c->rep(0) * shape_c[0] / layout_c->shape_per_cta(0);
  unsigned num_n = layout_c->rep(1) * shape_c[1] / layout_c->shape_per_cta(1);

  // create mma & unpack result
  auto call_mma = [&](unsigned m, unsigned n, unsigned K) {
    auto ha = has[{m, K}];
    auto hb = hbs[{n, K}];
    // arguments
    std::vector<size_t> idx = {
      (m*2 + 0) + (n*4 + 0)*num_m, (m*2 + 0) + (n*4 + 1)*num_m,
      (m*2 + 1) + (n*4 + 0)*num_m, (m*2 + 1) + (n*4 + 1)*num_m,
      (m*2 + 0) + (n*4 + 2)*num_m, (m*2 + 0) + (n*4 + 3)*num_m,
      (m*2 + 1) + (n*4 + 2)*num_m, (m*2 + 1) + (n*4 + 3)*num_m
    };
    std::vector<Value*> args = {ha.first, ha.second, hb.first, hb.second};
    for(unsigned i = 0; i < 8; i++)
      args.push_back(acc[idx[i]]);
    // execute mma
    Value *nc = call(mma, args);
    // unpack
    for(unsigned i = 0; i < 8; i++)
      acc[idx[i]] = extract_val(nc, {i});
  };

  ir::phi_node* phiA = dynamic_cast<ir::phi_node*>(A);
  ir::phi_node* phiB = dynamic_cast<ir::phi_node*>(B);

  // Cache lds value. If values are prefetched, create phi node
  // @param inc: incoming block (0 = header, 1 = loop)
  auto register_lds =
    [&](decltype(has)& vals, int m, int K, int inc, Value* val0, Value *val1, bool is_prefetch) {
      if (K == 0 && is_prefetch) {
        ir::basic_block* inc_block = phiA->get_incoming_block(inc);
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{m, K}].first, val0, inc_block));
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{m, K}].second, val1, inc_block));
      } else
        vals[{m, K}] = {val0, val1};
  };

  auto load_a = [&](int m, int K, int inc, bool is_prefetch) {
    int offidx = (is_a_row ? K/4 : m) % num_ptr_a;
    Value* ptra;
    if(K==0 && is_prefetch){
      if(inc == 0)
        ptra = gep(shared_pre_ptr_[layout_a], off_a[offidx]);
      else
        ptra = gep(shared_next_ptr_[layout_a], off_a[offidx]);
    }
    else
      ptra = ptr_a[offidx];
    int step_am = is_a_row ? m : m / (num_ptr_a)*(num_ptr_a);
    int step_ak = is_a_row ? K / (num_ptr_a*vec_a)*(num_ptr_a*vec_a) : K;
    Value* pa =  gep(ptra, i32(step_am*stride_rep_m*stride_am + step_ak*stride_ak));
    Value* ha = load(bit_cast(pa, ptr_ty(vec_ty(i32_ty, vec_a/2), 3)));
    // record lds that needs to be moved
    if (K == 0 && inc == 1 && is_prefetch)
      prefetch_latch_to_bb_[phiA->get_incoming_value(1)].push_back(ha);
    Value *ha00 = bit_cast(extract_elt(ha, i32(0)), f16x2_ty);
    Value *ha01 = bit_cast(extract_elt(ha, i32(1)), f16x2_ty);
    register_lds(has, m, K, inc, ha00, ha01, is_prefetch);
    if(vec_a > 4){
      Value *ha10 = bit_cast(extract_elt(ha, i32(2)), f16x2_ty);
      Value *ha11 = bit_cast(extract_elt(ha, i32(3)), f16x2_ty);
      if(is_a_row)
        register_lds(has, m, K+4, inc, ha10, ha11, is_prefetch);
      else
        register_lds(has, m+1, K, inc, ha10, ha11, is_prefetch);
    }
  };

  auto load_b = [&](int n, int K, int inc, bool is_prefetch) {
    int offidx = (is_b_row? n : K/4) % num_ptr_b;
    Value* ptrb;
    if(K==0 && is_prefetch){
      if(inc == 0)
        ptrb = gep(shared_pre_ptr_[layout_b], off_b[offidx]);
      else
        ptrb = gep(shared_next_ptr_[layout_b], off_b[offidx]);
    } else
      ptrb = ptr_b[offidx];

    int stepbn = is_b_row ? n / (num_ptr_b)*(num_ptr_b) : n;
    int stepbk = is_b_row ? K : K / (num_ptr_b*vec_b)*(num_ptr_b*vec_b);
    Value* pb =   gep(ptrb, i32(stepbn*stride_rep_n*stride_bn + stepbk*stride_bk));
    Value* hb =   load(bit_cast(pb, ptr_ty(vec_ty(i32_ty, vec_b/2), 3)));
    // record lds that needs to be moved
    if (K == 0 && inc == 1 && is_prefetch)
      prefetch_latch_to_bb_[phiB->get_incoming_value(1)].push_back(hb);
    Value *hb00 = bit_cast(extract_elt(hb, i32(0)), f16x2_ty);
    Value *hb01 = bit_cast(extract_elt(hb, i32(1)), f16x2_ty);
    register_lds(hbs, n, K, inc, hb00, hb01, is_prefetch);
    if(vec_b > 4){
      Value *hb10 = bit_cast(extract_elt(hb, i32(2)), f16x2_ty);
      Value *hb11 = bit_cast(extract_elt(hb, i32(3)), f16x2_ty);
      if(is_b_row)
        register_lds(hbs, n+1, K, inc, hb10, hb11, is_prefetch);
      else
        register_lds(hbs, n, K+4, inc, hb10, hb11, is_prefetch);
    }

  };

  // update accumulators
  if (C->is_prefetched()) {
    // create phis
    builder_->SetInsertPoint(curr_bb->getFirstNonPHI());
    for (unsigned m = 0; m < num_m/2; m += is_a_row?1:2) {
      has[{m, 0}].first = phi(f16x2_ty, 2);
      has[{m, 0}].second = phi(f16x2_ty, 2);
      if (!is_a_row && vec_a>4) {
        has[{m+1, 0}].first = phi(f16x2_ty, 2);
        has[{m+1, 0}].second = phi(f16x2_ty, 2);
      }
    }
    for (unsigned n = 0; n < num_n/2; n += is_b_row?2:1) {
      hbs[{n, 0}].first = phi(f16x2_ty, 2);
      hbs[{n, 0}].second = phi(f16x2_ty, 2);
      if (is_b_row && vec_b>4) {
        hbs[{n+1, 0}].first = phi(f16x2_ty, 2);
        hbs[{n+1, 0}].second = phi(f16x2_ty, 2);
      }
    }

    // insert prefetched lds at the end of loop header
    builder_->SetInsertPoint(bbs_[phiA->get_incoming_block(0)]->getTerminator());
    for (unsigned m = 0; m < num_m/2; m += is_a_row?1:2)
      load_a(m, 0, 0, true);
    for (unsigned n = 0; n < num_n/2; n += is_b_row?2:1)
      load_b(n, 0, 0, true);

    // update accumulators
    builder_->SetInsertPoint(curr_bb);
    for (unsigned K = 0; K < NK; K += 4) {
      int NEXTK = (K + 4) % NK;
      // prefetch A
      for (unsigned m = 0; m < num_m/2; m+=is_a_row?1:2)
        load_a(m, NEXTK, 1, true);
      // prefetch B
      for (unsigned n = 0; n < num_n/2; n+=is_b_row?2:1)
        load_b(n, NEXTK, 1, true);
      // tensor core ops
      for(unsigned m = 0; m < num_m/2; m++)
      for(unsigned n = 0; n < num_n/2; n++){
        call_mma(m, n, K);
      }
    }
  } else { // not prefetched
    for(unsigned K = 0; K < NK; K += 4)
    for(unsigned m = 0; m < num_m/2; m++)
    for(unsigned n = 0; n < num_n/2; n++) {
      if(has.find({m, K}) == has.end())
        load_a(m, K, /*inc*/0, /*is_prefetch*/false);
      if(hbs.find({n, K}) == hbs.end())
        load_b(n, K, /*inc*/0, /*is_prefetch*/false);
      call_mma(m, n, K);
    }
  }

  // write back accumulators
  for(size_t i = 0; i < idxs_.at(C).size(); i++)
    vals_[C][idxs_[C][i]] = acc[i];
}

namespace {
class mma16816_smem_loader {
public:
  mma16816_smem_loader(int wpt, std::vector<int> order, int k_order, 
                       std::vector<unsigned> tile_shape, 
                       std::vector<int> instr_shape, std::vector<int> mat_shape, 
                       int per_phase, int max_phase, int dtsize, Builder *builder, 
                       adder add, multiplier mul, geper gep)
                      : wpt_(wpt), order_(order), k_order_(k_order), tile_shape_(tile_shape),
                        instr_shape_(instr_shape), mat_shape_(mat_shape), 
                        per_phase_(per_phase), max_phase_(max_phase), dtsize_(dtsize), builder_(builder),
                        add(add), mul(mul), gep(gep) {
    // compute compile-time constant variables & types
    c_mat_shape_ = mat_shape[order[0]];
    s_mat_shape_ = mat_shape[order[1]];

    c_stride_ = tile_shape[order[1]];
    s_stride_ = tile_shape[order[0]];

    // rule: k must be the fast-changing axis
    need_trans_ = k_order_ != order_[0];
    can_use_ldmatrix_ = dtsize == 2 || (!need_trans_);

    // std::cout << can_use_ldmatrix_ << std::endl;
    // std::cout << need_trans_ << std::endl;

    // we need more pointers at the fast-changing axis, 
    if (can_use_ldmatrix_)
      num_ptr_ = tile_shape[order[0]] / (order[0] == k_order? 1 : wpt) / instr_shape[order[0]];
    else // warning: this only works for tf32 & need transpose
      num_ptr_ = tile_shape[order[0]] / wpt / mat_shape[order[0]];


    // load_v4 stride (in num of mats)
    int load_stride_in_mat[2];
    load_stride_in_mat[k_order] = 2; // instr_shape[k_order] / mat_shape[k_order], always 2
    load_stride_in_mat[k_order^1] = wpt * (instr_shape[k_order^1] / mat_shape[k_order^1]);
    p_load_stride_in_mat_ = load_stride_in_mat[order[0]];
    // stride in mat, used by load_v4
    s_mat_stride_ = load_stride_in_mat[order[1]] / (instr_shape[order[1]]/mat_shape[order[1]]);
  }

  std::vector<Value*> compute_offs(Value *warp_off, Value *lane) {
    // TODO: this needs to be moved to constructor (and extracted to arr_order)
    mat_arr_stride_  = (k_order_ == 1) ? 1 : wpt_;
    warp_off_stride_ = instr_shape_[k_order_^1] / mat_shape_[k_order_^1];
    // start matrix logic offset (rename it as base_mat_off?)
    Value *mat_off[2] = {nullptr, nullptr};

    if (can_use_ldmatrix_) {
      // c: lane idx inside a group (a group is a collection of 8 contiguous threads)
      // s: group idx (0,1,2,3) inside a warp
      Value *c = urem(lane, i32(8));
      Value *s = udiv(lane, i32(8));
      // We can decompose s => s_0, s_1...
      Value *s0 = urem(s, i32(2));
      Value *s1 = udiv(s, i32(2));

      // We use different orders for a & b for better performance. 
      // TODO: we could extract this to arr_order (?)
      Value *k_mat_arr  = (k_order_ == 1) ? s1 : s0;
      Value *nk_mat_arr = (k_order_ == 1) ? s0 : s1;
      mat_off[k_order_^1] = add(mul(warp_off,   i32(warp_off_stride_)),
                                mul(nk_mat_arr, i32(mat_arr_stride_)));
      mat_off[k_order_]   = k_mat_arr;
      // physical offset (before swizzling)
      Value *c_mat_off = mat_off[order_[0]];
      Value *s_mat_off = mat_off[order_[1]];
      // offset inside a matrix // TODO: obviously, for tf32, this is different
      Value *s_off_in_mat = c;
      
      std::vector<Value*> offs(num_ptr_);
      Value *phase = urem(udiv(s_off_in_mat, i32(per_phase_)), i32(max_phase_));
      // pre-compute strided offset
      Value *s_off = add(s_off_in_mat, mul(s_mat_off, i32(s_mat_shape_)));
      for (int i=0; i < num_ptr_; ++i) {
        Value *c_mat_off_i = add(c_mat_off, i32(i*p_load_stride_in_mat_));
        c_mat_off_i = xor_(c_mat_off_i, phase); // smem swizzle
        offs[i] = add(mul(c_mat_off_i, i32(c_mat_shape_)), mul(s_off, i32(s_stride_)));
      }
      return offs;
    } else if (dtsize_ == 4 && need_trans_) {
      // load tf32 matrices with lds32
      Value *c_off_in_mat = udiv(lane, i32(4)); // 4 = mat_shape[order[1]]
      Value *s_off_in_mat = urem(lane, i32(4)); // 

      // TODO: double-check this.
      Value *phase = urem(udiv(s_off_in_mat, i32(per_phase_)), i32(max_phase_));
      std::vector<Value*> offs(num_ptr_);
      for (int mat = 0; mat < 4; ++mat) { // loads 4 mats each time
        int k_mat_arr_int  = (k_order_ == 1) ? mat/2 : mat%2;
        int nk_mat_arr_int = (k_order_ == 1) ? mat%2 : mat/2;
        if (k_mat_arr_int > 0) // we don't need pointers for k
          continue;
        Value *k_mat_arr  = i32(k_mat_arr_int);
        Value *nk_mat_arr = i32(nk_mat_arr_int);
        // physical offset (before swizzling)
        // TODO: does this work for b?
        Value *c_mat_off = add(mul(warp_off, i32(warp_off_stride_)),
                               mul(nk_mat_arr, i32(mat_arr_stride_)));
        Value *s_mat_off = k_mat_arr; // always 0?
        Value *s_off = add(s_off_in_mat, mul(s_mat_off, i32(s_mat_shape_)));
        // FIXME: (k_order_ == 1?) is really dirty hack
        for (int i = 0; i < num_ptr_/2; ++i) {
          Value *c_mat_off_i = add(c_mat_off, i32(i*p_load_stride_in_mat_*(k_order_ == 1?1:2)));
          c_mat_off_i = xor_(c_mat_off_i, phase);
          // Value *c_off = add(c_off_in_mat, mul(c_mat_off_i, i32(c_mat_shape_));
          // // TODO: move this out
          // c_off = urem(c_off, tile_shape_[order_[0]]);
          // s_off = urem(s_off, tile_shape_[order_[1]]);
          offs[2*i + nk_mat_arr_int] = add(add(c_off_in_mat, mul(c_mat_off_i, i32(c_mat_shape_))), 
                                           mul(s_off, i32(s_stride_)));
        }
      }
      return offs;
      // throw std::runtime_error("not implemented");
    } else
      throw std::runtime_error("invalid smem load config");
  }

  std::tuple<Value*, Value*, Value*, Value*> 
  load_x4(int mat0, int mat1, int inc, bool is_prefetch, ir::phi_node *pn,
          Value *pre_ptr, Value *next_ptr, std::vector<Value*> &off, std::vector<Value*> &ptrs,
          FunctionType *ldmatrix_ty, Type *smem_ptr_ty, 
          std::map<ir::value*, std::vector<Value*>> &prefetch_latch_to_bb_) {
    assert(mat0 % 2 == 0 && mat1 % 2 == 0 && "smem matrix load must be aligned");
    int mat_idx[2] = {mat0, mat1};
    int k = mat_idx[k_order_];

    int ptr_idx = -1;
    if (can_use_ldmatrix_)
      ptr_idx = mat_idx[order_[0]] / (instr_shape_[order_[0]] / mat_shape_[order_[0]]);
    else // tf32 & trans
      ptr_idx = mat_idx[order_[0]];

    auto get_ptr = [&](int idx) -> Value* {
      Value *ptr = nullptr;
      if (k == 0 && is_prefetch) {
        if (inc == 0)
          ptr = bit_cast(gep(pre_ptr, off.at(idx)), smem_ptr_ty);
        else
          ptr = bit_cast(gep(next_ptr, off.at(idx)), smem_ptr_ty);
      } else
        ptr = ptrs.at(idx);
      return ptr;
    };
    Value *ptr = get_ptr(ptr_idx);

    Value *res_v4 = nullptr;
    if (can_use_ldmatrix_) {
      std::string trans = need_trans_ ? ".trans" : "";
      // the offset (in byte) on the strided axis is a constant
      int s_offset = mat_idx[order_[1]] * (s_mat_stride_*s_mat_shape_) * s_stride_ * dtsize_;
      InlineAsm *ld_fn = InlineAsm::get(ldmatrix_ty, 
                                        "ldmatrix.sync.aligned.m8n8.x4" + trans + ".shared.b16 "
                                        "{$0, $1, $2, $3}, "
                                        "[$4 + " + std::to_string(s_offset) + "];",
                                        "=r,=r,=r,=r,r", true);
      assert(ptr);
      res_v4 = call(ldmatrix_ty, ld_fn, {ptr});
      if (k == 0 && inc == 1 && is_prefetch)
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(res_v4);
      return {extract_val(res_v4, std::vector<unsigned>{0}), 
              extract_val(res_v4, std::vector<unsigned>{1}),
              extract_val(res_v4, std::vector<unsigned>{2}),
              extract_val(res_v4, std::vector<unsigned>{3})};
    } else {
      // assert(false && "should not be here");
      assert(dtsize_ == 4 && need_trans_);
      Value *ptr2 = get_ptr(ptr_idx+1);
      assert(s_mat_stride_ == 1);
      int s_offset_elem = mat_idx[order_[1]] * (s_mat_stride_*s_mat_shape_) * s_stride_;
      int s_offset_arr_elem = 1 * (s_mat_stride_*s_mat_shape_) * s_stride_;
      // std::cout << "s_offset_elem: " << s_offset_elem << "\n"
      //           << "s_offset_arr_elem: " << s_offset_arr_elem << std::endl;
      // TODO: s_offset ?
      Value *elem0, *elem1, *elem2, *elem3;
      if (k_order_ == 1) {
        elem0 = load(gep(ptr,  i32(s_offset_elem)));
        elem1 = load(gep(ptr2, i32(s_offset_elem)));
        elem2 = load(gep(ptr,  i32(s_offset_elem + s_offset_arr_elem)));
        elem3 = load(gep(ptr2, i32(s_offset_elem + s_offset_arr_elem)));
      } else { // for b (k first)
        elem0 = load(gep(ptr,  i32(s_offset_elem)));
        elem2 = load(gep(ptr2, i32(s_offset_elem)));
        elem1 = load(gep(ptr,  i32(s_offset_elem + s_offset_arr_elem)));
        elem3 = load(gep(ptr2, i32(s_offset_elem + s_offset_arr_elem)));
      }
      if (k == 0 && inc == 1 && is_prefetch) {
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(elem0);
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(elem1);
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(elem2);
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(elem3);
      }
      return {elem0, elem1, elem2, elem3};
    }
  }

  int get_num_ptr() const { return num_ptr_; }

private:
  int wpt_;
  std::vector<int> order_;
  int k_order_;
  std::vector<unsigned> tile_shape_;
  std::vector<int> instr_shape_;
  std::vector<int> mat_shape_;
  int per_phase_, max_phase_;
  int dtsize_;

  // generated
  int c_mat_shape_, s_mat_shape_;
  int c_stride_, s_stride_;
  // p_: on the pointer axis
  int p_load_stride_in_mat_;
  int s_mat_stride_;
  // stride when moving to next not-k mat
  int warp_off_stride_;
  int mat_arr_stride_; // matrix arrangement (inside a load) stride  
  bool need_trans_, can_use_ldmatrix_;
  int num_ptr_;

  Builder *builder_;
  adder add;
  multiplier mul;
  geper gep;
};
}

/**
 * \brief Code Generation for `mma.16816` (A100)
 */
//TODO: clean-up
void generator::visit_mma16816(ir::dot_inst* C, ir::value *A, ir::value *B, ir::value *D, unsigned NK) {
  const std::vector<unsigned>& shapes = C->get_type()->get_block_shapes();
  std::map<std::vector<Value*>, std::vector<Value*>> fcs;
  for(indices_t idx: idxs_.at(C)){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    fcs[key].push_back(vals_[D][idx]);
  };
  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();
  analysis::mma_layout* layout = layouts_->get(C)->to_mma();
  analysis::shared_layout* layout_a = (analysis::shared_layout*)layouts_->get(C->get_operand(0));
  analysis::shared_layout* layout_b = (analysis::shared_layout*)layouts_->get(C->get_operand(1));
  bool is_a_row = ord_a[0] == 1;
  bool is_b_row = ord_b[0] == 1;

  std::vector<int> mma_instr_shape = layout->get_mma_instr_shape();
  const int mma_instr_m = mma_instr_shape[0];
  const int mma_instr_n = mma_instr_shape[1];
  const int mma_instr_k = mma_instr_shape[2];

  std::vector<int> mat_shape = layout->get_mma_mat_shape();
  const int mat_shape_m = mat_shape[0];
  const int mat_shape_n = mat_shape[1];
  const int mat_shape_k = mat_shape[2];

  const int per_phase_a = swizzle_->get_per_phase(layout_a);
  const int max_phase_a = swizzle_->get_max_phase(layout_a);
  const int per_phase_b = swizzle_->get_per_phase(layout_b);
  const int max_phase_b = swizzle_->get_max_phase(layout_b);

  const int num_rep_m = shapes[0] / layout->shape_per_cta(0);
  const int num_rep_n = shapes[1] / layout->shape_per_cta(1);
  const int num_rep_k = std::max<int>(NK/mma_instr_k, 1);

  Type *fp32_ty = f32_ty;
  Type *fp16x2_ty = vec_ty(f16_ty, 2);
  Type *bf16x2_ty = vec_ty(bf16_ty, 2);
  Type *fp16x2_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty});
  Type *bf16x2_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{bf16x2_ty, bf16x2_ty, bf16x2_ty, bf16x2_ty});
  Type *fp32_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty});

  FunctionType *ldmatrix_ty = nullptr;
  FunctionType *mma_ty = nullptr;
  Type *phi_ty = nullptr;
  Type *smem_ptr_ty = nullptr;

  ir::type *A_ir_ty = A->get_type()->get_scalar_ty();
  ir::type *B_ir_ty = B->get_type()->get_scalar_ty();
  if (A_ir_ty->is_fp16_ty() && B_ir_ty->is_fp16_ty()) {
    mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    smem_ptr_ty = ptr_ty(f16_ty, 3);
    ldmatrix_ty = FunctionType::get(fp16x2_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    phi_ty = fp16x2_ty;    
  } else if (A_ir_ty->is_bf16_ty() && B_ir_ty->is_bf16_ty()) {
    // FIXME: We should use bf16 here.
    mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    smem_ptr_ty = ptr_ty(f16_ty, 3);
    ldmatrix_ty = FunctionType::get(fp16x2_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    phi_ty = fp16x2_ty;
    // mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{bf16x2_ty, bf16x2_ty, bf16x2_ty, bf16x2_ty, bf16x2_ty, bf16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    // smem_ptr_ty = ptr_ty(bf16_ty, 3);
    // ldmatrix_ty = FunctionType::get(bf16x2_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    // phi_ty = bf16x2_ty;
  } else if (A_ir_ty->is_fp32_ty() && B_ir_ty->is_fp32_ty()) {
    mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    smem_ptr_ty = ptr_ty(fp32_ty, 3);
    ldmatrix_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    phi_ty = fp32_ty;
  } else
    throw std::runtime_error("mma16816 data type not supported");

  // left-hand-side values
  std::map<std::pair<unsigned, unsigned>, Value*> ha;
  std::map<std::pair<unsigned, unsigned>, Value*> hb;

  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
  if(FirstBB != CurrBB)
    builder_->SetInsertPoint(FirstBB->getTerminator());

  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value *lane   = urem(thread, i32(32));
  Value *warp   = udiv(thread, i32(32));
  Value *warp_mn = udiv(warp, i32(layout->wpt(0)));
  Value *warp_m  = urem(warp, i32(layout->wpt(0)));
  Value *warp_n  = urem(warp_mn, i32(layout->wpt(1)));
  std::vector<Value *>& fc = fcs.begin()->second;

  size_t dtsize_a = A->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
  size_t dtsize_b = B->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;

  // | -> k (row-major), since we have ldmatrix.trans, we only need to change stride
  // v (s0_0(0), s1_0(2), | *num_rep_k
  // m  s0_1(1), s1_1(3)) |  (stride in num of matrices(mat_stride_ak): 2)
  // -----------
  //   *num_rep_m (stride in num of matrices(mat_stride_am): 2*layout->wpt(0))
  mma16816_smem_loader a_loader(layout->wpt(0), ord_a, /*k_order*/1, shape_a, 
                                {mma_instr_m, mma_instr_k}, {mat_shape_m, mat_shape_k}, 
                                per_phase_a, max_phase_a, dtsize_a, builder_, add, mul, gep);
  std::vector<Value*> off_a = a_loader.compute_offs(warp_m, lane);
  int num_ptr_a = a_loader.get_num_ptr();

  // | -> n (col-major)
  // v (s0_0(0), | (stride: wpt(1)) | s1_0(2)  | *num_rep_n
  // k  s0_1(1), |                  | s1_1(3)) | (stride in num of matrices(mat_stride_bn): wpt(1))
  // -----------
  //   *num_rep_k (stride in num of matrices(mat_stride_bk): 2)
  mma16816_smem_loader b_loader(layout->wpt(1), ord_b, /*k_order*/0, shape_b,
                                {mma_instr_k, mma_instr_n}, {mat_shape_k, mat_shape_n},
                                per_phase_b, max_phase_b, dtsize_b, builder_, add, mul, gep);
  std::vector<Value*> off_b = b_loader.compute_offs(warp_n, lane);
  int num_ptr_b = b_loader.get_num_ptr();

  // std::cout << "mat_shape_m: " << mat_shape_m << "\n"
  //           << "mat_shape_n: " << mat_shape_n << "\n"
  //           << "mat_shape_k: " << mat_shape_k << "\n"
  //           << "per_phase_a: " << per_phase_a << "\n"
  //           << "max_phase_a: " << max_phase_a << "\n"
  //           << "num_ptr_a: " << num_ptr_a << std::endl;

  builder_->SetInsertPoint(CurrBB);
  // A pointer
  std::vector<Value*> ptrs_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++)
    ptrs_a[i] = bit_cast(gep(shmems_[A], {off_a[i]}), smem_ptr_ty);
  // B pointer
  std::vector<Value*> ptrs_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++)
    ptrs_b[i] = bit_cast(gep(shmems_[B], {off_b[i]}), smem_ptr_ty);

  InlineAsm *mma_fn = InlineAsm::get(mma_ty, layout->get_ptx_instr() +
                                             " {$0, $1, $2, $3},"
                                             " {$4, $5, $6, $7},"
                                             " {$8, $9},"
                                             " {$10, $11, $12, $13};",
                                             "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3", true);

  // create mma & unpack result, m, n, k are offsets in mat
  auto call_mma = [&](unsigned m, unsigned n, unsigned k) {
      unsigned cols_per_thread = num_rep_m * 2;
      std::vector<size_t> idx = {
        (m + 0) + (n*2 + 0)*cols_per_thread,
        (m + 0) + (n*2 + 1)*cols_per_thread,
        (m + 1) + (n*2 + 0)*cols_per_thread,
        (m + 1) + (n*2 + 1)*cols_per_thread
      };
      Value *nc = call(mma_ty, mma_fn, 
                       {ha[{m, k}], ha[{m+1, k}], ha[{m, k+1}], ha[{m+1, k+1}],
                        hb[{n, k}], hb[{n, k+1}],
                        fc[idx[0]], fc[idx[1]], fc[idx[2]], fc[idx[3]]});
      fc[idx[0]] = extract_val(nc, std::vector<unsigned>{0});
      fc[idx[1]] = extract_val(nc, std::vector<unsigned>{1});
      fc[idx[2]] = extract_val(nc, std::vector<unsigned>{2});
      fc[idx[3]] = extract_val(nc, std::vector<unsigned>{3});
  };

  ir::phi_node* phiA = dynamic_cast<ir::phi_node*>(A);
  ir::phi_node* phiB = dynamic_cast<ir::phi_node*>(B);

  auto register_lds2 =
    [&](std::map<std::pair<unsigned, unsigned>, Value*>& vals, int n, int k, int inc, Value* val, bool is_prefetch) {
      if (k < 2 && is_prefetch) {
        ir::basic_block* inc_block = phiA->get_incoming_block(inc);
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{n, k}], val, inc_block));
      } else
        vals[{n, k}] = val;
  };

  auto load_a = [&](int m, int k, int inc, bool is_prefetch) {
      auto [ha0, ha1, ha2, ha3] = a_loader.load_x4(m, k, inc, is_prefetch, phiA, shared_pre_ptr_[layout_a],
                                                   shared_next_ptr_[layout_a], off_a, ptrs_a, 
                                                   ldmatrix_ty, smem_ptr_ty, prefetch_latch_to_bb_);
      register_lds2(ha, m,   k,   inc, ha0, is_prefetch);
      register_lds2(ha, m+1, k,   inc, ha1, is_prefetch);
      register_lds2(ha, m,   k+1, inc, ha2, is_prefetch);
      register_lds2(ha, m+1, k+1, inc, ha3, is_prefetch);
  };

  auto load_b = [&](int n, int k, int inc, bool is_prefetch) {
      auto [hb0, hb1, hb2, hb3] = b_loader.load_x4(k, n, inc, is_prefetch, phiB, shared_pre_ptr_[layout_b],
                                                   shared_next_ptr_[layout_b], off_b, ptrs_b, 
                                                   ldmatrix_ty, smem_ptr_ty, prefetch_latch_to_bb_);
      register_lds2(hb, n,   k,   inc, hb0, is_prefetch);
      register_lds2(hb, n+1, k,   inc, hb2, is_prefetch);
      register_lds2(hb, n,   k+1, inc, hb1, is_prefetch);
      register_lds2(hb, n+1, k+1, inc, hb3, is_prefetch);
  };

  if (C->is_prefetched()) {
      // create phis
      builder_->SetInsertPoint(CurrBB->getFirstNonPHI());
      for(unsigned m = 0; m < num_rep_m; m++){
        ha[{2*m, 0}]   = phi(phi_ty, 2);
        ha[{2*m+1, 0}] = phi(phi_ty, 2);
        ha[{2*m, 1}]   = phi(phi_ty, 2);
        ha[{2*m+1, 1}] = phi(phi_ty, 2);
      }
      for(unsigned n = 0; n < num_rep_n; n+=2){
        hb[{n, 0}]   = phi(phi_ty, 2);
        hb[{n+1, 0}] = phi(phi_ty, 2);
        hb[{n, 1}]   = phi(phi_ty, 2);
        hb[{n+1, 1}] = phi(phi_ty, 2);
      }
      // insert prefetched lds at the end of loop header
      builder_->SetInsertPoint(bbs_[phiA->get_incoming_block(0)]->getTerminator());
      for(unsigned m = 0; m < num_rep_m; m++)
        load_a(2*m, 0, 0, true);
      for(unsigned n = 0; n < num_rep_n; n+=2)
        load_b(n, 0, 0, true);
      // update accumulators
      builder_->SetInsertPoint(CurrBB);
      for(unsigned k = 0; k < num_rep_k; ++k){ // stride of instr in mat is 2
        int next_k = (k + 1) % num_rep_k;
        // prefetch A
        for(unsigned m = 0; m < num_rep_m; m++)
          load_a(2*m, 2*next_k, 1, true);
        // prefetch B
        for(unsigned n = 0; n < num_rep_n; n+=2)
          load_b(n, 2*next_k, 1, true);
        // tensor core ops
        for(unsigned m = 0; m < num_rep_m; m++)
        for(unsigned n = 0; n < num_rep_n; n++){
          call_mma(2*m, n, 2*k);
        }
      }
  }
  else{
    for (unsigned k = 0; k < num_rep_k; k++) {
      for (unsigned m = 0; m < num_rep_m; m++)
        load_a(2*m, 2*k, 0, /*is_prefetch*/false);
      for (unsigned n = 0; n < num_rep_n; n+=2)
        load_b(n,   2*k, 0, /*is_prefetch*/false);
      for (unsigned m = 0; m < num_rep_m; m++)
      for (unsigned n = 0; n < num_rep_n; n++)
        call_mma(2*m, n, 2*k);
    }
  }
  // write back
  unsigned i = 0;
  for(indices_t idx: idxs_.at(C)){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    if(i >= fcs.at(key).size())
      i = 0;
    vals_[C][idx] = fcs.at(key)[i++];
  };
}

/**
 * \brief Code Generation for FMA-based `dot` (FP32, FP64, Default)
 */
void generator::visit_fmadot(ir::dot_inst* C, ir::value* A, ir::value* B, ir::value* D, unsigned NK, Type *c_ty, Function *f_mul_add) {
  auto shape_c = C->get_type()->get_block_shapes();
  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();
  analysis::scanline_layout* layout_c = layouts_->get(C)->to_scanline();
  analysis::shared_layout* layout_a = (analysis::shared_layout*)layouts_->get(C->get_operand(0));
  analysis::shared_layout* layout_b = (analysis::shared_layout*)layouts_->get(C->get_operand(1));
  bool is_a_row = ord_a[0] == 1;
  bool is_b_row = ord_b[0] == 1;
  std::string a_trans = is_a_row ? "" : ".trans";
  std::string b_trans = is_b_row ? ".trans" : "";
  int stride_a_m = is_a_row ? shape_a[1] : 1;
  int stride_a_k = is_a_row ? 1 : shape_a[0];
  int stride_b_n = is_b_row ? 1 : shape_b[0];
  int stride_b_k = is_b_row ? shape_b[1] : 1;
  int stride_a0 = is_a_row ? stride_a_k : stride_a_m;
  int stride_a1 = is_a_row ? stride_a_m : stride_a_k;
  int stride_b0 = is_b_row ? stride_b_n : stride_b_k;
  int stride_b1 = is_b_row ? stride_b_k : stride_b_n;
  int lda = is_a_row ? stride_a_m : stride_a_k;
  int ldb = is_b_row ? stride_b_k : stride_b_n;
  int per_phase_a = swizzle_->get_per_phase(layout_a);
  int max_phase_a = swizzle_->get_max_phase(layout_a);
  int per_phase_b = swizzle_->get_per_phase(layout_b);
  int max_phase_b = swizzle_->get_max_phase(layout_b);
  int num_ptr_a   = 8;
  int num_ptr_b   = 8;
  int vec_a = 2;
  int vec_b = 4;
  distributed_axis ax_m = axes_.at(a_axes_->get(C, 0));
  distributed_axis ax_n = axes_.at(a_axes_->get(C, 1));
//  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);

  Value* off_a0 = is_a_row ? i32(0) : mul(ax_m.thread_id, i32(ax_m.contiguous));
  Value* off_a1 = is_a_row ? mul(ax_m.thread_id, i32(ax_m.contiguous)): i32(0);
  std::vector<Value*> off_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++){
//    Value* off_a0i = add(off_a0, i32(is_a_row ? vec_a : layout_c->mts(0)*vec_a));
//    off_a0i = exact_udiv(off_a0i, i32(vec_a));
//    off_a0i = xor_(off_a0i, phase_a);
//    off_a0i = mul(off_a0i, i32(vec_a));
    off_a[i] = add(mul(off_a0, i32(stride_a0)), mul(off_a1, i32(stride_a1)));
  }
  Value* off_b0 = is_b_row ? mul(ax_n.thread_id, i32(ax_n.contiguous)): i32(0);
  Value* off_b1 = is_b_row ? i32(0) : mul(ax_n.thread_id, i32(ax_n.contiguous));
  std::vector<Value*> off_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++){
//    Value* off_b0i = add(off_b0, i32(is_b_row ? layout_c->mts(1)*vec_b : vec_b));
//    off_b0i = exact_udiv(off_b0i, i32(vec_b));
//    off_b0i = xor_(off_b0i, phase_b);
//    off_b0i = mul(off_b0i, i32(vec_b));
    off_b[i] = add(mul(off_b0, i32(stride_b0)), mul(off_b1, i32(stride_b1)));
  }
  std::vector<Value*> ptrs_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++)
    ptrs_a[i] = gep(shmems_[A], off_a[i]);
  std::vector<Value*> ptrs_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++)
    ptrs_b[i] = gep(shmems_[B], off_b[i]);

  std::map<indices_t, Value*> ret = vals_[D];
  std::map<std::pair<int, int>, Value*> has, hbs;
  auto ord = layout_c->get_order();
  for(unsigned k = 0; k < NK; k++){
    int z = 0;
    for(unsigned i = 0; i < shape_c[ord[1]]; i += layout_c->shape_per_cta(ord[1]))
    for(unsigned j = 0; j < shape_c[ord[0]]; j += layout_c->shape_per_cta(ord[0]))
    for(unsigned ii = 0; ii < layout_c->nts(ord[1]); ii++)
    for(unsigned jj = 0; jj < layout_c->nts(ord[0]); jj++){
      unsigned m = (ord[0] == 1) ? i : j;
      unsigned n = (ord[0] == 1) ? j : i;
      unsigned mm = (ord[0] == 1) ? ii : jj;
      unsigned nn = (ord[0] == 1) ? jj : ii;
      if(has.find({m + mm, k}) == has.end()){
        Value* pa = gep(ptrs_a[0], i32((m + mm)*stride_a_m + k*stride_a_k));
        Value* va = load(pa);
        has[{m + mm, k}] = va;
      }
      if(hbs.find({n + nn, k}) == hbs.end()){
        Value* pb = gep(ptrs_b[0], i32((n + nn)*stride_b_n + k*stride_b_k));
        Value* vb = load(pb);
        hbs[{n + nn, k}] = vb;
      }
      ret[idxs_[C].at(z)] = call(f_mul_add, {has[{m+mm,k}], hbs[{n+nn, k}], ret[idxs_[C].at(z)]});
      z++;
    }
  }

  for(indices_t idx: idxs_.at(C)){
    vals_[C][idx] = ret[idx];
  }
}

/**
 * \brief Code Generation for `dot`
 * Dispatches to appropriate specialized function
 */
void generator::visit_dot_inst(ir::dot_inst* dot) {
  Function *fn = builder_->GetInsertBlock()->getParent();
  Module *module = fn->getParent();
  ir::value *A = dot->get_operand(0);
  ir::value *B = dot->get_operand(1);
  ir::value *D = dot->get_operand(2);
  Type *c_ty = cvt(D->get_type()->get_scalar_ty());
  Function *f_mul_add = Intrinsic::getDeclaration(module, Intrinsic::fmuladd, std::vector<llvm::Type*>{c_ty});
  auto A_shapes = A->get_type()->get_block_shapes();
  size_t red_axis = 1;
  unsigned NK = A_shapes[red_axis];
  bool is_outer = NK == 1;
  bool is_mma = layouts_->get(dot)->to_mma();
  if(!is_outer && is_mma && tgt_->as_nvidia()->sm() < 80)
    return visit_mma884(dot, A, B, D, NK);
  if(!is_outer && is_mma && tgt_->as_nvidia()->sm() >= 80)
    return visit_mma16816(dot, A, B, D, NK); // rename it as visit_mma_v2()?
  return visit_fmadot(dot, A, B, D, NK, c_ty, f_mul_add);
}

} // namespace triton::codegen