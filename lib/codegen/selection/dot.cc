#include "common.h"


namespace triton::codegen {
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
    return visit_mma16816(dot, A, B, D, NK);
  return visit_fmadot(dot, A, B, D, NK, c_ty, f_mul_add);
}


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
  int vec_a = 8;
  int vec_b = 8;

  Type *fp32_ty = f32_ty;
  Type *fp16x2_ty = vec_ty(f16_ty, 2);
  Type *fp16x2_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty});
  Type *fp32_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty});
  FunctionType *ld_x4_ty = FunctionType::get(fp16x2_pack4_ty, std::vector<llvm::Type*>{ptr_ty(f16_ty, 3)}, false);

  // left-hand-side values
  std::map<std::pair<unsigned, unsigned>, std::pair<Value*, Value*>> ha;
  std::map<std::pair<unsigned, unsigned>, Value*> hb;

  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
  if(FirstBB != CurrBB)
    builder_->SetInsertPoint(FirstBB->getTerminator());

  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value *lane   = urem(thread, i32(32));
  Value *warp   = udiv(thread, i32(32));
  Value *warp12 = udiv(warp, i32(layout->wpt(0)));
  Value *warp0  = urem(warp, i32(layout->wpt(0)));
  Value *warp1  = urem(warp12, i32(layout->wpt(1)));
  std::vector<Value *>& fc = fcs.begin()->second;

  Value *tidr8  = urem(lane, i32(8));
  Value *phase_a = urem(udiv(tidr8, i32(per_phase_a)), i32(max_phase_a));
  Value* off_a0   = mul(tidr8, i32(lda));
  Value *off_am  = mul(add(urem(udiv(lane, i32(8)), i32(2)), mul(warp0, i32(2))), i32(8));
  Value *off_ak  = mul(udiv(lane, i32(16)), i32(8));
  off_am = urem(off_am, i32(shape_a[0]));
  off_ak = urem(off_ak, i32(shape_a[1]));
  off_a0 = add(off_a0, is_a_row ? off_ak : off_am);
  Value* off_a1 = is_a_row ? off_am : off_ak;
  std::vector<Value*> off_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++){
    Value* off_a0i = add(off_a0, i32(i*16*(is_a_row?1:layout->wpt(0))));
    off_a0i = exact_udiv(off_a0i, i32(vec_a));
    off_a0i = xor_(off_a0i, phase_a);
    off_a0i = mul(off_a0i, i32(vec_a));
    off_a[i] = add(mul(off_a0i, i32(stride_a0)), mul(off_a1, i32(stride_a1)));
  }

  Value *phase_b = urem(udiv(tidr8, i32(per_phase_b)), i32(max_phase_b));
  Value* off_b0   = mul(tidr8, i32(ldb));
  Value *off_bn  = mul(add(mul(udiv(lane, i32(16)), i32(layout->wpt(1))), mul(warp1, i32(1))), i32(8));
  Value *off_bk  = mul(urem(udiv(lane, i32(8)), i32(2)), i32(8));
  off_bn = urem(off_bn, i32(shape_b[1]));
  off_bk = urem(off_bk, i32(shape_b[0]));
  off_b0 = add(off_b0, is_b_row ? off_bn : off_bk);
  Value* off_b1 = is_b_row ? off_bk : off_bn;
  std::vector<Value*> off_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++){
    Value* off_b0i = add(off_b0, i32(i*(is_b_row?8*layout->wpt(1):16)));
    off_b0i = exact_udiv(off_b0i, i32(vec_b));
    off_b0i = xor_(off_b0i, phase_b);
    off_b0i = mul(off_b0i, i32(vec_b));
    off_b[i] = add(mul(off_b0i, i32(stride_b0)), mul(off_b1, i32(stride_b1)));
  }

  builder_->SetInsertPoint(CurrBB);
  // A pointer
  std::vector<Value*> ptrs_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++)
    ptrs_a[i] = gep(shmems_[A], {off_a[i]});
  // B pointer
  std::vector<Value*> ptrs_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++)
    ptrs_b[i] = gep(shmems_[B], {off_b[i]});

  FunctionType *mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
  InlineAsm *mma_fn = InlineAsm::get(mma_ty, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                             "{$0, $1, $2, $3}, "
                                             "{$4, $5, $6, $7}, "
                                             "{$8, $9}, "
                                             "{$10, $11, $12, $13};",
                                             "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3", true);

  unsigned num_rep_0 = shapes[0] / layout->shape_per_cta(0);
  unsigned num_rep_1 = shapes[1] / layout->shape_per_cta(1);

  // create mma & unpack result
  auto call_mma = [&](unsigned m, unsigned n, unsigned K) {
      unsigned cols_per_thread = num_rep_0 * 2;
      std::vector<size_t> idx = {
        (m*2 + 0) + (n*2 + 0)*cols_per_thread,
        (m*2 + 0) + (n*2 + 1)*cols_per_thread,
        (m*2 + 1) + (n*2 + 0)*cols_per_thread,
        (m*2 + 1) + (n*2 + 1)*cols_per_thread
      };
      Value *nc = call(mma_ty, mma_fn, {ha[{m, K}].first, ha[{m, K}].second,ha[{m, K+8}].first, ha[{m, K+8}].second,
                                                        hb[{n, K}], hb[{n, K+8}],
                                                        fc[idx[0]], fc[idx[1]], fc[idx[2]], fc[idx[3]]});
      fc[idx[0]] = extract_val(nc, std::vector<unsigned>{0});
      fc[idx[1]] = extract_val(nc, std::vector<unsigned>{1});
      fc[idx[2]] = extract_val(nc, std::vector<unsigned>{2});
      fc[idx[3]] = extract_val(nc, std::vector<unsigned>{3});
  };

  ir::phi_node* phiA = dynamic_cast<ir::phi_node*>(A);
  ir::phi_node* phiB = dynamic_cast<ir::phi_node*>(B);

  auto register_lds =
    [&](decltype(ha)& vals, int m, int K, int inc, Value* val0, Value *val1, bool is_prefetch) {
      if (K <= 8 && is_prefetch) {
        ir::basic_block* inc_block = phiA->get_incoming_block(inc);
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{m, K}].first, val0, inc_block));
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{m, K}].second, val1, inc_block));
      } else
        vals[{m, K}] = {val0, val1};
  };

  auto register_lds2 =
    [&](decltype(hb)& vals, int m, int K, int inc, Value* val, bool is_prefetch) {
      if (K <= 8 && is_prefetch) {
        ir::basic_block* inc_block = phiA->get_incoming_block(inc);
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{m, K}], val, inc_block));
      } else
        vals[{m, K}] = val;
  };

  auto load_a = [&](int m, int K, int inc, bool is_prefetch) {
      int offidx = (is_a_row ? K/16 : m) % num_ptr_a;
      Value* ptra;
      if(K == 0 && is_prefetch){
        if(inc == 0)
          ptra = gep(shared_pre_ptr_[layout_a], off_a[offidx]);
        else
          ptra = gep(shared_next_ptr_[layout_a], off_a[offidx]);
      }
      else
        ptra = ptrs_a[offidx];
      int step_am = is_a_row ? m : m / (num_ptr_a)*(num_ptr_a);
      int step_ak = is_a_row ? K / (num_ptr_a*16)*(num_ptr_a*16) : K;
      InlineAsm *ld_a0_fn = InlineAsm::get(ld_x4_ty, "ldmatrix.sync.aligned.m8n8.x4" + a_trans + ".shared.b16 "
                                                "{$0, $1, $2, $3}, [$4 + " +
                                                std::to_string(2*step_am*16*layout->wpt(0)*stride_a_m + 2*step_ak*stride_a_k) + "];",
                                                "=r,=r,=r,=r,r", true);
      Value *haa = call(ld_x4_ty, ld_a0_fn, {ptra});
      if(K == 0 && inc == 1 && is_prefetch)
          prefetch_latch_to_bb_[phiA->get_incoming_value(1)].push_back(haa);
      Value *ha0 = extract_val(haa, std::vector<unsigned>{0});
      Value *ha1 = extract_val(haa, std::vector<unsigned>{1});
      Value *ha2 = extract_val(haa, std::vector<unsigned>{2});
      Value *ha3 = extract_val(haa, std::vector<unsigned>{3});
      register_lds(ha, m, K, inc, ha0, ha1, is_prefetch);
      register_lds(ha, m, K + 8, inc, ha2, ha3, is_prefetch);
  };

  auto load_b = [&](int n, int K, int inc, bool is_prefetch) {
      int offidx = (is_b_row ? n : K/16) % num_ptr_b;
      Value* ptrb;
      if(K == 0 && is_prefetch){
        if(inc == 0)
          ptrb = gep(shared_pre_ptr_[layout_b], off_b[offidx]);
        else
          ptrb = gep(shared_next_ptr_[layout_b], off_b[offidx]);
      }
      else
        ptrb = ptrs_b[offidx];
      int step_bn = is_b_row ? n / (num_ptr_b)*(num_ptr_b) : n;
      int step_bk = is_b_row ? K : K / (num_ptr_b*8)*(num_ptr_b*8);
      InlineAsm *ld_b_fn = InlineAsm::get(ld_x4_ty, "ldmatrix.sync.aligned.m8n8.x4" + b_trans + ".shared.b16 "
                                                    "{$0, $1, $2, $3}, [$4 + " +
                                                    std::to_string(2*step_bn*8*layout->wpt(1)*stride_b_n + 2*step_bk*stride_b_k) + "];",
                                                    "=r,=r,=r,=r,r", true);
      Value *hbb = call(ld_x4_ty, ld_b_fn, {ptrb});
      if(K == 0 && inc == 1 && is_prefetch)
          prefetch_latch_to_bb_[phiB->get_incoming_value(1)].push_back(hbb);
      Value *hb0 = extract_val(hbb, std::vector<unsigned>{0});
      Value *hb1 = extract_val(hbb, std::vector<unsigned>{1});
      Value *hb2 = extract_val(hbb, std::vector<unsigned>{2});
      Value *hb3 = extract_val(hbb, std::vector<unsigned>{3});
      register_lds2(hb, n, K, inc, hb0, is_prefetch);
      register_lds2(hb, n+1, K, inc, hb2, is_prefetch);
      register_lds2(hb, n, K+8, inc, hb1, is_prefetch);
      register_lds2(hb, n+1, K+8, inc, hb3, is_prefetch);
  };

  if (C->is_prefetched()) {
      // create phis
      builder_->SetInsertPoint(CurrBB->getFirstNonPHI());
      for(unsigned m = 0; m < num_rep_0; m++){
        ha[{m, 0}].first = phi(fp16x2_ty, 2);
        ha[{m, 0}].second = phi(fp16x2_ty, 2);
        ha[{m, 8}].first = phi(fp16x2_ty, 2);
        ha[{m, 8}].second = phi(fp16x2_ty, 2);
      }
      for(unsigned n = 0; n < num_rep_1; n+=2){
        hb[{n, 0}] = phi(fp16x2_ty, 2);
        hb[{n+1, 0}] = phi(fp16x2_ty, 2);
        hb[{n, 8}] = phi(fp16x2_ty, 2);
        hb[{n+1, 8}] = phi(fp16x2_ty, 2);
      }
      // insert prefetched lds at the end of loop header
      builder_->SetInsertPoint(bbs_[phiA->get_incoming_block(0)]->getTerminator());
      for(unsigned m = 0; m < num_rep_0; m++)
        load_a(m, 0, 0, true);
      for(unsigned n = 0; n < num_rep_1; n+=2)
        load_b(n, 0, 0, true);
      // update accumulators
      builder_->SetInsertPoint(CurrBB);
      for(unsigned K = 0; K < NK; K += 16){
        int NEXTK = (K + 16) % NK;
        // prefetch A
        for(unsigned m = 0; m < num_rep_0; m++)
          load_a(m, NEXTK, 1, true);
        // prefetch B
        for(unsigned n = 0; n < num_rep_1; n+=2)
          load_b(n, NEXTK, 1, true);
        // tensor core ops
        for(unsigned m = 0; m < num_rep_0; m++)
        for(unsigned n = 0; n < num_rep_1; n++){
          call_mma(m, n, K);
        }
      }
  }
  else{
      for(unsigned K = 0; K < NK; K += 16)
      for(unsigned m = 0; m < num_rep_0; m++)
      for(unsigned n = 0; n < num_rep_1; n++){
        if(ha.find({m, K}) == ha.end())
          load_a(m, K, 0, false);
        if(hb.find({n, K})==hb.end())
          load_b(n, K, 0, false);
        call_mma(m, n, K);
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
  for(unsigned k = 0; k < NK; k++){
    int z = 0;
    for(unsigned m = 0; m < shape_c[0]; m += layout_c->shape_per_cta(0))
    for(unsigned n = 0; n < shape_c[1]; n += layout_c->shape_per_cta(1))
    for(unsigned mm = 0; mm < layout_c->nts(0); mm++)
    for(unsigned nn = 0; nn < layout_c->nts(1); nn++)
    {
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

} // namespace triton::codegen