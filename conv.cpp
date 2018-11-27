#include <iostream>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"



void autotune(llvm::TargetMachine *machine, llvm::Module &module){
  // Target parameters
  std::vector<unsigned> ranges = {
    // asm
    2, 16, 1, 64,
    // bsn
    2, 16, 1, 64,
    // pa
    1, 2, 4, 8,
    // pb
    1, 2, 4,
    // sm
    2, 1, 16, 2, 2, 2
  };

  // Function
  llvm::Function *F = module.getFunction("kernel");

  // Auto-tuning
  llvm::legacy::PassManager pass;
  llvm::TargetPassConfig *pass_config = static_cast<llvm::LLVMTargetMachine*>(machine)->createPassConfig(pass);
  llvm::FunctionPass *tuning_params = pass_config->createTargetTuningParameters();
  tuning_params->runOnFunction(*F);


  // Gather all parameters
  llvm::DenseSet<unsigned*> unique;
  llvm::SmallVector<unsigned*, 8> params;
  for(llvm::BasicBlock &bb: *F)
  for(llvm::Instruction &instr: bb){
    // Get tuning parameters for this particular instruction
    std::vector<llvm::TargetTuner::ParamType> tuning_params;
    machine->getTargetTuner().getParams(&instr, tuning_params);
    for(llvm::TargetTuner::ParamType &param: tuning_params){
      // This parameter has not been seen before
      if(unique.insert(param.Value).second){
        std::cout << "PARAM: " << instr.getName().data() << " " << param.Name << std::endl;
        params.push_back(param.Value);
      }
    }
  }

  // Gather all constraints
  std::vector<std::function<bool()>> constraints;
  for(llvm::BasicBlock &bb: *F)
  for(llvm::Instruction &instr: bb)
    machine->getTargetTuner().getConstraints(&instr, constraints);

  // Assign parameters
  std::cout << params.size() << " " << ranges.size() << std::endl;
  for(unsigned i = 0; i < params.size(); i++)
    *params[i] = ranges[i];

  // Verify constraints
  bool valid = true;
  for(auto &constraint: constraints){
    valid = valid & constraint();
  }

  if(!valid){
    printf("Invalid kernel parameters\n");
    exit(EXIT_FAILURE);
  }
}

int main(){
//    llvm::DebugFlag = true;

    std::string error;

    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    // Module
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module = llvm::make_unique<llvm::Module>("TLVM toy example", context);
    llvm::IRBuilder<> builder(context);

    unsigned RR = 3, SS = 3;
    unsigned Nfilt = RR * SS;
    unsigned block = 8;
    unsigned nlut = (block + Nfilt - 1)/Nfilt * Nfilt;

    // Globals
    llvm::Type* bool_t = llvm::Type::getInt1Ty(context);
    llvm::Type* mask_tile_t = llvm::TileType::get(bool_t, 2);
    llvm::Type* numeric_t = llvm::Type::getFloatTy(context);
    llvm::PointerType* numeric_ptr_t = llvm::PointerType::get(numeric_t, 0);
    llvm::IntegerType* int32_t = llvm::Type::getInt32Ty(context);
    llvm::PointerType* lut_ptr_t = llvm::PointerType::get(int32_t, 4);
    llvm::IntegerType* int1_t = llvm::Type::getInt1Ty(context);

    llvm::Type* tile_t = llvm::TileType::get(numeric_t, 2);
    llvm::Type* int32_slice_t = llvm::TileType::get(int32_t, 1);
    llvm::Type* int32_tile_t = llvm::TileType::get(int32_t, 2);
    llvm::Type* int1_slice_t = llvm::TileType::get(int1_t, 1);
    llvm::Type* int1_tile_t = llvm::TileType::get(int1_t, 2);

    llvm::PointerType* tile_ptr_t = llvm::PointerType::get(tile_t, 0);
    llvm::Function* read_slice_x = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_read_slice_x, {int32_slice_t});
    llvm::Function* read_slice_y = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_read_slice_y, {int32_slice_t});
    llvm::Function* range = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_range, {int32_slice_t});
    llvm::Function* gtp_1d = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_gtp_1d, {int32_slice_t->getPointerTo(4), int32_t->getPointerTo(4), int32_slice_t});
    llvm::Function* stp_1d = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_stp_1d, {int32_slice_t->getPointerTo(4), int32_slice_t});

    llvm::Function* gtp_2d = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_gtp_2d, {tile_ptr_t, numeric_ptr_t, int32_tile_t});
    llvm::Function* stp_2d = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_stp_2d, {tile_ptr_t, int32_tile_t});
    llvm::Intrinsic::ID mma_id = llvm::Intrinsic::tlvm_mma_nt;
    llvm::Function* outer_add = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_outer_add, {int32_tile_t, int32_slice_t, int32_slice_t});
    llvm::Function* outer_and = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_outer_and, {int1_tile_t, int1_slice_t, int1_slice_t});
    llvm::Function* outer_and_int32 = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_outer_and, {int1_tile_t, int32_slice_t, int32_slice_t});
    llvm::Function* mma = llvm::Intrinsic::getDeclaration(module.get(), mma_id, {tile_t});
    llvm::Function* reshape = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_reshape_2d, {tile_t});
    llvm::Function* splat_2d = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_splat_2d, {mask_tile_t, bool_t});
    llvm::Function* splat_1d = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_splat_1d, {int32_slice_t, int32_t});

    llvm::Function* masked_load = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_masked_load, {tile_t, tile_ptr_t, mask_tile_t});
    llvm::Function* masked_store = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_masked_store, {tile_t, tile_ptr_t, mask_tile_t});

    // Hyperparameters
    llvm::Hyperparameter *bm = llvm::Hyperparameter::get(int32_t, 0);
    llvm::Hyperparameter *bn = llvm::Hyperparameter::get(int32_t, 1);
    llvm::Hyperparameter *bk = llvm::Hyperparameter::get(int32_t, 2);

    // Constants
    llvm::Constant *_s0 = llvm::ConstantInt::get(int32_t, 0);
    llvm::Constant *_f0 = llvm::ConstantFP::get(numeric_t, 0);
    llvm::Constant *_0 = llvm::ConstantTile::get(_f0, {bm, bn});

    // LUT
    llvm::GlobalVariable *lut_array =
      new llvm::GlobalVariable(*module, llvm::ArrayType::get(int32_t, nlut), false, llvm::GlobalVariable::InternalLinkage,
                         nullptr, "lut_array", nullptr, llvm::GlobalVariable::NotThreadLocal, 4);
    llvm::Value *lut_ptr = builder.CreateBitCast(lut_array, lut_ptr_t);


    // Function
    llvm::FunctionType* prototype = llvm::FunctionType::get(llvm::Type::getVoidTy(context), std::vector<llvm::Type*>{numeric_ptr_t, numeric_ptr_t, numeric_ptr_t, int32_t, int32_t, int32_t, int32_t, int32_t}, false);
    llvm::Function* F = llvm::Function::Create(prototype, llvm::Function::ExternalLinkage, "kernel", module.get());
    std::vector<llvm::Value*> args;
    F->addAttribute(1, llvm::Attribute::ReadOnly);
    F->addAttribute(1, llvm::Attribute::NoAlias);
    F->addAttribute(2, llvm::Attribute::ReadOnly);
    F->addAttribute(2, llvm::Attribute::NoAlias);
    std::transform(F->arg_begin(), F->arg_end(), std::back_inserter(args), [&](llvm::Argument& x){ return &x;});
    llvm::Value *base_o_ptr = args[0], *base_i_ptr = args[1], *base_f_ptr = args[2];
    llvm::Value *C = args[3], *H = args[4], *W = args[5], *N = args[6], *K = args[7];
    llvm::Value *R = builder.getInt32(RR), *S = builder.getInt32(SS);

    // All basic blocks
    llvm::BasicBlock* PrologBB = llvm::BasicBlock::Create(context, "prologue", F);
    llvm::BasicBlock* LoopBB = llvm::BasicBlock::Create(context, "loop", F);
    llvm::BasicBlock* EarlyExitBB = llvm::BasicBlock::Create(context, "early_exit", F);
    llvm::BasicBlock* LastIterBB = llvm::BasicBlock::Create(context, "last_iter", F);
    llvm::BasicBlock* EpilogueBB = llvm::BasicBlock::Create(context, "epilogue", F);


    // First basic block
    builder.SetInsertPoint(PrologBB);
    llvm::Value* sa0 = builder.CreateCall(read_slice_x, {bm}, "i_slice_pqn");
    llvm::Value* sb0 = builder.CreateCall(read_slice_y, {bn}, "f_slice_k");
    llvm::Value* sa1 = builder.CreateCall(range, {builder.getInt32(0), bk}, "i_slice_crs");
    llvm::Value* sb1 = builder.CreateCall(range, {builder.getInt32(0), bk}, "f_slice_crs");

    llvm::Value* lda_w = builder.getInt32(1);
    llvm::Value* lda_h = builder.CreateMul(lda_w, W);
    llvm::Value* lda_c = builder.CreateMul(lda_h, H);
    llvm::Value* lda_n = builder.CreateMul(lda_c, C);

    llvm::Value* ldb_s = builder.getInt32(1);
    llvm::Value* ldb_r = builder.CreateMul(ldb_s, S);
    llvm::Value* ldb_c = builder.CreateMul(ldb_r, R);
    llvm::Value* ldb_k = builder.CreateMul(ldb_c, C);

    llvm::Value* CRS = builder.CreateMul(C, builder.CreateMul(R, S));
    llvm::Value* PQN = builder.CreateMul(H, builder.CreateMul(W, N));

    // Images HWN offset
    llvm::Value* sa_hw = builder.CreateUDiv(sa0, builder.CreateCall(splat_1d, {bm, N}));
    llvm::Value* sa_n = builder.CreateURem(sa0, builder.CreateCall(splat_1d, {bm, N}));
    llvm::Value* sa_h = builder.CreateUDiv(sa_hw, builder.CreateCall(splat_1d, {bm, W}));
    llvm::Value* sa_w = builder.CreateURem(sa_hw, builder.CreateCall(splat_1d, {bm, W}));
    llvm::Value* offa_0 = builder.CreateMul(sa_n, builder.CreateCall(splat_1d, {bm, lda_n}));
    offa_0 = builder.CreateAdd(offa_0, builder.CreateMul(sa_h, builder.CreateCall(splat_1d, {bm, lda_h})));
    offa_0 = builder.CreateAdd(offa_0, builder.CreateMul(sa_w, builder.CreateCall(splat_1d, {bm, lda_w})));
    // Images CRS offset
    llvm::Value* sa_cr = builder.CreateUDiv(sa1, builder.CreateCall(splat_1d, {bk, S}));
    llvm::Value* sa_s = builder.CreateURem(sa1, builder.CreateCall(splat_1d, {bk, S}));
    llvm::Value* sa_c = builder.CreateUDiv(sa_cr, builder.CreateCall(splat_1d, {bk, R}));
    llvm::Value* sa_r = builder.CreateURem(sa_cr, builder.CreateCall(splat_1d, {bk, R}));
    llvm::Value* offa_1 = builder.CreateMul(sa_c, builder.CreateCall(splat_1d, {bk, lda_c}));
    offa_1 = builder.CreateAdd(offa_1, builder.CreateMul(sa_r, builder.CreateCall(splat_1d, {bk, lda_h})));
    offa_1 = builder.CreateAdd(offa_1, builder.CreateMul(sa_s, builder.CreateCall(splat_1d, {bk, lda_w})));
    // Images pointer
    llvm::Value* off_a = builder.CreateCall(outer_add, {offa_0, offa_1});
    llvm::Value* start_pa = builder.CreateCall(gtp_2d, {base_i_ptr, off_a}, "start_i_ptr");
    llvm::LoadInst* start_aa = builder.CreateLoad(start_pa, false, "start_i_val");
    llvm::Value* start_a = builder.CreateCall(reshape, {start_aa, bm, bk}, "start_i");
    // Filters pointer
    llvm::Value* tldb_s = builder.CreateCall(splat_1d, {bk, K});
    llvm::Value* off_b = builder.CreateCall(outer_add, {sb0, builder.CreateMul(sb1, tldb_s)}, "off_f");
    llvm::Value* start_pb = builder.CreateCall(gtp_2d, {base_f_ptr, off_b}, "start_f_ptr");
    llvm::Value* start_bb = builder.CreateLoad(start_pb, false, "start_f_val");
    llvm::Value* start_b = builder.CreateCall(reshape, {start_bb, bn, bk}, "start_f");
    // Filters increment
    llvm::Value* inc_b_0 = builder.CreateCall(splat_1d, {bn, _s0}, "inc_f_0");
    llvm::Value* inc_b_1 = builder.CreateCall(splat_1d, {bk, builder.CreateMul(bk, ldb_k)}, "inc_f_1");
    llvm::Value* inc_b = builder.CreateCall(outer_add, {inc_b_0, inc_b_1}, "inc_f");
    // Delta pointers
    llvm::Value* base_incdelta = lut_ptr;
    llvm::Value* start_pincdelta = builder.CreateCall(gtp_1d, {base_incdelta, sa1}, "start_pincdelta");
    llvm::Value* base_delta = builder.CreateGEP(lut_ptr, builder.getInt32(nlut));
    llvm::Value* start_pdelta = builder.CreateCall(gtp_1d, {base_delta, builder.CreateCall(splat_1d, {bk, _s0})}, "start_pdelta");
    // Masks
    llvm::Value* _1 = builder.CreateCall(splat_1d, {bk, builder.getInt32(1)});
    llvm::Value* mask_a_1 = builder.CreateShl(_1, sa1);
    llvm::Value* base_incmask = builder.CreateGEP(lut_ptr, builder.getInt32(2*nlut), "base_incmask");
    llvm::Value* start_pincmask = builder.CreateCall(gtp_1d, {base_incmask, sa0}, "start_pincmask");
    llvm::Value* base_mask = builder.CreateGEP(lut_ptr, builder.getInt32(3*nlut), "base_mask");
    llvm::Value* start_pmask = builder.CreateCall(gtp_1d, {base_mask, sa0}, "start_pmask");
    // Enter loop
    builder.CreateBr(LoopBB);
    builder.SetInsertPoint(LoopBB);
    // PHI nodes
    llvm::PHINode* c = builder.CreatePHI(_0->getType(), 3, "c");
    llvm::PHINode* crs = builder.CreatePHI(int32_t, 3, "crs");
    llvm::PHINode* pa = builder.CreatePHI(start_pa->getType(), 3, "pa");
    llvm::PHINode* pb = builder.CreatePHI(start_pb->getType(), 3, "pb");
    llvm::PHINode *a = builder.CreatePHI(start_a->getType(), 3, "a");
    llvm::PHINode *b = builder.CreatePHI(start_b->getType(), 3, "b");
    llvm::PHINode *pdelta = builder.CreatePHI(start_pdelta->getType(), 3);
    llvm::PHINode *pincdelta = builder.CreatePHI(start_pincdelta->getType(), 3);
    llvm::PHINode *pmasks = builder.CreatePHI(start_pmask->getType(), 3);
    llvm::PHINode *pincmasks = builder.CreatePHI(start_pincmask->getType(), 3);
    llvm::Value* next_c = builder.CreateCall(mma, {a, b, c}, "next_c");
    c->addIncoming(_0, PrologBB);
    c->addIncoming(next_c, LoopBB);
    // Induction variable
    llvm::Value *next_crs = builder.CreateSub(crs, bk);
    crs->addIncoming(CRS, PrologBB);
    crs->addIncoming(next_crs, LoopBB);
    // Update pointer
    llvm::Value *inc_delta = builder.CreateLoad(pincdelta);
    llvm::Value *inc_mask = builder.CreateLoad(pincmasks);
    llvm::Value *inc_a_1 = builder.CreateLoad(pdelta);
    llvm::Value *inc_a_0 = builder.CreateCall(splat_1d, {bm, builder.getInt32(0)});
    llvm::Value *inc_a = builder.CreateCall(outer_add, {inc_a_0, inc_a_1});
    llvm::Value *next_pa = builder.CreateCall(stp_2d, {pa, inc_a}, "next_pa");
    llvm::Value *next_pb = builder.CreateCall(stp_2d, {pb, inc_b}, "next_pb");
    llvm::Value *next_pdelta = builder.CreateCall(stp_1d, {pdelta, inc_delta}, "next_pdelta");
    llvm::Value *next_pincdelta = builder.CreateCall(stp_1d, {pincdelta, inc_delta}, "next_pincdelta");
    llvm::Value *next_pmask = builder.CreateCall(stp_1d, {pmasks, inc_mask}, "next_pmask");
    llvm::Value *next_pincmask = builder.CreateCall(stp_1d, {pincmasks, inc_mask}, "next_pincmask");
    pdelta->addIncoming(start_pdelta, PrologBB);
    pdelta->addIncoming(next_pdelta, LoopBB);
    pincdelta->addIncoming(start_pincdelta, PrologBB);
    pincdelta->addIncoming(next_pincdelta, LoopBB);
    pmasks->addIncoming(start_pmask, PrologBB);
    pmasks->addIncoming(next_pmask, LoopBB);
    pincmasks->addIncoming(start_pincmask, PrologBB);
    pincmasks->addIncoming(next_pincmask, LoopBB);
    pa->addIncoming(start_pa, PrologBB);
    pa->addIncoming(next_pa, LoopBB);
    pb->addIncoming(start_pb, PrologBB);
    pb->addIncoming(next_pb, LoopBB);
    // End condition
    llvm::Value* no_bounds_check = builder.CreateICmpSGT(next_crs, builder.getInt32(0), "no_bounds_check");
    // Masks
    llvm::Value* mask_a_0 = builder.CreateLoad(pmasks, "mask_a_0");
    llvm::Value* mask_a_i32 = builder.CreateCall(outer_and_int32, {mask_a_0, mask_a_1}, "mask_a_i32");
    llvm::Value* mask_a = builder.CreateICmpNE(mask_a_i32, llvm::ConstantTile::get(_s0, {bm, bk}), "mask_a");
    llvm::Value* mask_b = builder.CreateCall(splat_2d, {bn, bk, no_bounds_check}, "mask_b");
    // Pre-fetch
    llvm::Value* next_aa = builder.CreateCall(masked_load, {next_pa, mask_a}, "next_aa");
    llvm::Value* next_bb = builder.CreateCall(masked_load, {next_pb, mask_b}, "next_bb");
    llvm::Value* next_a = builder.CreateCall(reshape, {next_aa, bm, bk}, "next_a");
    llvm::Value* next_b = builder.CreateCall(reshape, {next_bb, bn, bk}, "next_b");
    a->addIncoming(start_a, PrologBB);
    a->addIncoming(next_a, LoopBB);
    b->addIncoming(start_b, PrologBB);
    b->addIncoming(next_b, LoopBB);
    // End condition
    builder.CreateCondBr(no_bounds_check, LoopBB,  EarlyExitBB);
    // Early exit
    builder.SetInsertPoint(EarlyExitBB);
    llvm::Value* exit = builder.CreateICmpSLE(next_crs, _s0);
    builder.CreateCondBr(exit, EpilogueBB, LastIterBB);
    // Last Iteration
    builder.SetInsertPoint(LastIterBB);
    llvm::Value* in_bounds_b0 = builder.CreateICmpSLT(sb0, builder.CreateCall(splat_1d, {bn, K}));
    llvm::Value* in_bounds_b1 = builder.CreateICmpSLT(sb1, builder.CreateCall(splat_1d, {bk, next_crs}));
    llvm::Value* last_maskb = builder.CreateCall(outer_and, {in_bounds_b0, in_bounds_b1}, "last_maskb");
    llvm::Value* last_bb = builder.CreateCall(masked_load, {next_pb, last_maskb}, "last_bb");
    llvm::Value* last_b = builder.CreateCall(reshape, {last_bb, bn, bk}, "last_b");
    llvm::Value* loop = builder.CreateICmpSGT(next_crs, _s0);
    a->addIncoming(next_a, LastIterBB);
    b->addIncoming(last_b, LastIterBB);
    c->addIncoming(next_c, LastIterBB);
    crs->addIncoming(next_crs, LastIterBB);
    pa->addIncoming(next_pa, LastIterBB);
    pb->addIncoming(next_pb, LastIterBB);
    pdelta->addIncoming(next_pdelta, LastIterBB);
    pincdelta->addIncoming(next_pincdelta, LastIterBB);
    pmasks->addIncoming(next_pmask, LastIterBB);
    pincmasks->addIncoming(next_pincmask, LastIterBB);
    builder.CreateCondBr(loop, LoopBB,  EpilogueBB);

    // Epilogue
    builder.SetInsertPoint(EpilogueBB);
    llvm::Value* sc_pqn = builder.CreateCall(read_slice_x, {bm}, "o_slice_pqn");
    llvm::Value* sc_k = builder.CreateCall(read_slice_y, {bn}, "o_slice_k");
    // Output strides
    llvm::Value* ldc_q = builder.getInt32(1);
    llvm::Value* ldc_p = builder.CreateMul(lda_w, W);
    llvm::Value* ldc_k = builder.CreateMul(lda_h, H);
    llvm::Value* ldb_n = builder.CreateMul(lda_c, K);
    // Output PQN offset
    llvm::Value* sc_pq = builder.CreateUDiv(sc_pqn, builder.CreateCall(splat_1d, {bm, N}));
    llvm::Value* sc_n = builder.CreateURem(sc_pqn, builder.CreateCall(splat_1d, {bm, N}));
    llvm::Value* sc_p = builder.CreateUDiv(sc_pq, builder.CreateCall(splat_1d, {bm, W}));
    llvm::Value* sc_q = builder.CreateURem(sc_pq, builder.CreateCall(splat_1d, {bm, W}));
    llvm::Value* offc0 = builder.CreateMul(sc_n, builder.CreateCall(splat_1d, {bm, ldb_n}));
    offc0 = builder.CreateAdd(offc0, builder.CreateMul(sc_p, builder.CreateCall(splat_1d, {bm, ldc_p})));
    offc0 = builder.CreateAdd(offc0, builder.CreateMul(sc_q, builder.CreateCall(splat_1d, {bm, ldc_q})));
    // Output K offset
    llvm::Value* offc1 = builder.CreateMul(sc_k, builder.CreateCall(splat_1d, {bn, ldc_k}));
    // Output pointer
    llvm::Value* offc = builder.CreateCall(outer_add, {offc0, offc1});
    llvm::Value* pc = builder.CreateCall(gtp_2d, {base_o_ptr, offc});
    // Output masks
    llvm::Value* in_bounds_c0 = builder.CreateICmpSLT(sc_pqn, builder.CreateCall(splat_1d, {bm, PQN}));
    llvm::Value* in_bounds_c1 = builder.CreateICmpSLT(sc_k, builder.CreateCall(splat_1d, {bn, K}));
    llvm::Value* maskc = builder.CreateCall(outer_and, {in_bounds_c0, in_bounds_c1});
    builder.CreateCall(masked_store, {next_c, pc, maskc});
    builder.CreateRet(NULL);


    // Set metadata
    llvm::Metadata *md_args[] = {
      llvm::ValueAsMetadata::get(F),
      llvm::MDString::get(context, "kernel"),
      llvm::ValueAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1))
    };
    module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(llvm::MDNode::get(context, md_args));

    // Machine
    module->setTargetTriple("nvptx64-nvidia-cuda");
    auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);

    llvm::TargetMachine *machine = target->createTargetMachine(module->getTargetTriple(), "sm_52", "",
                                                               llvm::TargetOptions(), llvm::Reloc::Model(),
                                                               llvm::CodeModel::Model(), llvm::CodeGenOpt::Aggressive);
    module->setDataLayout(machine->createDataLayout());

    // Auto-tuning
    autotune(machine, *module);

    // Emit
    llvm::legacy::PassManager pass;
    llvm::SmallVector<char, 0> buffer;
    llvm::raw_svector_ostream stream(buffer);
    machine->addPassesToEmitFile(pass, stream, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);
    pass.run(*module);
    std::string src(buffer.begin(), buffer.end());

    // Execute
    std::cout << src << std::endl;
}
