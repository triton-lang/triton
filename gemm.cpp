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


bool AT = false;
bool BT = true;


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
        std::cout << instr.getName().data() << " " << param.Name << std::endl;
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

    // Globals
    llvm::Type* bool_t = llvm::Type::getInt1Ty(context);
    llvm::Type* mask_tile_t = llvm::TileType::get(bool_t, 2);
    llvm::Type* numeric_t = llvm::Type::getFloatTy(context);
    llvm::PointerType* numeric_ptr_t = llvm::PointerType::get(numeric_t, 0);
    llvm::IntegerType* int32_t = llvm::Type::getInt32Ty(context);
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
    llvm::Function* gtp = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_gtp_2d, {tile_ptr_t, numeric_ptr_t, int32_tile_t});
    llvm::Function* stp = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_stp_2d, {tile_ptr_t, int32_tile_t});
    llvm::Intrinsic::ID mma_id;
    if(!AT && !BT) mma_id = llvm::Intrinsic::tlvm_mma_nn;
    if(!AT && BT) mma_id = llvm::Intrinsic::tlvm_mma_nt;
    if(AT && !BT) mma_id = llvm::Intrinsic::tlvm_mma_tn;
    if(AT && BT) mma_id = llvm::Intrinsic::tlvm_mma_tt;
    llvm::Function* broadcast_int32 = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_broadcast_1d, {int32_tile_t, int32_slice_t});
    llvm::Function* broadcast_int1 = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_broadcast_1d, {int1_tile_t, int1_slice_t});
    llvm::Function* outer_add = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_outer_add, {int32_tile_t, int32_slice_t, int32_slice_t});
    llvm::Function* outer_and = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_outer_and, {int1_tile_t, int1_slice_t, int1_slice_t});
    llvm::Function* mma = llvm::Intrinsic::getDeclaration(module.get(), mma_id, {tile_t});
    llvm::Function* reshape = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_reshape_2d, {tile_t});
    llvm::Function* splat_2d = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_splat_2d, {mask_tile_t, tile_t, bool_t});
    llvm::Function* splat_1d = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::tlvm_splat_1d, {int32_slice_t, int32_slice_t, int32_t});
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

    // Function
    llvm::FunctionType* prototype = llvm::FunctionType::get(llvm::Type::getVoidTy(context), std::vector<llvm::Type*>{numeric_ptr_t, numeric_ptr_t, numeric_ptr_t, int32_t, int32_t, int32_t, int32_t}, false);
    llvm::Function* F = llvm::Function::Create(prototype, llvm::Function::ExternalLinkage, "kernel", module.get());
    std::vector<llvm::Value*> arguments;
    F->addAttribute(1, llvm::Attribute::ReadOnly);
    F->addAttribute(1, llvm::Attribute::NoAlias);
    F->addAttribute(2, llvm::Attribute::ReadOnly);
    F->addAttribute(2, llvm::Attribute::NoAlias);
    std::transform(F->arg_begin(), F->arg_end(), std::back_inserter(arguments), [&](llvm::Argument& x){ return &x;});
    arguments[0]->setName("pa");
    arguments[1]->setName("pb");
    arguments[2]->setName("pc");
    arguments[3]->setName("M");
    arguments[4]->setName("N");
    arguments[5]->setName("K");
    arguments[6]->setName("bound");

    // All basic blocks
    llvm::BasicBlock* PrologBB = llvm::BasicBlock::Create(context, "prologue", F);
    llvm::BasicBlock* LoopBB = llvm::BasicBlock::Create(context, "loop", F);
    llvm::BasicBlock* EarlyExitBB = llvm::BasicBlock::Create(context, "early_exit", F);
    llvm::BasicBlock* LastIterBB = llvm::BasicBlock::Create(context, "last_iter", F);
    llvm::BasicBlock* EpilogueBB = llvm::BasicBlock::Create(context, "epilogue", F);


    // First basic block
    builder.SetInsertPoint(PrologBB);

    llvm::CallInst* aasm = builder.CreateCall(read_slice_x, {bm}, "asm");
    llvm::CallInst* bbsn = builder.CreateCall(read_slice_y, {bn}, "bsn");
    llvm::CallInst* ask = builder.CreateCall(range, {builder.getInt32(0), bk}, "ask");
    llvm::CallInst* bsk = builder.CreateCall(range, {builder.getInt32(0), bk}, "bsk");

    llvm::Value *M = arguments[3], *N = arguments[4], *K = arguments[5];
    llvm::Value *bound = arguments[6];
    llvm::Value *AS0 = M, *AS1 = K;
    llvm::Value *sa0 = aasm, *sa1 = ask;
    llvm::Value *ba0 = bm, *ba1 = bk;
    llvm::Value *inca0 = _s0, *inca1 = bk;
    if(AT){
      std::swap(AS0, AS1);
      std::swap(sa0, sa1);
      std::swap(ba0, ba1);
      std::swap(inca0, inca1);
    }
    llvm::Value *BS0 = K, *BS1 = N;
    llvm::Value *sb0 = bsk, *sb1 = bbsn;
    llvm::Value *bb0 = bk, *bb1 = bn;
    llvm::Value *incb0 = bk, *incb1 = _s0;
    if(BT){
      std::swap(BS0, BS1);
      std::swap(sb0, sb1);
      std::swap(bb0, bb1);
      std::swap(incb0, incb1);
    }

    llvm::CallInst* tlda = builder.CreateCall(splat_1d, {sa1, AS0}, "lda");
    llvm::CallInst* tldb = builder.CreateCall(splat_1d, {sb1, BS1}, "ldb");
    llvm::CallInst* offa = builder.CreateCall(outer_add, {sa0, builder.CreateMul(sa1, tlda)}, "offa");
    llvm::CallInst* offb = builder.CreateCall(outer_add, {sb0, builder.CreateMul(sb1, tldb)}, "offb");
    llvm::CallInst* startpa = builder.CreateCall(gtp, {arguments[0], offa}, "startpa");
    llvm::CallInst* startpb = builder.CreateCall(gtp, {arguments[1], offb}, "startpb");
    llvm::LoadInst* startfa = builder.CreateLoad(startpa, "startfa");
    llvm::LoadInst* startfb = builder.CreateLoad(startpb, "startfb");
    llvm::Value* starta = builder.CreateCall(reshape, {startfa, ba0, ba1}, "starta");
    llvm::Value* startb = builder.CreateCall(reshape, {startfb, bb0, bb1}, "startb");
    llvm::Value* tinca0 = builder.CreateCall(splat_1d, {sa0, builder.CreateMul(inca0, AS0)});
    llvm::Value* tinca1 = builder.CreateCall(splat_1d, {sa1, builder.CreateMul(inca1, AS1)});
    llvm::Value* tincb0 = builder.CreateCall(splat_1d, {sb0, builder.CreateMul(incb0, BS0)});
    llvm::Value* tincb1 = builder.CreateCall(splat_1d, {sb1, builder.CreateMul(incb1, BS1)});
    llvm::Value* inca = builder.CreateCall(outer_add, {tinca0, tinca1}, "inca");
    llvm::Value* incb = builder.CreateCall(outer_add, {tincb0, tincb1}, "incb");
    // Enter loop
    builder.CreateBr(LoopBB);
    builder.SetInsertPoint(LoopBB);
    // PHI nodes
    llvm::PHINode* c = builder.CreatePHI(_0->getType(), 2, "c");
    llvm::PHINode* k = builder.CreatePHI(int32_t, 2, "k");
    llvm::PHINode* pa = builder.CreatePHI(startpa->getType(), 2, "pa");
    llvm::PHINode* pb = builder.CreatePHI(startpb->getType(), 2, "pb");
    llvm::PHINode *a = builder.CreatePHI(starta->getType(), 2, "a");
    llvm::PHINode *b = builder.CreatePHI(startb->getType(), 2, "b");
    llvm::Value* nextc = builder.CreateCall(mma, {a, b, c}, "nextc");
    c->addIncoming(_0, PrologBB);
    c->addIncoming(nextc, LoopBB);
    // Induction variable
    llvm::Value *nextk = builder.CreateSub(k, bk);
    k->addIncoming(K, PrologBB);
    k->addIncoming(nextk, LoopBB);
    // Update pointer
    llvm::Value *nextpa = builder.CreateCall(stp, {pa, inca}, "nextpa");
    llvm::Value *nextpb = builder.CreateCall(stp, {pb, incb}, "nextpb");
    pa->addIncoming(startpa, PrologBB);
    pa->addIncoming(nextpa, LoopBB);
    pb->addIncoming(startpb, PrologBB);
    pb->addIncoming(nextpb, LoopBB);
    // End condition
    llvm::Value* no_bounds_check = builder.CreateICmpSGT(nextk, bound);
    // Masks
    llvm::Value* maska = builder.CreateCall(splat_2d, {startfa, no_bounds_check}, "maska");
    llvm::Value* maskb = builder.CreateCall(splat_2d, {startfb, no_bounds_check}, "maskb");
    // Pre-fetch
    llvm::Value* nextfa = builder.CreateCall(masked_load, {nextpa, maska}, "nextfa");
    llvm::Value* nextfb = builder.CreateCall(masked_load, {nextpb, maskb}, "nextfb");
    llvm::Value* nexta = builder.CreateCall(reshape, {nextfa, ba0, ba1}, "nexta");
    llvm::Value* nextb = builder.CreateCall(reshape, {nextfb, bb0, bb1}, "nextb");
    a->addIncoming(starta, PrologBB);
    a->addIncoming(nexta, LoopBB);
    b->addIncoming(startb, PrologBB);
    b->addIncoming(nextb, LoopBB);
    // End condition
    builder.CreateCondBr(no_bounds_check, LoopBB,  EarlyExitBB);
    // Early exit
    builder.SetInsertPoint(EarlyExitBB);
    llvm::Value* exit = builder.CreateICmpSLE(nextk, _s0);
    builder.CreateCondBr(exit, EpilogueBB, LastIterBB);
    // Last Iteration
    builder.SetInsertPoint(LastIterBB);
    llvm::Value* in_bounds_a0 = builder.CreateICmpSLT(aasm, builder.CreateCall(splat_1d, {aasm, M}));
    llvm::Value* in_bounds_a1 = builder.CreateICmpSLT(ask, builder.CreateCall(splat_1d, {ask, bk}));
    llvm::Value* in_bounds_b0 = builder.CreateICmpSLT(bbsn, builder.CreateCall(splat_1d, {bbsn, N}));
    llvm::Value* in_bounds_b1 = builder.CreateICmpSLT(bsk, builder.CreateCall(splat_1d, {bsk, bk}));
    llvm::Value* lastmaska = builder.CreateCall(outer_and, {in_bounds_a0, in_bounds_a1}, "lastmaska");
    llvm::Value* lastmaskb = builder.CreateCall(outer_and, {in_bounds_b0, in_bounds_b1}, "lastmaskb");
    llvm::Value* lastfa = builder.CreateCall(masked_load, {nextpa, lastmaska}, "lastfa");
    llvm::Value* lastfb = builder.CreateCall(masked_load, {nextpb, lastmaskb}, "lastfb");
    llvm::Value* lasta = builder.CreateCall(reshape, {lastfa, ba0, ba1}, "lasta");
    llvm::Value* lastb = builder.CreateCall(reshape, {lastfb, bb0, bb1}, "lastb");
    llvm::Value* loop = builder.CreateICmpSGT(nextk, _s0);
    a->addIncoming(lasta, LastIterBB);
    b->addIncoming(lastb, LastIterBB);
    c->addIncoming(nextc, LastIterBB);
    k->addIncoming(nextk, LastIterBB);
    pa->addIncoming(nextpa, LastIterBB);
    pb->addIncoming(nextpb, LastIterBB);
    builder.CreateCondBr(loop, LoopBB,  EpilogueBB);
    // Epilogue
    builder.SetInsertPoint(EpilogueBB);
    llvm::CallInst* sm = builder.CreateCall(read_slice_x, {bm}, "sm");
    llvm::CallInst* sn = builder.CreateCall(read_slice_y, {bn}, "sn");
    llvm::CallInst* ldc = builder.CreateCall(splat_1d, {sn, M}, "lda");
    llvm::CallInst* offc = builder.CreateCall(outer_add, {sm, builder.CreateMul(sn, ldc)}, "offc");
    llvm::CallInst* pc = builder.CreateCall(gtp, {arguments[2], offc}, "pc");
    llvm::Value* in_bounds_c0 = builder.CreateICmpSLT(sm, builder.CreateCall(splat_1d, {sm, M}));
    llvm::Value* in_bounds_c1 = builder.CreateICmpSLT(sn, builder.CreateCall(splat_1d, {sn, N}));
    llvm::Value* maskc =  builder.CreateCall(outer_and, {in_bounds_c0, in_bounds_c1}, "maskc");
    builder.CreateCall(masked_store, {nextc, pc, maskc});
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
