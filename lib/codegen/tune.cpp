//#include "codegen/tune.h"

//namespace tdl{
//namespace codegen{


//// Layout binding pass
//class TLVMAddTunerConstraints: public FunctionPass {
//public:
//  static char ID;
//  TLVMAddTunerConstraints(): FunctionPass(ID){ }

//  void getAnalysisUsage(AnalysisUsage & AU) const override;
//  bool runOnFunction(Function &F) override;
//};

//// Initialization
//char TLVMAddTunerConstraints::ID = 0;
//INITIALIZE_PASS_BEGIN(TLVMAddTunerConstraints, "tlvm-add-tuner-constraints",
//                "Add Tuner Constraints (TLVM)", false, true)
//INITIALIZE_PASS_END(TLVMAddTunerConstraints, "tlvm-add-tuner-constraints",
//                    "Add Tuner Constraints (TLVM)", false, true)
//FunctionPass *llvm::createTLVMAddTunerConstraintsPass() { return new TLVMAddTunerConstraints(); }

//// Analysis usage
//void TLVMAddTunerConstraints::getAnalysisUsage(AnalysisUsage &AU) const {
//  AU.setPreservesAll();
//  FunctionPass::getAnalysisUsage(AU);
//}


//inline unsigned MDRead(MDNode* Node){
//  Metadata *MD = Node->getOperand(0).get();
//  Constant *Cst = ((ConstantAsMetadata*)MD)->getValue();
//  unsigned Result = Cst->getUniqueInteger().getZExtValue();
//  return Result;
//}

//inline unsigned getNumGT1Dim(Instruction &I){
//  unsigned Res = 0;
//  for(unsigned K = 0; K < I.getType()->getTileNumDimensions(); K++)
//  if(MDRead(I.getMetadata("nvvm.param.shape.d" + itostr(K))) > 1)
//    Res++;
//  return Res;
//}
//// Run
//bool TLVMAddTunerConstraints::runOnFunction(Function &F) {
//  LLVMContext &Ctx = F.getContext();

//  DenseMap<MDNode*, Instruction*> Refs;
//  for(Function::iterator::value_type &BB: F)
//  for(Instruction &I : BB)
//  if(isTLVMValue(&I)){
//    SmallVector<std::pair<unsigned, MDNode*>, 4> MDs;
//    I.getAllMetadata(MDs);
//    for(auto &X: MDs){
//      if(MDRead(X.second)==1)
//        continue;
//      Instruction *&Ref = Refs[X.second];
//      if(!Ref || getNumGT1Dim(I) > getNumGT1Dim(*Ref))
//        Ref = &I;
//    }
//  }
//  SmallVector<Instruction*, 4> Grids;
//  for(auto &R: Refs)
//  if(std::find(Grids.begin(), Grids.end(), R.second) == Grids.end())
//    Grids.push_back(R.second);


//  Instruction *FirstTile = Grids.front();
//  for(Instruction *I: Grids){
//    Type *Ty = I->getType();
//    size_t NumDim = Ty->getTileNumDimensions();

//    // For each dimension, the product of layout components
//    // must divide shape
//    for(size_t K = 0; K < NumDim; K++){
//      unsigned Shape = MDRead(I->getMetadata("nvvm.param.shape.d" + itostr(K)));
//      unsigned S0 = MDRead(I->getMetadata("nvvm.param.layout.p0.d" + itostr(K)));
//      unsigned S1 = MDRead(I->getMetadata("nvvm.param.layout.p1.d" + itostr(K)));
//      unsigned S2 = MDRead(I->getMetadata("nvvm.param.layout.p2.d" + itostr(K)));
//      bool Constraint = Shape % (S0*S1*S2)== 0;
//      Constant *Cst = Constraint?ConstantInt::getTrue(Ctx):ConstantInt::getFalse(Ctx);
//      I->setMetadata("nvvm.constraint.shape.d" + itostr(K), MDNode::get(Ctx, ConstantAsMetadata::get(Cst)));
//    };
//    // The number of threads per warp is 32
//    {
//      int NumThreads = 1;
//      for(size_t K = 0; K < NumDim; K++){
//        unsigned PC = MDRead(I->getMetadata("nvvm.param.layout.p1.d" + itostr(K)));
//        NumThreads *= PC;
//      }
//      bool Constraint = NumThreads==32;
//      Constant *Cst = Constraint?ConstantInt::getTrue(Ctx):ConstantInt::getFalse(Ctx);
//      I->setMetadata("nvvm.constraint.threads", MDNode::get(Ctx, ConstantAsMetadata::get(Cst)));
//    }
//    // The number of warps required by the layout is the same
//    // for all tiles in the function
//    {
//      int NumWarps = 1;
//      int RefNumWarps = 1;
//      for(size_t K = 0; K < NumDim; K++){
//        unsigned PC = MDRead(I->getMetadata("nvvm.param.layout.p2.d" + itostr(K)));
//        unsigned PR = MDRead(FirstTile->getMetadata("nvvm.param.layout.p2.d" + itostr(K)));
//        NumWarps *= PC;
//        RefNumWarps *= PR;
//      }
//      bool Constraint = NumWarps==RefNumWarps;
//      Constant *Cst = Constraint?ConstantInt::getTrue(Ctx):ConstantInt::getFalse(Ctx);
//      I->setMetadata("nvvm.constraint.warps", MDNode::get(Ctx, ConstantAsMetadata::get(Cst)));
//    };
//  }
//  return true;
//}


//// Layout binding pass
//class TLVMAddTunerParams: public FunctionPass {
//private:
//  enum CType{
//    Layout = 0, Shape = 1
//  };
//  // Params pool
//  SmallVector<MDNode*, 8> LParamsPool;
//  // Constraints
//  typedef std::pair<Value*, unsigned> CNodeType;
//  typedef DenseMap<CNodeType, DenseSet<CNodeType>> CGraphType;
//  // Layout constraints
//  CGraphType LCGraph;
//  DenseSet<CNodeType> LCNodes;
//  // Shape constraints
//  CGraphType SCGraph;
//  DenseSet<CNodeType> SCNodes;
//  // Relational
//  std::map<std::pair<Value*, std::string>, std::function<unsigned* ()>> ExtraParams;
//  DenseSet<unsigned> Constants;

//  void addConstraint(CNodeType X, CNodeType Y, CType CT);
//  void initCPhi(Instruction *I);
//  void initCGraph(Instruction *V);
//  void connectedComponents(CNodeType X, ArrayRef<MDNode *> Vals, CType CT, DenseSet<CNodeType> &Nodes, CGraphType &Graph);

//public:
//  static char ID;
//  TLVMAddTunerParams(): FunctionPass(ID){ }

//  void getAnalysisUsage(AnalysisUsage & AU) const override;
//  bool runOnFunction(Function &F) override;

//private:
//  std::map<std::pair<Instruction*, std::string>, Constant*> KnownParams;
//};

//// Initialization
//char TLVMAddTunerParams::ID = 0;
//INITIALIZE_PASS_BEGIN(TLVMAddTunerParams, "tlvm-add-tuner-parameters",
//                "Add Tuner Parameters (TLVM)", false, true)
//INITIALIZE_PASS_END(TLVMAddTunerParams, "tlvm-add-tuner-parameters",
//                    "Add Tuner Parameters (TLVM)", false, true)
//FunctionPass *llvm::createTLVMAddTunerParamsPass() { return new TLVMAddTunerParams(); }

//// Analysis usage
//void TLVMAddTunerParams::getAnalysisUsage(AnalysisUsage &AU) const {
//  AU.setPreservesAll();
//  FunctionPass::getAnalysisUsage(AU);
//}

//void TLVMAddTunerParams::addConstraint(CNodeType X, CNodeType Y, CType CT){
//  // Layout Constraint
//  if(CT == Layout){
//    LCGraph[X].insert(Y);
//    LCGraph[Y].insert(X);
//    LCNodes.insert(X);
//    LCNodes.insert(Y);
//  }
//  if(CT == Shape || CT == Layout){
//    SCGraph[X].insert(Y);
//    SCGraph[Y].insert(X);
//    SCNodes.insert(X);
//    SCNodes.insert(Y);
//  }
//}

//void TLVMAddTunerParams::initCPhi(Instruction *I){
//  unsigned NumDim = 0;
//  // Phi Nodes: all the incoming value share the result layout
//  if(PHINode *Phi = dyn_cast<PHINode>(I)){
//    Type *Ty = Phi->getType();
//    NumDim = Ty->getTileNumDimensions();
//    unsigned NumInc = Phi->getNumIncomingValues();
//    for(unsigned PI = 0; PI < NumInc; PI++){
//      Value *Inc = Phi->getIncomingValue(PI);
//      for(unsigned K = 0; K < NumDim; K++){
//        CType CT = (LCGraph.find({Inc,K}) != LCGraph.end() ||
//                    LCGraph.find({Phi,K}) != LCGraph.end())?Layout:Shape;
//        addConstraint({Phi, K}, {Inc, K}, CT);
//      }
//    }
//  }
//}

//void TLVMAddTunerParams::initCGraph(Instruction *I) {
//  unsigned NumDim = 0;
//  LLVMContext &Context = I->getContext();
//  Constant *_1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
//  // Function call
//  if(CallInst *Call = dyn_cast<CallInst>(I))
//  if(Function *Callee = Call->getCalledFunction()){
//    Intrinsic::ID IntrinsicID = Callee->getIntrinsicID();
//    switch (IntrinsicID) {
//    // Outer
//    case Intrinsic::tlvm_outer_add: LLVM_FALLTHROUGH;
//    case Intrinsic::tlvm_outer_and: {
//      addConstraint({Call, 0}, {Call->getOperand(0), 0}, Layout);
//      addConstraint({Call, 1}, {Call->getOperand(1), 0}, Layout);
//      break;
//    }
//    // Slice
//    case Intrinsic::tlvm_read_slice_x: LLVM_FALLTHROUGH;
//    case Intrinsic::tlvm_read_slice_y: {
//      addConstraint({Call, 0}, {Call->getOperand(0), 0}, Shape);
//      break;
//    }
//    // Range
//    case Intrinsic::tlvm_range: {
//      addConstraint({Call, 0}, {Call->getOperand(1), 0}, Shape);
//      break;
//    }
//    // GetTilePtr
//    case Intrinsic::tlvm_gtp_2d: NumDim++; LLVM_FALLTHROUGH;
//    case Intrinsic::tlvm_gtp_1d: NumDim++; {
//      Value *Offset = Call->getOperand(1);
//      for(unsigned K = 0; K < NumDim; K++){
//        addConstraint({Call, K}, {Offset, K}, Layout);
//      }
//      break;
//    }
//    // SlideTilePtr: Pointer shares result layout
//    case Intrinsic::tlvm_stp_2d: NumDim++; LLVM_FALLTHROUGH;
//    case Intrinsic::tlvm_stp_1d: NumDim++; {
//      for(unsigned K = 0; K < NumDim; K++){
//        addConstraint({Call, K}, {Call->getOperand(0), K}, Layout);
//        addConstraint({Call, K}, {Call->getOperand(1), K}, Layout);
//      }
//      break;
//    }
//    // Transpose
//    case Intrinsic::tlvm_transpose_2d: NumDim++; NumDim++; {
//      Value *Op = Call->getOperand(0);
//      addConstraint({Call, 0}, {Op, 1}, Shape);
//      addConstraint({Call, 1}, {Op, 0}, Shape);
//      break;
//    }
//    // Reshape
//    case Intrinsic::tlvm_reshape_2d: NumDim++; NumDim++; {
//      for(unsigned K = 0; K < NumDim; K++)
//        addConstraint({Call, K}, {Call->getOperand(1 + K), 0}, Shape);
//      break;
//    }
//    // Reshape distributed
//    case Intrinsic::tlvm_reshape_2d_1d: NumDim++; NumDim++; {
//      size_t Current = 0;
//      for(unsigned K = 0; K < NumDim; K++){
//        if(Call->getOperand(1 + K) == _1)
//          addConstraint({Call, K}, {_1, 0}, Layout);
//        else
//          addConstraint({Call, K}, {Call->getOperand(0), Current++}, Layout);
//      }
//      break;
//    }
//    // Broadcast
//    case Intrinsic::tlvm_broadcast_2d: NumDim++; LLVM_FALLTHROUGH;
//    case Intrinsic::tlvm_broadcast_1d: NumDim++; {
//      for(unsigned K = 0; K < NumDim; K++)
//        addConstraint({Call, K}, {Call->getOperand(1 + K), 0}, Shape);
//      break;
//    }
//    // Splat
//    case Intrinsic::tlvm_splat_2d: NumDim++; LLVM_FALLTHROUGH;
//    case Intrinsic::tlvm_splat_1d: NumDim++; {
//      for(unsigned K = 0; K < NumDim; K++)
//        addConstraint({Call, K}, {Call->getOperand(K), 0}, Shape);
//      break;
//    }

//    case Intrinsic::tlvm_load:{
//      NumDim = Call->getType()->getTileNumDimensions();
//      Value *Ptr = Call->getOperand(0);
//      for(unsigned K = 0; K < NumDim; K++)
//        addConstraint({Call, K}, {Ptr, K}, Layout);
//      break;
//    }

//    // Masked Load
//    case Intrinsic::tlvm_masked_load: {
//      NumDim = Call->getType()->getTileNumDimensions();
//      for(unsigned K = 0; K < NumDim; K++){
//        addConstraint({Call, K}, {Call->getOperand(0), K}, Layout);
//        addConstraint({Call, K}, {Call->getOperand(1), K}, Layout);
//      }
//      break;
//    }
//    // Masked store
//    case Intrinsic::tlvm_atomic_load_add_f32: LLVM_FALLTHROUGH;
//    case Intrinsic::tlvm_masked_store: {
//      Value *Val = Call->getOperand(0);
//      Value *Ptr = Call->getOperand(1);
//      Value *Mask = Call->getOperand(2);
//      NumDim = Val->getType()->getTileNumDimensions();
//      for(unsigned K = 0; K < NumDim; K++){
//        addConstraint({Val, K}, {Ptr, K}, Layout);
//        addConstraint({Val, K}, {Mask, K}, Layout);
//      }
//      break;
//    }
//    // Set Mask
//    case Intrinsic::tlvm_set_mask_2d: NumDim++; NumDim++; {
//      for(unsigned K = 0; K < NumDim; K++){
//        Value *Op = Call->getOperand(NumDim + K);
//        addConstraint({Call, K}, {Op, 0}, Layout);
//      }
//      break;
//    }
//    // MMA
//    // A shares first axis with C
//    // B shares last axis with C
//    case Intrinsic::tlvm_mma_nn:
//    case Intrinsic::tlvm_mma_nt:
//    case Intrinsic::tlvm_mma_tn:
//    case Intrinsic::tlvm_mma_tt:{
//      bool AT = IntrinsicID == Intrinsic::tlvm_mma_tn || IntrinsicID == Intrinsic::tlvm_mma_tt;
//      bool BT = IntrinsicID == Intrinsic::tlvm_mma_nt || IntrinsicID == Intrinsic::tlvm_mma_tt;
//      Value *A = Call->getOperand(0);
//      Value *B = Call->getOperand(1);
//      Value *D = Call->getOperand(2);
//      size_t AOuter = 0, AInner = 1;
//      size_t BOuter = 1, BInner = 0;
//      if(AT) std::swap(AOuter, AInner);
//      if(BT) std::swap(BOuter, BInner);
//      addConstraint({Call, 0}, {A, AOuter}, Shape);
//      addConstraint({Call, 1}, {B, BOuter}, Shape);
//      addConstraint({A, AInner}, {B, BInner}, Shape);
//      addConstraint({Call, 0}, {D, 0}, Layout);
//      addConstraint({Call, 1}, {D, 1}, Layout);
//      break;
//    }
//    default:
//      break;
//    }
//  }
//  // LoadInst: Pointer shares the result layout
//  if(LoadInst *Load = dyn_cast<LoadInst>(I)){
//    NumDim = Load->getType()->getTileNumDimensions();
//    Value *Ptr = Load->getPointerOperand();
//    for(unsigned K = 0; K < NumDim; K++)
//      addConstraint({Load, K}, {Ptr, K}, Layout);
//  }
//  // StoreInst: Pointer shares the value layout
//  if(StoreInst *Store = dyn_cast<StoreInst>(I)){
//    Value *Ptr = Store->getPointerOperand();
//    Value *Val = Store->getValueOperand();
//    NumDim = Val->getType()->getTileNumDimensions();
//    for(unsigned K = 0; K < NumDim; K++)
//      addConstraint({Ptr, K}, {Val, K}, Layout);
//  }
//  // SelectInst: Selected tensor share layout
//  if(SelectInst *Select = dyn_cast<SelectInst>(I)){
//    NumDim = Select->getType()->getTileNumDimensions();
//    for(unsigned K = 0; K < NumDim; K++){
//      addConstraint({Select->getTrueValue(), K}, {Select, K}, Layout);
//      addConstraint({Select->getFalseValue(), K}, {Select, K}, Layout);
//    }
//  }
//  if(isa<CastInst>(I)){
//    NumDim = I->getType()->getTileNumDimensions();
//    for(unsigned K = 0; K < NumDim; K++){
//      addConstraint({I->getOperand(0), K}, {I, K}, Layout);
//    }
//  }
//  // Phi Nodes: all the incoming value share the result layout
//  if(PHINode *Phi = dyn_cast<PHINode>(I)){
//    Type *Ty = Phi->getType();
//    NumDim = Ty->getTileNumDimensions();
//    unsigned NumInc = Phi->getNumIncomingValues();
//    for(unsigned PI = 0; PI < NumInc; PI++){
//      Value *Inc = Phi->getIncomingValue(PI);
//      for(unsigned K = 0; K < NumDim; K++){
//        CType CT = (LCGraph.find({Inc,K}) != LCGraph.end() ||
//                    LCGraph.find({Phi,K}) != LCGraph.end())?Layout:Shape;
//        addConstraint({Phi, K}, {Inc, K}, CT);
//      }
//    }
//  }
//  // Binary op: All the arguments share the result layout
//  Instruction *BinOp = static_cast<Instruction*>(I);
//  if(isa<BinaryOperator>(BinOp) || isa<CmpInst>(BinOp)){
//    NumDim = BinOp->getType()->getTileNumDimensions();
//    Value *A = BinOp->getOperand(0);
//    Value *B = BinOp->getOperand(1);
//    for(unsigned K = 0; K < NumDim; K++){
//      addConstraint({BinOp, K}, {A, K}, Layout);
//      addConstraint({BinOp, K}, {B, K}, Layout);
//    }
//  }
//}

//void TLVMAddTunerParams::connectedComponents(CNodeType X, ArrayRef<llvm::MDNode*> Vals, CType CT,
//                                                DenseSet<CNodeType> &Nodes, CGraphType &Graph){
//  if(Nodes.find(X) != Nodes.end()){
//    Nodes.erase(X);
//    std::string Suffix = ".d" + itostr(X.second);
//    if(Instruction *Instr = dyn_cast<Instruction>(X.first)){
//      if(CT==Shape){
//        Instr->setMetadata("nvvm.param.shape" + Suffix, Vals[0]);
//      }
//      if(CT==Layout){
//        Instr->setMetadata("nvvm.param.layout.p0" + Suffix, Vals[0]);
//        Instr->setMetadata("nvvm.param.layout.p1" + Suffix, Vals[1]);
//        Instr->setMetadata("nvvm.param.layout.p2" + Suffix, Vals[2]);
//      }
//    }
//    if(ConstantInt *Cst = dyn_cast<ConstantInt>(X.first)){
//      Metadata *CstMD = ConstantAsMetadata::get(Cst);
//      if(CT==Shape){
//        Vals[0]->replaceOperandWith(0, CstMD);
//      }
//      if(CT==Layout){
//        Vals[0]->replaceOperandWith(0, CstMD);
//        Vals[1]->replaceOperandWith(0, CstMD);
//        Vals[2]->replaceOperandWith(0, CstMD);
//      }
//    }
//    for(CNodeType &E: Graph[X])
//      connectedComponents(E, Vals, CT, Nodes, Graph);
//  }
//}

//// Run
//bool TLVMAddTunerParams::runOnFunction(Function &F) {
//  // Build constraints graph
//  for(Function::iterator::value_type &BB: F)
//  for(BasicBlock::iterator::value_type &I : BB)
//  if(isTLVMValue(&I))
//    initCGraph(&I);
//  for(Function::iterator::value_type &BB: F)
//  for(BasicBlock::iterator::value_type &I : BB)
//  if(isTLVMValue(&I))
//    initCPhi(&I);
//  // Add parameters
//  LLVMContext &Ctx = F.getContext();
//  Metadata *UndefMD = ConstantAsMetadata::get(UndefValue::get(Type::getInt32Ty(Ctx)));
//  // Shape parameters
//  while(!SCNodes.empty()){
//    MDNode *V0 = MDNode::getTemporary(Ctx, UndefMD).release();
//    connectedComponents(*SCNodes.begin(), {V0}, Shape, SCNodes, SCGraph);
//  }
//  // Layout parameters
//  while(!LCNodes.empty()){
//    MDNode *V0 = MDNode::getTemporary(Ctx, UndefMD).release();
//    MDNode *V1 = MDNode::getTemporary(Ctx, UndefMD).release();
//    MDNode *V2 = MDNode::getTemporary(Ctx, UndefMD).release();
//    connectedComponents(*LCNodes.begin(), {V0, V1, V2}, Layout, LCNodes, LCGraph);
//  }
//  return true;
//}

//}
//}
