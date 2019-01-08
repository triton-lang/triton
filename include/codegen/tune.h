//#ifndef TDL_INCLUDE_IR_CODEGEN_TUNE_H
//#define TDL_INCLUDE_IR_CODEGEN_TUNE_H

//namespace tdl{
//namespace codegen{

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

//class TLVMAddTunerConstraints: public FunctionPass {
//public:
//  static char ID;
//  TLVMAddTunerConstraints(): FunctionPass(ID){ }

//  void getAnalysisUsage(AnalysisUsage & AU) const override;
//  bool runOnFunction(Function &F) override;
//};

//}
//}

//#endif
