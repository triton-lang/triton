#include "codegen/tune.h"
#include "ir/instructions.h"
#include "ir/type.h"
#include "ir/module.h"
#include "ir/function.h"
#include <cstdlib>

namespace tdl{
namespace codegen{

void tune::add_constraint(node_t x, node_t y) {
  dependencies_[x].insert(y);
  dependencies_[y].insert(x);
  nodes_.insert(x);
  nodes_.insert(y);
}

void tune::init_c_phi(ir::instruction *v) {
  // Phi Nodes: all the incoming value share the result layout
  if(auto *phi = dynamic_cast<ir::phi_node*>(v))
    for(ir::value *inc: phi->ops())
      for(unsigned k = 0; k < phi->get_type()->get_tile_shapes().size(); k++)
        if(dependencies_.find({inc, k}) != dependencies_.end()
           || dependencies_.find({phi, k}) != dependencies_.end())
          add_constraint({phi, k}, {inc, k});
}

void tune::init_c_graph(ir::instruction *v) {
  unsigned num_dim = v->get_type()->get_tile_shapes().size();
  if(dynamic_cast<ir::reshape_inst*>(v)){

  }
  else if(dynamic_cast<ir::splat_inst*>(v)){

  }
  else if(dynamic_cast<ir::broadcast_inst*>(v)){

  }
  else if(auto *ii = dynamic_cast<ir::matmul_inst*>(v)){
    ir::value *D = ii->get_operand(2);
    add_constraint({v, 0}, {D, 0});
    add_constraint({v, 1}, {D, 1});
  }
  else if(dynamic_cast<ir::user*>(v))
    for(unsigned i = 0; i < num_dim; i ++)
      for(ir::value* op: v->ops())
        add_constraint({v, i}, {op, i});
}

void tune::connected_components(node_t x, const std::vector<unsigned *> vals, std::set<node_t> &nodes, graph_t &graph) {
  if(nodes.find(x) != nodes.end()){
    nodes.erase(x);
    std::string suffix = ".d" + std::to_string(x.second);
    if(auto *instr = dynamic_cast<ir::instruction*>(x.first)){
      params_[instr].insert({"p0" + suffix, vals[0]});
      params_[instr].insert({"p1" + suffix, vals[1]});
      params_[instr].insert({"p2" + suffix, vals[2]});
    }
    for(const node_t &y: graph[x])
      connected_components(y, vals, nodes, graph);
  }
}

void tune::run(ir::module &mod) {
  for(ir::function *fn: mod.get_function_list()){
    // Build constraints graph
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->get_type()->is_tile_ty())
      init_c_graph(i);
    // Build phi constraints
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->get_type()->is_tile_ty())
      init_c_phi(i);
    // Layout parameters
    while(!nodes_.empty()){
      unsigned *v0 = new unsigned(0);
      unsigned *v1 = new unsigned(0);
      unsigned *v2 = new unsigned(0);
      connected_components(*nodes_.begin(), {v0, v1, v2}, nodes_, dependencies_);
    }
  }
}

bool tune::check_constraints(std::map<ir::value *, std::string> &errors) {

  return true;
}

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


}
}
