#include "ir/basic_block.h"
#include "ir/instructions.h"

namespace tdl{
namespace ir{


instruction::instruction(type *ty, unsigned num_ops, instruction *next)
    : user(ty, num_ops) {
  if(next){
    basic_block *block = next->get_parent();
    assert(block && "Next instruction is not in a basic block!");
  }
}

//  // If requested, insert this instruction into a basic block...
//  if (InsertBefore) {
//    BasicBlock *BB = InsertBefore->getParent();
//    assert(BB && "Instruction to insert before is not in a basic block!");
//    BB->getInstList().insert(InsertBefore->getIterator(), this);
//  }



}
}
