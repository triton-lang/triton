#ifndef _TRITON_IR_PRINT_H_
#define _TRITON_IR_PRINT_H_

#include "builder.h"

namespace triton{
namespace ir{

class module;
class function;
class basic_block;
class instruction;

void print(module &mod, std::ostream& os);
void print(function &func, std::ostream& os);
void print(basic_block &bb, std::ostream& os);
void print(instruction &instr, std::ostream& os);

}
}

#endif
