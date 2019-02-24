#ifndef TDL_INCLUDE_IR_PRINT_H
#define TDL_INCLUDE_IR_PRINT_H


#include "builder.h"

namespace tdl{
namespace ir{

class module;

void print(module &mod, std::ostream& os);

}
}

#endif
