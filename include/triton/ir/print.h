#pragma once

#ifndef _TRITON_IR_PRINT_H_
#define _TRITON_IR_PRINT_H_

#include "builder.h"

namespace triton{
namespace ir{

class module;

void print(module &mod, std::ostream& os);

}
}

#endif
