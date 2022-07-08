#pragma once

#ifndef _TRITON_IR_CFG_H_
#define _TRITON_IR_CFG_H_

#include <vector>
#include <functional>

namespace triton{
namespace ir{

class module;
class function;
class basic_block;
class instruction;
class value;

class cfg {
public:
  static std::vector<basic_block *> post_order(function* fn);
  static std::vector<basic_block *> reverse_post_order(function* fn);
};

void for_each_instruction(ir::module& mod, const std::function<void(triton::ir::instruction*)> &fn);
void for_each_instruction_backward(module &mod, const std::function<void (instruction *)> &do_work);
void for_each_value(ir::module& mod, const std::function<void(triton::ir::value *)> &fn);

}
}

#endif
