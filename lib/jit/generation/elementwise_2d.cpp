/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <cstring>
#include <iostream>
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/syntax/engine/process.h"
#include "tools/arguments.hpp"
#include "tools/loop.hpp"
#include "tools/vector_types.hpp"

namespace isaac
{
namespace templates
{

int elementwise_2d::is_invalid_impl(driver::Device const &, expression_tree const  &) const
{
  if (vwidth_>1)
    return TEMPLATE_INVALID_SIMD_WIDTH;
  return TEMPLATE_VALID;
}

expression_type elementwise_2d::type() const
{ return ELEMENTWISE_2D; }

std::string elementwise_2d::generate_impl(std::string const & suffix, expression_tree const  & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  std::vector<std::size_t> assigned = symbolic::find(tree, [&](expression_tree::node const & node){return node.type==COMPOSITE_OPERATOR_TYPE && is_assignment(node.binary_operator.op.type);});
  std::vector<std::size_t> assigned_left;
  std::vector<std::size_t> assigned_right;
  for(std::size_t idx: assigned){
    assigned_left.push_back(tree[idx].binary_operator.lhs);
    assigned_right.push_back(tree[idx].binary_operator.rhs);
  }
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"vector.h\"" << std::endl; break;
    case driver::OPENCL:
      if(tree.dtype()==HALF_TYPE)
        stream << "#pragma OPENCL EXTENSION cl_khr_fp16: enable" << std::endl;
      stream << " __attribute__((reqd_work_group_size(" << ls0_ << "," << ls1_ << ",1)))" << std::endl; break;
  }

  stream << "$KERNEL void elementwise_2d" << suffix << "($SIZE_T M, $SIZE_T N, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << tools::join(negative_inc_process(device, symbols, tree), "  ") << std::endl;
  element_wise_loop_1D(stream, 1, "i", "M", "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", [&](unsigned int){
    element_wise_loop_1D(stream, 1, "j", "N", "$GLOBAL_IDX_1", "$GLOBAL_SIZE_1", [&](unsigned int){
      //Declares register to store results
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_left, false))
        stream << sym->process("#scalartype #name;") << std::endl;

      //Load to registers
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_right, false))
        stream << sym->process("#scalartype #name = at(i, j);") << std::endl;

      for(std::size_t idx: assigned)
        stream << symbols.at(idx)->evaluate({{"leaf", "#name"}}) << ";" << std::endl;

      //Writes back
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_left, false))
        stream << sym->process("at(i, j) = #name;") << std::endl;
    });
  });


  stream.dec_tab();
  stream << "}" << std::endl;

  return stream.str();
}

elementwise_2d::elementwise_2d(unsigned int vwidth, unsigned int ls0, unsigned int ls1,
                               unsigned int ng0, unsigned int ng1):
    parameterized_base(vwidth, ls0, ls1), ng0_(ng0), ng1_(ng1)
{}

std::vector<int_t> elementwise_2d::input_sizes(expression_tree const  & expression) const{
  return expression.shape();
}

void elementwise_2d::enqueue(driver::CommandQueue & /*queue*/, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  expression_tree const  & expressions = control.x();
  std::string name = "elementwise_2d";
  name +=suffix;
  driver::Kernel kernel(program, name.c_str());
  driver::NDRange global(ls0_*ng0_, ls1_*ng1_);
  driver::NDRange local(ls0_, ls1_);
  unsigned int current_arg = 0;
  std::vector<int_t> MN = input_sizes(expressions);
  kernel.setSizeArg(current_arg++, MN[0]);
  kernel.setSizeArg(current_arg++, MN[1]);
  symbolic::set_arguments(expressions, kernel, current_arg);

  control.execution_options().enqueue(program.context(), kernel, global, local);
}

}
}
