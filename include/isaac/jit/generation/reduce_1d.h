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

#ifndef ISAAC_BACKEND_TEMPLATES_DOT_H
#define ISAAC_BACKEND_TEMPLATES_DOT_H

#include "isaac/jit/generation/base.h"

namespace isaac
{
namespace templates
{

class reduce_1d : public parameterized_base
{
private:
  unsigned int lmem_usage(expression_tree const  & expressions) const;
  unsigned int temporary_workspace(expression_tree const & expressions) const;
  inline void reduce_1d_local_memory(kernel_generation_stream & stream, unsigned int size, std::vector<symbolic::reduce_1d*> exprs,
                                     std::string const & buf_str, std::string const & buf_value_str, driver::backend_type backend) const;
  std::string generate_impl(std::string const & suffix,  expression_tree const  & expressions, driver::Device const & device, symbolic::symbols_table const & mapping) const;

public:
  reduce_1d(unsigned int vwidth, unsigned int ls, unsigned int ng);
  std::vector<int_t> input_sizes(expression_tree const  & expressions) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const &);
  expression_type type() const;

private:
  unsigned int ng_;
  std::vector< driver::Buffer > tmp_;
  std::vector< driver::Buffer > tmpidx_;
};

}
}

#endif
