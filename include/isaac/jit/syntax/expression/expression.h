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

#ifndef _ISAAC_SYMBOLIC_EXPRESSION_H
#define _ISAAC_SYMBOLIC_EXPRESSION_H

#include <utility>
#include <vector>
#include <list>
#include "isaac/driver/backend.h"
#include "isaac/driver/context.h"
#include "isaac/driver/command_queue.h"
#include "isaac/driver/event.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/driver/buffer.h"

#include "isaac/jit/syntax/expression/operations.h"
#include "isaac/tools/cpp/tuple.hpp"

#include "isaac/types.h"
#include "isaac/value_scalar.h"
#include <memory>
#include <iostream>

namespace isaac
{

class array_base;

struct invalid_node{};

enum node_type
{
  INVALID_SUBTYPE = 0,
  COMPOSITE_OPERATOR_TYPE,
  VALUE_SCALAR_TYPE,
  DENSE_ARRAY_TYPE,
};

union handle_t
{
  cl_mem cl;
  CUdeviceptr cu;
};


struct array_holder
{
  int_t start;
  handle_t handle;
  array_base* base;
};

class expression_tree
{
public:
  struct node
  {
    //Constructors
    node();
    node(invalid_node);
    node(value_scalar const & x);
    node(array_base const & x);
    node(int_t lhs, op_element op, int_t rhs, numeric_type dtype, tuple const & shape);

    //Common
    node_type type;
    numeric_type dtype;
    tuple shape;
    tuple ld;

    //Type-specific
    union
    {
      //Operator
      struct{
        int_t lhs;
        op_element op;
        int_t rhs;
      }binary_operator;
      //Scalar
      values_holder scalar;
      //Array
      array_holder array;
    };
  };

  typedef std::vector<node>     data_type;

public:
  expression_tree(node const & lhs, node const & rhs, op_element const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression_tree(expression_tree const & lhs, node const & rhs, op_element const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression_tree(node const & lhs, expression_tree const & rhs, op_element const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression_tree(expression_tree const & lhs, expression_tree const & rhs, op_element const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);

  tuple shape() const;
  int_t dim() const;
  data_type const & data() const;
  std::size_t root() const;
  driver::Context const & context() const;
  numeric_type const & dtype() const;

  node const & operator[](size_t) const;
  node & operator[](size_t);

  expression_tree operator-();
  expression_tree operator!();

private:
  data_type tree_;
  std::size_t root_;
  driver::Context const * context_;
};

template<class T> typename std::enable_if<!std::is_arithmetic<T>::value, T const &>::type wrap_generic(T const & x){ return x;}
template<class T> typename std::enable_if<std::is_arithmetic<T>::value, value_scalar>::type wrap_generic(T x) { return value_scalar(x); }

//Visual Studio doesn't support use call convention and template together, therefore, the ISAACUNWINAPI is defined to fix this problem.
template<typename T>
ISAACNOTWINAPI typename std::conditional<std::is_arithmetic<T>::value, value_scalar, T const &>::type make_tuple(driver::Context const &, T const & x)
{ return wrap_generic(x); }

template<typename T, typename... Args>
ISAACNOTWINAPI expression_tree make_tuple(driver::Context const & context, T const & x, Args... args)
{ return expression_tree(wrap_generic(x), make_tuple(context, args...), op_element(BINARY_ARITHMETIC, PAIR_TYPE), &context, numeric_type_of(x), {1}); }

//io
std::string to_string(node_type const & f);
std::string to_string(expression_tree::node const & e);
std::ostream & operator<<(std::ostream & os, expression_tree::node const & s_node);
std::string to_string(isaac::expression_tree const & s);

}

#endif
