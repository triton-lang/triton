/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
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

#include "isaac/symbolic/expression/operations.h"
#include "isaac/tools/cpp/tuple.hpp"

#include "isaac/types.h"
#include "isaac/value_scalar.h"
#include <memory>
#include <iostream>

namespace isaac
{

class array_base;

struct placeholder
{
  expression_tree operator=(value_scalar const & ) const;
  expression_tree operator=(expression_tree const & ) const;

  expression_tree operator+=(value_scalar const & ) const;
  expression_tree operator-=(value_scalar const & ) const;
  expression_tree operator*=(value_scalar const & ) const;
  expression_tree operator/=(value_scalar const & ) const;

  int level;
};

struct invalid_node{};

enum node_type
{
  INVALID_SUBTYPE = 0,
  COMPOSITE_OPERATOR_TYPE,
  VALUE_SCALAR_TYPE,
  DENSE_ARRAY_TYPE,
  PLACEHOLDER_TYPE
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
};

class expression_tree
{
public:
  struct node
  {
    //Constructors
    node();
    node(invalid_node);
    node(placeholder x);
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
      //Binary Operator
      struct{
        int_t lhs;
        op_element op;
        int_t rhs;
      }binary_operator;
      //Scalar
      values_holder scalar;
      //Array
      array_holder array;
      //Placeholder
      placeholder ph;
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



struct execution_options_type
{
  execution_options_type(unsigned int _queue_id = 0, std::list<driver::Event>* _events = NULL, std::vector<driver::Event>* _dependencies = NULL) :
     events(_events), dependencies(_dependencies), queue_id_(_queue_id)
  {}

  execution_options_type(driver::CommandQueue const & queue, std::list<driver::Event> *_events = NULL, std::vector<driver::Event> *_dependencies = NULL) :
      events(_events), dependencies(_dependencies), queue_id_(-1), queue_(new driver::CommandQueue(queue))
  {}

  void enqueue(driver::Context const & context, driver::Kernel const & kernel, driver::NDRange global, driver::NDRange local) const
  {
    driver::CommandQueue & q = queue(context);
    if(events)
    {
      driver::Event event(q.backend());
      q.enqueue(kernel, global, local, dependencies, &event);
      events->push_back(event);
    }
    else
      q.enqueue(kernel, global, local, dependencies, NULL);
  }

  driver::CommandQueue & queue(driver::Context const & context) const
  {
    if(queue_)
        return *queue_;
    return driver::backend::queues::get(context, queue_id_);
  }

  std::list<driver::Event>* events;
  std::vector<driver::Event>* dependencies;

private:
  int queue_id_;
  std::shared_ptr<driver::CommandQueue> queue_;
};

struct dispatcher_options_type
{
  dispatcher_options_type(bool _tune = false, int _label = -1) : tune(_tune), label(_label){}
  bool tune;
  int label;
};

struct compilation_options_type
{
  compilation_options_type(std::string const & _program_name = "", bool _recompile = false) : program_name(_program_name), recompile(_recompile){}
  std::string program_name;
  bool recompile;
};

class execution_handler
{
public:
  execution_handler(expression_tree const & x, execution_options_type const& execution_options = execution_options_type(),
             dispatcher_options_type const & dispatcher_options = dispatcher_options_type(),
             compilation_options_type const & compilation_options = compilation_options_type())
                : x_(x), execution_options_(execution_options), dispatcher_options_(dispatcher_options), compilation_options_(compilation_options){}
  execution_handler(expression_tree const & x, execution_handler const & other) : x_(x), execution_options_(other.execution_options_), dispatcher_options_(other.dispatcher_options_), compilation_options_(other.compilation_options_){}
  expression_tree const & x() const { return x_; }
  execution_options_type const & execution_options() const { return execution_options_; }
  dispatcher_options_type const & dispatcher_options() const { return dispatcher_options_; }
  compilation_options_type const & compilation_options() const { return compilation_options_; }
private:
  expression_tree x_;
  execution_options_type execution_options_;
  dispatcher_options_type dispatcher_options_;
  compilation_options_type compilation_options_;
};

//

template<class T> typename std::enable_if<!std::is_arithmetic<T>::value, T const &>::type wrap_generic(T const & x){ return x;}
template<class T> typename std::enable_if<std::is_arithmetic<T>::value, value_scalar>::type wrap_generic(T x) { return value_scalar(x); }

template<typename T>
ISAACAPI typename std::conditional<std::is_arithmetic<T>::value, value_scalar, T const &>::type make_tuple(driver::Context const &, T const & x)
{ return wrap_generic(x); }

template<typename T, typename... Args>
ISAACAPI expression_tree make_tuple(driver::Context const & context, T const & x, Args... args)
{ return expression_tree(wrap_generic(x), make_tuple(context, args...), op_element(BINARY_ARITHMETIC, PAIR_TYPE), &context, numeric_type_of(x), {1}); }

}

#endif
