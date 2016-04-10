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

#ifndef _ISAAC_SYMBOLIC_HANDLER_H
#define _ISAAC_SYMBOLIC_HANDLER_H

#include "isaac/jit/syntax/expression/expression.h"

namespace isaac
{

namespace runtime
{

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
  execution_handler(expression_tree const & x, runtime::execution_handler const & other) : x_(x), execution_options_(other.execution_options_), dispatcher_options_(other.dispatcher_options_), compilation_options_(other.compilation_options_){}
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

}
}

#endif
