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

#ifndef _ISAAC_SYMBOLIC_EXECUTE_H
#define _ISAAC_SYMBOLIC_EXECUTE_H

#include "isaac/runtime/profiles.h"
#include "isaac/runtime/execute.h"

namespace isaac
{
namespace runtime
{

namespace detail
{
  typedef std::vector<std::pair<size_t, expression_type> > breakpoints_t;
  ISAACWINAPI expression_type parse(expression_tree const & tree, breakpoints_t & bp);
  ISAACWINAPI expression_type parse(expression_tree const & tree, size_t idx, breakpoints_t & bp);
}

/** @brief Executes a expression_tree on the given queue for the given models map*/
void execute(execution_handler const & , profiles::map_type &);

/** @brief Executes a expression_tree on the default models map*/
void execute(execution_handler const &);

}

}

#endif
