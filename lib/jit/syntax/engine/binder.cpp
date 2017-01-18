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

#include "isaac/jit/syntax/engine/binder.h"

namespace isaac
{

//Base
symbolic_binder::~symbolic_binder()
{}

symbolic_binder::symbolic_binder(driver::backend_type backend) : current_arg_(0), memory(backend)
{}

unsigned int symbolic_binder::get()
{ return current_arg_++; }

//Sequential
bind_sequential::bind_sequential(driver::backend_type backend) : symbolic_binder(backend)
{ }

bool bind_sequential::bind(handle_t const & h, bool)
{ return memory.insert(std::make_pair(h, current_arg_)).second; }

unsigned int bind_sequential::get(handle_t const & h, bool is_assigned)
{ return bind(h, is_assigned)?current_arg_++:memory.at(h); }

//Independent
bind_independent::bind_independent(driver::backend_type backend) : symbolic_binder(backend)
{ }

bool bind_independent::bind(handle_t const & h, bool is_assigned)
{ return is_assigned?true:memory.insert(std::make_pair(h, current_arg_)).second; }

unsigned int bind_independent::get(handle_t const & h, bool is_assigned)
{ return bind(h, is_assigned)?current_arg_++:memory.at(h); }

}
