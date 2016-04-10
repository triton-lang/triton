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
