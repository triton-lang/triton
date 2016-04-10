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

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "isaac/jit/syntax/engine/object.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/array.h"

namespace isaac
{
namespace templates
{

//Generate
inline std::vector<std::string> kernel_arguments(driver::Device const &, symbolic::symbols_table const & symbols, expression_tree const & expressions)
{
    std::vector<std::string> result;
    for(symbolic::object* obj: symbolic::extract<symbolic::object>(expressions, symbols))
    {
      if(symbolic::host_scalar* sym = dynamic_cast<symbolic::host_scalar*>(obj))
        result.push_back(sym->process("#scalartype #name_value"));
      if(symbolic::buffer* sym = dynamic_cast<symbolic::buffer*>(obj))
      {
        result.push_back("$GLOBAL " + sym->process("#scalartype* #pointer"));
        if(sym->hasattr("off")) result.push_back("$SIZE_T " + sym->process("#off"));
        if(sym->hasattr("inc0")) result.push_back("$SIZE_T " + sym->process("#inc0"));
        if(sym->hasattr("inc1")) result.push_back("$SIZE_T " + sym->process("#inc1"));
      }
      if(symbolic::reshape* sym = dynamic_cast<symbolic::reshape*>(obj))
      {
        if(sym->hasattr("new_inc1")) result.push_back("$SIZE_T " + sym->process("#new_inc1"));
        if(sym->hasattr("old_inc1")) result.push_back("$SIZE_T " + sym->process("#old_inc1"));
      }
    }
    return result;
}


}
}
