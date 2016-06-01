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

#ifndef ISAAC_DRIVER_PROGRAM_H
#define ISAAC_DRIVER_PROGRAM_H

#include <map>

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/handle.h"
#include "isaac/driver/context.h"
namespace isaac
{

namespace driver
{

class Context;
class Device;

class ISAACAPI Program: public has_handle_comparators<Program>
{
public:
  typedef Handle<cl_program, CUmodule> handle_type;

private:
  friend class Kernel;

public:
  //Constructors
  Program(Context const & context, std::string const & source);
  //Accessors
  handle_type const & handle() const;
  Context const & context() const;

private:
DISABLE_MSVC_WARNING_C4251
  backend_type backend_;
  Context context_;
  std::string source_;
  handle_type h_;
RESTORE_MSVC_WARNING_C4251
};


}

}

#endif
