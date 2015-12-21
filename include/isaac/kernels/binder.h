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

#ifndef ISAAC_BACKEND_BINDER_H
#define ISAAC_BACKEND_BINDER_H

#include <map>
#include "isaac/driver/buffer.h"

namespace isaac
{

enum binding_policy_t
{
  BIND_INDEPENDENT,
  BIND_SEQUENTIAL
};

class array_base;

class symbolic_binder
{
public:
  symbolic_binder();
  virtual ~symbolic_binder();
  virtual bool bind(array_base const * a, bool) = 0;
  virtual unsigned int get(array_base const * a, bool) = 0;
  unsigned int get();
protected:
  unsigned int current_arg_;
  std::map<array_base const *,unsigned int> memory;
};


class bind_sequential : public symbolic_binder
{
public:
  bind_sequential();
  bool bind(array_base const * a, bool);
  unsigned int get(array_base const * a, bool);
};

class bind_independent : public symbolic_binder
{
public:
  bind_independent();
  bool bind(array_base const * a, bool);
  unsigned int get(array_base const * a, bool);
};

}

#endif
