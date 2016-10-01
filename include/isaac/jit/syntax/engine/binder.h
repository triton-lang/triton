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
#include "isaac/jit/syntax/expression/expression.h"

namespace isaac
{

class array_base;


class symbolic_binder
{
  class cmp
  {
  public:
    cmp(driver::backend_type backend) : backend_(backend) {}

    bool operator()(handle_t const & x, handle_t const & y) const
    {
      if(backend_==driver::OPENCL)
        return x.cl < y.cl;
      else
        return x.cu < y.cu;
    }

  private:
    driver::backend_type backend_;
  };

public:
  symbolic_binder(driver::backend_type backend);
  virtual ~symbolic_binder();
  virtual bool bind(handle_t const &, bool) = 0;
  virtual unsigned int get(handle_t const &, bool) = 0;
  unsigned int get();
protected:
  unsigned int current_arg_;
  std::map<handle_t,unsigned int, cmp> memory;
};


class bind_sequential : public symbolic_binder
{
public:
  bind_sequential(driver::backend_type backend);
  bool bind(handle_t const & a, bool);
  unsigned int get(handle_t const & a, bool);
};

class bind_independent : public symbolic_binder
{
public:
  bind_independent(driver::backend_type backend);
  bool bind(handle_t const & a, bool);
  unsigned int get(const handle_t &a, bool);
};

}

#endif
