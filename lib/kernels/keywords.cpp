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
#include "isaac/kernels/keywords.h"

namespace isaac
{

keyword::keyword(driver::backend_type backend, std::string const & opencl, std::string const & cuda) : backend_(backend), opencl_(opencl), cuda_(cuda)
{

}

std::string const & keyword::get() const
{
  switch(backend_)
  {
    case driver::OPENCL:
      return opencl_;
    case driver::CUDA:
      return cuda_;
    default: throw;
  }
}

std::ostream &  operator<<(std::ostream & ss, keyword const & kw)
{
  return ss << kw.get();
}


}
