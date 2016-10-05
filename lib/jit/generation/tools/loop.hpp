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

#include "isaac/jit/generation/engine/stream.h"
#include "isaac/jit/generation/base.h"
#include <string>
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace templates
{

template<class Fun>
inline void element_wise_loop_1D(kernel_generation_stream & stream, unsigned int vwidth,
                                 std::string const & i, std::string const & bound, std::string const & domain_id, std::string const & domain_size, Fun const & generate_body)
{
  std::string svwidth = tools::to_string(vwidth);
  std::string init = domain_id + "*" + svwidth;
  std::string lbound = bound + "/" + svwidth + "*" + svwidth;
  std::string inc = domain_size + "*" + svwidth;
  stream << "for(unsigned int " << i << " = " << init << "; " << i << " < " << lbound << "; " << i << " += " << inc << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  generate_body(vwidth);
  stream.dec_tab();
  stream << "}" << std::endl;

  if (vwidth>1)
  {
    stream << "for(unsigned int " << i << " = " << lbound << " + " << domain_id << "; " << i << " < " << bound << "; " << i << " += " + domain_size + ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    generate_body(1);
    stream.dec_tab();
    stream << "}" << std::endl;
  }
}

}
}
