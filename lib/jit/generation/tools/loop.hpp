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

inline void fetching_loop_info(fetching_policy_type policy, std::string const & bound, kernel_generation_stream & stream, std::string & init, std::string & upper_bound, std::string & inc, std::string const & domain_id, std::string const & domain_size, driver::Device const &)
{
  if (policy==FETCH_FROM_GLOBAL_STRIDED)
  {
    init = domain_id;
    upper_bound = bound;
    inc = domain_size;
  }
  else if (policy==FETCH_FROM_GLOBAL_CONTIGUOUS)
  {
    std::string chunk_size = "chunk_size";
    std::string chunk_start = "chunk_start";
    std::string chunk_end = "chunk_end";

    stream << "$SIZE_T " << chunk_size << " = (" << bound << "+" << domain_size << "-1)/" << domain_size << ";" << std::endl;
    stream << "$SIZE_T " << chunk_start << " =" << domain_id << "*" << chunk_size << ";" << std::endl;
    stream << "$SIZE_T " << chunk_end << " = min(" << chunk_start << "+" << chunk_size << ", " << bound << ");" << std::endl;
    init = chunk_start;
    upper_bound = chunk_end;
    inc = "1";
  }
}


template<class Fun>
inline void element_wise_loop_1D(kernel_generation_stream & stream, fetching_policy_type fetch, unsigned int simd_width,
                                 std::string const & i, std::string const & bound, std::string const & domain_id, std::string const & domain_size, driver::Device const & device, Fun const & generate_body)
{
  std::string strwidth = tools::to_string(simd_width);

  std::string init, upper_bound, inc;
  fetching_loop_info(fetch, bound, stream, init, upper_bound, inc, domain_id, domain_size, device);
  std::string boundround = upper_bound + "/" + strwidth + "*" + strwidth;
  stream << "for(unsigned int " << i << " = " << init << "*" << strwidth << "; " << i << " < " << boundround << "; " << i << " += " << inc << "*" << strwidth << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  generate_body(simd_width);
  stream.dec_tab();
  stream << "}" << std::endl;

  if (simd_width>1)
  {
    stream << "for(unsigned int " << i << " = " << boundround << " + " << domain_id << "; " << i << " < " << bound << "; " << i << " += " + domain_size + ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    generate_body(1);
    stream.dec_tab();
    stream << "}" << std::endl;
  }
}

}
}
