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
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

kernel_generation_stream::kgenstream::kgenstream(std::ostringstream& oss,unsigned int const & tab_count) :
  oss_(oss), tab_count_(tab_count)
{ }


int kernel_generation_stream::kgenstream::sync()
{
  for (unsigned int i=0; i<tab_count_;++i)
    oss_ << "  ";
  std::string next = str();
  oss_ << next;
  str("");
  return !oss_;
}

kernel_generation_stream::kgenstream:: ~kgenstream()
{  pubsync(); }

void kernel_generation_stream::process(std::string& str)
{

#define ADD_KEYWORD(NAME, OPENCL_NAME, CUDA_NAME) tools::find_and_replace(str, "$" + std::string(NAME), (backend_==driver::CUDA)?CUDA_NAME:OPENCL_NAME);


ADD_KEYWORD("GLOBAL_IDX_0", "get_global_id(0)", "(blockIdx.x*blockDim.x + threadIdx.x)")
ADD_KEYWORD("GLOBAL_IDX_1", "get_global_id(1)", "(blockIdx.y*blockDim.y + threadIdx.y)")
ADD_KEYWORD("GLOBAL_IDX_2", "get_global_id(2)", "(blockIdx.z*blockDim.z + threadIdx.z)")

ADD_KEYWORD("GLOBAL_SIZE_0", "get_global_size(0)", "(blockDim.x*gridDim.x)")
ADD_KEYWORD("GLOBAL_SIZE_1", "get_global_size(1)", "(blockDim.y*gridDim.y)")
ADD_KEYWORD("GLOBAL_SIZE_2", "get_global_size(2)", "(blockDim.z*gridDim.z)")

ADD_KEYWORD("LOCAL_IDX_0", "get_local_id(0)", "threadIdx.x")
ADD_KEYWORD("LOCAL_IDX_1", "get_local_id(1)", "threadIdx.y")
ADD_KEYWORD("LOCAL_IDX_2", "get_local_id(2)", "threadIdx.z")

ADD_KEYWORD("LOCAL_SIZE_0", "get_local_size(0)", "blockDim.x")
ADD_KEYWORD("LOCAL_SIZE_1", "get_local_size(1)", "blockDim.y")
ADD_KEYWORD("LOCAL_SIZE_2", "get_local_size(2)", "blockDim.z")

ADD_KEYWORD("GROUP_IDX_0", "get_group_id(0)", "blockIdx.x")
ADD_KEYWORD("GROUP_IDX_1", "get_group_id(1)", "blockIdx.y")
ADD_KEYWORD("GROUP_IDX_2", "get_group_id(2)", "blockIdx.z")

ADD_KEYWORD("GROUP_SIZE_0", "get_ng(0)", "GridDim.x")
ADD_KEYWORD("GROUP_SIZE_1", "get_ng(1)", "GridDim.y")
ADD_KEYWORD("GROUP_SIZE_2", "get_ng(2)", "GridDim.z")

ADD_KEYWORD("LOCAL_BARRIER", "barrier(CLK_LOCAL_MEM_FENCE)", "__syncthreads()")
ADD_KEYWORD("LOCAL_PTR", "__local", "")

ADD_KEYWORD("LOCAL", "__local", "__shared__")
ADD_KEYWORD("GLOBAL", "__global", "")

ADD_KEYWORD("SIZE_T", "int", "int")
ADD_KEYWORD("KERNEL", "__kernel", "extern \"C\" __global__")

ADD_KEYWORD("MAD", "mad", "fma")

#undef ADD_KEYWORD
}

kernel_generation_stream::kernel_generation_stream(driver::backend_type backend) : std::ostream(new kgenstream(oss,tab_count_)), tab_count_(0), backend_(backend)
{ }

kernel_generation_stream::~kernel_generation_stream()
{ delete rdbuf(); }

std::string kernel_generation_stream::str()
{
  std::string next = oss.str();
  process(next);
  return next;
}

void kernel_generation_stream::inc_tab()
{ ++tab_count_; }

void kernel_generation_stream::dec_tab()
{ --tab_count_; }

}

