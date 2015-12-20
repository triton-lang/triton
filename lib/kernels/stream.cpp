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
#include "isaac/kernels/stream.h"

namespace isaac
{

kernel_generation_stream::kgenstream::kgenstream(std::ostringstream& oss,unsigned int const & tab_count) :
  oss_(oss), tab_count_(tab_count)
{ }

int kernel_generation_stream::kgenstream::sync()
{
  for (unsigned int i=0; i<tab_count_;++i)
    oss_ << "    ";
  oss_ << str();
  str("");
  return !oss_;
}

kernel_generation_stream::kgenstream:: ~kgenstream()
{  pubsync(); }

kernel_generation_stream::kernel_generation_stream() : std::ostream(new kgenstream(oss,tab_count_)), tab_count_(0)
{ }

kernel_generation_stream::~kernel_generation_stream()
{ delete rdbuf(); }

std::string kernel_generation_stream::str()
{ return oss.str(); }

void kernel_generation_stream::inc_tab()
{ ++tab_count_; }

void kernel_generation_stream::dec_tab()
{ --tab_count_; }

}

