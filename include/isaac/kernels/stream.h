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

#ifndef ISAAC_BACKEND_STREAM_H
#define ISAAC_BACKEND_STREAM_H

#include <sstream>

namespace isaac
{

class kernel_generation_stream : public std::ostream
{
  class kgenstream : public std::stringbuf
  {
  public:
    kgenstream(std::ostringstream& oss,unsigned int const & tab_count) ;
    int sync();
    ~kgenstream();
  private:
    std::ostream& oss_;
    unsigned int const & tab_count_;
  };

public:
  kernel_generation_stream();
  ~kernel_generation_stream();

  std::string str();
  void inc_tab();
  void dec_tab();
private:
  unsigned int tab_count_;
  std::ostringstream oss;
};

}

#endif
