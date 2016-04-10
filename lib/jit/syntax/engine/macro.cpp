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

#include <algorithm>
#include <tuple>
#include "isaac/jit/syntax/engine/macro.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

namespace symbolic
{

macro::macro(std::string const & code): code_(code)
{
  size_t pos_po = code_.find('(');
  size_t pos_pe = code_.find(')');
  name_ = code.substr(0, pos_po);
  args_ = tools::split(code.substr(pos_po + 1, pos_pe - pos_po - 1), ',');
  tokens_ = tools::tokenize(code_.substr(code_.find(":") + 1), "()[],*+/-=>< ");
}

macro::macro(const char *code) : macro(std::string(code))
{

}

int macro::expand(std::string & str) const
{
  size_t pos = 0;
  size_t num_touched = 0;
  while((pos=str.find(name_ + "(", pos==0?0:pos + 1))!=std::string::npos){
    size_t pos_po = str.find('(', pos);
    size_t pos_pe = str.find(')', pos_po);
    size_t next = str.find('(', pos_po + 1);
    while(next < pos_pe){
      pos_pe = str.find(')', pos_pe + 1);
      if(next < pos_pe)
        next = str.find('(', next + 1);
    }

    std::vector<std::string> args = tools::split(str.substr(pos_po + 1, pos_pe - pos_po - 1), ',');
    if(args_.size() != args.size()){
      pos = pos_pe;
      continue;
    }

    //Process
    std::vector<std::string> tokens = tokens_;
    for(size_t i = 0 ; i < args_.size() ; ++i)
      std::replace(tokens.begin(), tokens.end(), args_[i], args[i]);

    //Replace
    str.replace(pos, pos_pe + 1 - pos, tools::join(tokens.begin(), tokens.end(), ""));
    num_touched++;
  }
  return num_touched;
}

bool macro::operator<(macro const & o) const
{
  return std::make_tuple(name_, args_.size()) < std::make_tuple(o.name_, o.args_.size());
}

}
}
