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

#ifndef ISAAC_TOOLS_FIND_AND_REPLACE_HPP
#define ISAAC_TOOLS_FIND_AND_REPLACE_HPP

#include <string>

namespace isaac
{
namespace tools
{

int inline find_and_replace(std::string & source, std::string const & find, std::string const & replace)
{
  int num=0;
  size_t fLen = find.size();
  size_t rLen = replace.size();
  for (size_t pos=0; (pos=source.find(find, pos))!=std::string::npos; pos+=rLen)
  {
    num++;
    source.replace(pos, fLen, replace);
  }
  return num;
}

}
}

#endif
