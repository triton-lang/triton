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

#include "isaac/driver/program_cache.h"

namespace isaac
{

namespace driver
{

Program & ProgramCache::add(Context const & context, std::string const & name, std::string const & src)
{
    std::map<std::string, Program>::iterator it = cache_.find(name);
    if(it==cache_.end())
    {
        std::string extensions;
        std::string ext = "cl_khr_fp64";
        if(context.device().extensions().find(ext)!=std::string::npos)
          extensions = "#pragma OPENCL EXTENSION " + ext + " : enable\n";
        return cache_.insert(std::make_pair(name, driver::Program(context, extensions + src))).first->second;
    }
    return it->second;
}

Program const * ProgramCache::find(const std::string &name)
{
    std::map<std::string, Program>::const_iterator it = cache_.find(name);
    if(it==cache_.end())
        return NULL;
    return &(it->second);
}

void ProgramCache::clear()
{
    cache_.clear();
}

}

}

