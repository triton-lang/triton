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
#ifndef ISAAC_DRIVER_PROGRAM_CACHE_H
#define ISAAC_DRIVER_PROGRAM_CACHE_H

#include <map>
#include "isaac/defines.h"
#include "isaac/driver/program.h"

namespace isaac
{

namespace driver
{

class ISAACAPI ProgramCache
{
    friend class backend;
public:
    void clear();
    Program & add(Context const & context, std::string const & name, std::string const & src);
    Program const *find(std::string const & name);
private:
DISABLE_MSVC_WARNING_C4251
    std::map<std::string, Program> cache_;
RESTORE_MSVC_WARNING_C4251
};


}

}

#endif
