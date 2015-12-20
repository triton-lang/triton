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
#ifndef ISAAC_TOOLS_GETENV
#define ISAAC_TOOLS_GETENV

#include <string>
#include <cstdlib>

namespace isaac
{

namespace tools
{

    inline std::string getenv(const char * name)
    {
        #ifdef _MSC_VER
            char* cache_path = 0;
            std::size_t sz = 0;
            _dupenv_s(&cache_path, &sz, name);
        #else
            const char * cache_path = std::getenv(name);
        #endif
        if(!cache_path)
            return "";
        std::string result(cache_path);
        #ifdef _MSC_VER
            free(cache_path);
        #endif
        return result;
    }

}

}

#endif
