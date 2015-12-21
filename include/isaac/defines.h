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

#ifndef ISAAC_DEFINES_H
#define ISAAC_DEFINES_H

#if defined(_WIN32) || defined(_MSC_VER)
    #ifdef ISAAC_DLL
        #define ISAACAPI  __declspec(dllexport)
    #else
        #define ISAACAPI  __declspec(dllimport)
    #endif
#else
    #define ISAACAPI   __attribute__((visibility("default")))
#endif

#if defined(_WIN32) || defined(_MSC_VER)
	#define DISABLE_MSVC_WARNING_C4251 __pragma(warning(disable: 4251))
	#define RESTORE_MSVC_WARNING_C4251 __pragma(warning(default: 4251))
	#define DISABLE_MSVC_WARNING_C4275 __pragma(warning(disable: 4275))
	#define RESTORE_MSVC_WARNING_C4275 __pragma(warning(disable: 4275))

#else
    #define DISABLE_MSVC_WARNING_C4251
    #define RESTORE_MSVC_WARNING_C4251
    #define DISABLE_MSVC_WARNING_C4275
    #define RESTORE_MSVC_WARNING_C4275
#endif

#endif
