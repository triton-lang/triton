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

#ifndef TDL_TOOLS_SYS_MKDIR_HPP
#define TDL_TOOLS_SYS_MKDIR_HPP

#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <string>
#include <sys/stat.h>
#if defined(_WIN32)
#include <direct.h>
#endif

namespace triton {

namespace tools {

inline int mkdir(std::string const &path) {
#if defined(_WIN32)
  return _mkdir(path.c_str());
#else
  return ::mkdir(path.c_str(), 0777);
#endif
}

inline int mkpath(std::string const &path) {
  int status = 0;
  size_t pp = 0;
  size_t sp;
  while ((sp = path.find('/', pp)) != std::string::npos) {
    if (sp != pp) {
      status = mkdir(path.substr(0, sp));
    }
    pp = sp + 1;
  }
  return (status == 0 || errno == EEXIST) ? 0 : -1;
}

inline int mtime(std::string const &path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0)
    return 0;
  return st.st_mtime;
}

} // namespace tools

} // namespace triton

#endif
