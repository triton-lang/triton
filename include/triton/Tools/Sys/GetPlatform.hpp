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

#ifndef TDL_TOOLS_SYS_GETPLATFORM_HPP
#define TDL_TOOLS_SYS_GETPLATFORM_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>

inline std::map<std::string, bool> cache{};
inline bool isROCM() {
  // only need to run function once after that return cached value
  if (cache.find("isROCM") != cache.end()) {
    return cache["isROCM"];
  }

  // run command
  std::string cmd = "apt-cache show rocm-libs | grep 'Package:'";
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    std::cout << ("cmd failed!") << std::endl;
  }

  // get output
  std::string result;
  std::array<char, 128> buffer;
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  // check ROCM that is found
  cache["isROCM"] = result.find("rocm") != std::string::npos;
  return cache["isROCM"];
}

#endif
