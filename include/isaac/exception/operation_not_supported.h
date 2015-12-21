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

#ifndef ISAAC_EXCEPTION_OPERATION_NOT_SUPPORTED_H
#define ISAAC_EXCEPTION_OPERATION_NOT_SUPPORTED_H

#include <string>
#include <exception>

#include "isaac/defines.h"

namespace isaac
{

/** @brief Exception for the case the generator is unable to deal with the operation */
DISABLE_MSVC_WARNING_C4275
class operation_not_supported_exception : public std::exception
{
public:
  operation_not_supported_exception();
  operation_not_supported_exception(std::string message);
  virtual const char* what() const throw();
  virtual ~operation_not_supported_exception() throw();
private:
DISABLE_MSVC_WARNING_C4251
  std::string message_;
RESTORE_MSVC_WARNING_C4251
};
RESTORE_MSVC_WARNING_C4275

}

#endif
