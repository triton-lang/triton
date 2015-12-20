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
#include "isaac/exception/operation_not_supported.h"

namespace isaac
{

operation_not_supported_exception::operation_not_supported_exception() : message_()
{}

operation_not_supported_exception::operation_not_supported_exception(std::string message) :
  message_("ISAAC: Internal error: The internal generator cannot handle the operation provided: " + message) {}

const char* operation_not_supported_exception::what() const throw()
{ return message_.c_str(); }

operation_not_supported_exception::~operation_not_supported_exception() throw()
{}

}
