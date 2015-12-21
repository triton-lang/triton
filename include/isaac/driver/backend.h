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

#ifndef ISAAC_CL_QUEUES_H
#define ISAAC_CL_QUEUES_H

#include <map>
#include <list>
#include <vector>

#include "isaac/common/expression_type.h"
#include "isaac/common/numeric_type.h"

#include "isaac/driver/dispatch.h"
#include "isaac/defines.h"
#include "isaac/types.h"

namespace isaac
{
namespace driver
{

class Buffer;
class CommandQueue;
class Context;
class Platform;
class ProgramCache;

class ISAACAPI backend
{
public:
  class ISAACAPI workspaces
  {
  public:
      static const size_t SIZE = 8000000; //8MB of temporary workspace per queue
      static void release();
      static driver::Buffer & get(CommandQueue const & key);
  private:
      DISABLE_MSVC_WARNING_C4251
      static std::map<CommandQueue, Buffer * > cache_;
      RESTORE_MSVC_WARNING_C4251
  };

  class ISAACAPI programs
  {
      friend class backend;
  public:
      static void release();
      static ProgramCache & get(CommandQueue const & queue, expression_type expression, numeric_type dtype);
  private:
DISABLE_MSVC_WARNING_C4251
      static std::map<std::tuple<CommandQueue, expression_type, numeric_type>, ProgramCache * > cache_;
RESTORE_MSVC_WARNING_C4251

  };

  class ISAACAPI contexts
  {
      friend class backend;
  private:
      static void init(std::vector<Platform> const &);
      static void release();
  public:
      static Context const & get_default();
      static Context const & import(CUcontext context);
      static Context const & import(cl_context context);
      static void get(std::list<Context const *> &);
  private:
DISABLE_MSVC_WARNING_C4251
      static std::list<Context const *> cache_;
RESTORE_MSVC_WARNING_C4251
  };

  class ISAACAPI queues
  {
      friend class backend;
  private:
      static void init(std::list<Context const *> const &);
      static void release();
  public:
      static void get(Context const &, std::vector<CommandQueue *> &queues);
      static CommandQueue & get(Context const &, unsigned int id = 0);
  private:
DISABLE_MSVC_WARNING_C4251
      static std::map< Context, std::vector<CommandQueue*> > cache_;
RESTORE_MSVC_WARNING_C4251
  };

  static void init();
  static void release();

  static void platforms(std::vector<Platform> &);
  static void synchronize(Context const &);

public:
  static unsigned int default_device;
  static cl_command_queue_properties default_queue_properties;
};

}
}

#endif
