/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
class Program;
class Kernel;
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

  class ISAACAPI kernels
  {
      friend class backend;
  public:
      static void release();
      static Kernel & get(Program const & program, std::string const & name);
  private:
DISABLE_MSVC_WARNING_C4251
      static std::map<std::tuple<Program, std::string>, Kernel * > cache_;
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
