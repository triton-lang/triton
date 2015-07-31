#ifndef ISAAC_CL_QUEUES_H
#define ISAAC_CL_QUEUES_H

#include <map>
#include <list>
#include <vector>

#include "isaac/driver/common.h"
#include "isaac/driver/program.h"
#include "isaac/defines.h"

namespace isaac
{
namespace driver
{

class Context;
class CommandQueue;

class ISAACAPI backend
{
private:
  static void cuinit();
  static void clinit();
  static void init();
public:
  class programs
  {
  public:
      static Program const & add(Context const & scontext, std::string const & name, std::string const & src);
      static Program const * find(Context const & context, std::string const & name);
      static void release();
  private:
      static std::map<driver::Context const *, std::map<std::string, Program*> > programs_;
  };

  static std::list<Context const *> const  & contexts();
  static Context const & import(cl_context context);
  static Context const & default_context();
  static void synchronize(Context const &);
  static CommandQueue & queue(Context const &, unsigned int id);

  static void release();
private:
  static std::list<Context const *> contexts_;
  static std::map<Context const *, std::vector<CommandQueue*>> queues_;
public:
  static unsigned int default_device;
  static cl_command_queue_properties queue_properties;

};

}
}

#endif
