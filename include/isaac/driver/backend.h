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

class CommandQueue;
class Context;
class Platform;

class ISAACAPI backend
{
private:
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

  class contexts
  {
      friend class backend;
  public:
      static Context const & get_default();
      static Context const & import(cl_context context);
      static void get(std::list<Context const *> &);
  private:
      static void release();
      static std::list<Context const *> contexts_;
  };

  class queues
  {
      friend class backend;
  public:
      static void get(Context const &, std::vector<CommandQueue*> queues);
      static CommandQueue & get(Context const &, unsigned int id);
  private:
      static void release();
      static std::map< Context const *, std::vector<CommandQueue*> > queues_;
  };

  static void platforms(std::vector<Platform> &);
  static void synchronize(Context const &);
  static void release();

private:

public:
  static unsigned int default_device;
  static cl_command_queue_properties queue_properties;

};

}
}

#endif
