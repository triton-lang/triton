#ifndef ISAAC_CL_QUEUES_H
#define ISAAC_CL_QUEUES_H

#include <map>
#include <list>
#include <vector>

#include "isaac/driver/common.h"
#include "isaac/driver/program.h"
#include "isaac/defines.h"
#include "isaac/types.h"

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

public:
  class programs
  {
      friend class backend;
  public:
      static void release();
      static ProgramCache & get(CommandQueue const & queue, expression_type expression, numeric_type dtype);
  private:
      static std::map<std::tuple<CommandQueue, expression_type, numeric_type>, ProgramCache * > cache_;
  };

  class contexts
  {
      friend class backend;
  private:
      static void init(std::vector<Platform> const &);
      static void release();
  public:
      static Context const & get_default();
      static Context const & import(cl_context context);
      static void get(std::list<Context const *> &);
  private:
      static std::list<Context const *> cache_;
  };

  class queues
  {
      friend class backend;
  private:
      static void init(std::list<Context const *> const &);
      static void release();
  public:
      static void get(Context const &, std::vector<CommandQueue *> &queues);
      static CommandQueue & get(Context const &, unsigned int id);
  private:
      static std::map< Context, std::vector<CommandQueue*> > cache_;
  };

  static void init();
  static void release();

  static void platforms(std::vector<Platform> &);
  static void synchronize(Context const &);

public:
  static unsigned int default_device;
  static cl_command_queue_properties queue_properties;

};

}
}

#endif
