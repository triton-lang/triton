#ifndef ISAAC_CL_QUEUES_H
#define ISAAC_CL_QUEUES_H

#include <map>
#include <list>
#include <vector>

#include "isaac/driver/common.h"
#include "isaac/defines.h"

namespace isaac
{
namespace driver
{

class Context;
class CommandQueue;

class ISAACAPI backend
{
public:
  typedef std::list<Context> context_container;
  typedef std::map<Context*, std::vector<CommandQueue>> queues_container;
private:
  static void cuinit();
  static void clinit();
  static void init();
public:
  static std::list<Context> const & contexts();
  static Context const & import(cl_context context);
  static Context const & default_context();
  static std::vector<CommandQueue> & queues(Context const &);
private:
  static std::list<Context> contexts_;
  static std::map<Context*, std::vector<CommandQueue>> queues_;
public:
  static unsigned int default_device;
  static cl_command_queue_properties queue_properties;

};

ISAACAPI void synchronize(std::vector<CommandQueue> const &);
ISAACAPI void synchronize(Context const &);

}
}

#endif
