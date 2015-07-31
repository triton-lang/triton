#ifndef ISAAC_CL_QUEUES_H
#define ISAAC_CL_QUEUES_H

#include <map>
#include <list>

#include "isaac/defines.h"
#include "isaac/driver/command_queue.h"
#include "isaac/driver/context.h"

namespace isaac
{
namespace driver
{

class ISAACAPI backend
{
public:
  typedef std::list<std::pair<Context, std::vector<CommandQueue> > > container_type;
private:
  static std::vector<CommandQueue> & append( Context const & context);
  static void cuinit();
  static void clinit();
  static void init();
public:
  static container_type const & contexts();
  static Context default_context();
  static std::vector<CommandQueue> & default_queues();
  static std::vector<CommandQueue> & queues(Context const &);
private:
  static container_type data_;
public:
  static unsigned int default_device;
  static cl_command_queue_properties queue_properties;

};

ISAACAPI void synchronize(std::vector<CommandQueue> const &);
ISAACAPI void synchronize(Context const &);

}
}

#endif
