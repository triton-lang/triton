#ifndef ISAAC_CL_QUEUES_H
#define ISAAC_CL_QUEUES_H

#include <map>
#include <list>

#include "isaac/driver/command_queue.h"
#include "isaac/driver/context.h"

namespace isaac
{
namespace driver
{

class queues_type
{
public:
  typedef std::list<std::pair<Context, std::vector<CommandQueue> > > container_type;
private:
  std::vector<CommandQueue> & append( Context const & context);
  void cuinit();
  void clinit();
  void init();
public:
  queues_type();
  std::vector<CommandQueue> & operator[](Context const &);
  Context default_context();
  std::vector<CommandQueue> & default_queues();
  container_type const & contexts();
private:
  container_type data_;
public:
  unsigned int default_device;
  cl_command_queue_properties queue_properties;

};

void synchronize(std::vector<CommandQueue> const &);
void synchronize(Context const &);

extern queues_type queues;


}
}

#endif
