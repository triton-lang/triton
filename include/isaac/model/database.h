#ifndef ISAAC_MODEL_DATABASE_H
#define ISAAC_MODEL_DATABASE_H

#include <map>
#include <memory>

#include "isaac/driver/command_queue.h"
#include "isaac/driver/device.h"
#include "isaac/common/expression_type.h"
#include "isaac/common/numeric_type.h"
#include "isaac/model/model.h"

namespace isaac
{

struct database
{
public:
    typedef std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<model> > map_type;
private:
    static void import(std::string const & fname, driver::CommandQueue const & queue);
    static map_type & init(driver::CommandQueue const & queue);
public:
    static map_type & get(driver::CommandQueue const & queue);
    static void set(driver::CommandQueue const & queue, expression_type operation, numeric_type dtype, std::shared_ptr<model> const & model);
private:
    static const std::map<std::tuple<driver::Device::Vendor, driver::Device::Architecture> , const char *> presets_;
    static std::map<driver::CommandQueue, map_type> cache_;
};

}

#endif
