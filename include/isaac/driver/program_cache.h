#ifndef ISAAC_DRIVER_PROGRAM_CACHE_H
#define ISAAC_DRIVER_PROGRAM_CACHE_H

#include <map>
#include "isaac/defines.h"
#include "isaac/driver/program.h"

namespace isaac
{

namespace driver
{

class ISAACAPI ProgramCache
{
    friend class backend;
public:
    void clear();
    Program & add(Context const & context, std::string const & name, std::string const & src);
    Program const *find(std::string const & name);
private:
    std::map<std::string, Program> cache_;
};


}

}

#endif
