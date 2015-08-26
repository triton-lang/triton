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
DISABLE_MSVC_WARNING_C4251
    std::map<std::string, Program> cache_;
RESTORE_MSVC_WARNING_C4251
};


}

}

#endif
