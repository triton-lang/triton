#include "isaac/driver/device.h"
#include "isaac/model/database.h"

#include "presets/broadwell.hpp"

namespace isaac
{


#define DATABASE_ENTRY(VENDOR, ARCHITECTURE, STRING) \
            {std::make_tuple(driver::Device::Vendor::VENDOR, driver::Device::Architecture::ARCHITECTURE), STRING}

const std::map<std::tuple<driver::Device::Vendor, driver::Device::Architecture> , const char *> database::presets_ =
            { DATABASE_ENTRY(INTEL, BROADWELL, presets::broadwell) };


#undef DATABASE_ENTRY

}
