#include "isaac/driver/device.h"
#include "isaac/profiles/profiles.h"

#include "presets/broadwell.hpp"

namespace isaac
{


#define DATABASE_ENTRY(TYPE, VENDOR, ARCHITECTURE, STRING) \
            {std::make_tuple(driver::Device::Type::TYPE, driver::Device::Vendor::VENDOR, driver::Device::Architecture::ARCHITECTURE), STRING}

const profiles::presets_type profiles::presets_ =
            {  };


#undef DATABASE_ENTRY

}
