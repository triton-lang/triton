#include "isaac/driver/device.h"
#include "isaac/profiles/profiles.h"

//Intel
#include "presets/broadwell.hpp"
//NVidia
#include "presets/maxwell.hpp"
//AMD
#include "presets/fiji.hpp"

namespace isaac
{


#define DATABASE_ENTRY(TYPE, VENDOR, ARCHITECTURE, STRING) \
            {std::make_tuple(driver::Device::Type::TYPE, driver::Device::Vendor::VENDOR, driver::Device::Architecture::ARCHITECTURE), STRING}

const profiles::presets_type profiles::presets_ =
{
    //INTEL
    DATABASE_ENTRY(GPU, INTEL, BROADWELL, presets::broadwell),
    //NVIDIA
    DATABASE_ENTRY(GPU, NVIDIA, MAXWELL, presets::maxwell),
    //AMD
    DATABASE_ENTRY(GPU, AMD, GCN_1_2, presets::fiji)
};


#undef DATABASE_ENTRY

}
