/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */
#include "isaac/driver/device.h"
#include "isaac/profiles/profiles.h"

//Intel
#include "presets/broadwell.hpp"
//NVidia
#include "presets/maxwell.hpp"
//AMD
#include "presets/fiji.hpp"
#include "presets/hawaii.hpp"

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
    DATABASE_ENTRY(GPU, AMD, GCN_1_1, presets::hawaii),
    DATABASE_ENTRY(GPU, AMD, GCN_1_2, presets::fiji)
};


#undef DATABASE_ENTRY

}
