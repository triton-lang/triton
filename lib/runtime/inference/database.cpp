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
#include "isaac/runtime/inference/profiles.h"

//Default
#include "database/unknown/unknown.hpp"

//Intel
#include "database/intel/broadwell.hpp"

//NVidia
#include "database/nvidia/sm_2_0.hpp"
#include "database/nvidia/sm_3_0.hpp"
#include "database/nvidia/sm_3_5.hpp"
#include "database/nvidia/sm_5_2.hpp"
#include "database/nvidia/sm_6_1.hpp"

//AMD
#include "database/amd/gcn_1_1.hpp"
#include "database/amd/gcn_1_2.hpp"

namespace isaac
{
namespace runtime
{

#define DATABASE_ENTRY(TYPE, VENDOR, ARCHITECTURE, STRING) \
            {std::make_tuple(driver::Device::Type::TYPE, driver::Device::Vendor::VENDOR, driver::Device::Architecture::ARCHITECTURE), STRING}

const profiles::presets_type profiles::presets_ =
{
    //DEFAULT
    DATABASE_ENTRY(UNKNOWN, UNKNOWN, UNKNOWN, database::unknown::unknown),
    //INTEL
    DATABASE_ENTRY(GPU, INTEL, BROADWELL, database::intel::broadwell),
    //NVIDIA
    DATABASE_ENTRY(GPU, NVIDIA, SM_2_0, database::nvidia::sm_2_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_2_1, database::nvidia::sm_2_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_3_0, database::nvidia::sm_3_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_3_5, database::nvidia::sm_3_5),
    DATABASE_ENTRY(GPU, NVIDIA, SM_3_7, database::nvidia::sm_3_5),
    DATABASE_ENTRY(GPU, NVIDIA, SM_5_0, database::nvidia::sm_5_2),
    DATABASE_ENTRY(GPU, NVIDIA, SM_5_2, database::nvidia::sm_5_2),
    DATABASE_ENTRY(GPU, NVIDIA, SM_5_2, database::nvidia::sm_6_1),
    //AMD
    DATABASE_ENTRY(GPU, AMD, GCN_1_0, database::amd::gcn_1_1),
    DATABASE_ENTRY(GPU, AMD, GCN_1_1, database::amd::gcn_1_1),
    DATABASE_ENTRY(GPU, AMD, GCN_1_2, database::amd::gcn_1_2)
};

#undef DATABASE_ENTRY

}
}
