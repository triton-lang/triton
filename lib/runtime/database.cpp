/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "isaac/driver/device.h"
#include "isaac/runtime/profiles.h"

//Default
#include "database/unknown/unknown.hpp"

//Intel
#include "database/intel/broadwell.hpp"
#include "database/intel/skylake.hpp"

//NVidia
#include "database/nvidia/sm_3_0.hpp"
#include "database/nvidia/sm_5_2.hpp"
#include "database/nvidia/sm_6_0.hpp"
#include "database/nvidia/sm_6_1.hpp"

//AMD
#include "database/amd/gcn_3.hpp"

namespace isaac
{
namespace runtime
{

#define DATABASE_ENTRY(TYPE, VENDOR, ARCHITECTURE, STRING) \
            {std::make_tuple(driver::Device::Type::TYPE, driver::Device::Vendor::VENDOR, driver::Device::Architecture::ARCHITECTURE), STRING}

const profiles::presets_type profiles::presets_ =
{
    //DEFAULT
    DATABASE_ENTRY(UNKNOWN, UNKNOWN, UNKNOWN, database::intel::broadwell),

    //INTEL
    DATABASE_ENTRY(GPU, INTEL, HASWELL, database::intel::broadwell),
    DATABASE_ENTRY(GPU, INTEL, BROADWELL, database::intel::broadwell),
    DATABASE_ENTRY(GPU, INTEL, SKYLAKE, database::intel::skylake),
    DATABASE_ENTRY(GPU, INTEL, KABYLAKE, database::intel::skylake),

    //NVIDIA
    DATABASE_ENTRY(GPU, NVIDIA, SM_2_0, database::nvidia::sm_3_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_2_1, database::nvidia::sm_3_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_3_0, database::nvidia::sm_3_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_3_5, database::nvidia::sm_3_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_3_7, database::nvidia::sm_3_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_5_0, database::nvidia::sm_5_2),
    DATABASE_ENTRY(GPU, NVIDIA, SM_5_2, database::nvidia::sm_5_2),
    DATABASE_ENTRY(GPU, NVIDIA, SM_6_0, database::nvidia::sm_6_0),
    DATABASE_ENTRY(GPU, NVIDIA, SM_6_1, database::nvidia::sm_6_1),
    //AMD
    DATABASE_ENTRY(GPU, AMD, GCN_1, database::amd::gcn_3),
    DATABASE_ENTRY(GPU, AMD, GCN_2, database::amd::gcn_3),
    DATABASE_ENTRY(GPU, AMD, GCN_3, database::amd::gcn_3),
    DATABASE_ENTRY(GPU, AMD, GCN_4, database::amd::gcn_3)
};

#undef DATABASE_ENTRY

}
}
