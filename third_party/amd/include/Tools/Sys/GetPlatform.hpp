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

#ifndef TDL_TOOLS_SYS_GETPLATFORM_HPP
#define TDL_TOOLS_SYS_GETPLATFORM_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "hsa.h"
#include "hsa_ext_amd.h"

// This structure holds agent information acquired through hsa info related
// calls, and is later used for reference when displaying the information.
struct agent_info_t {
  char name[64];
  char uuid[24];
  char vendor_name[64];
  char device_mkt_name[64];
  hsa_agent_feature_t agent_feature;
  hsa_profile_t agent_profile;
  hsa_default_float_rounding_mode_t float_rounding_mode;
  uint32_t max_queue;
  uint32_t queue_min_size;
  uint32_t queue_max_size;
  hsa_queue_type_t queue_type;
  uint32_t node;
  hsa_device_type_t device_type;
  uint32_t cache_size[4];
  uint32_t chip_id;
  uint32_t asic_revision;
  uint32_t cacheline_size;
  uint32_t max_clock_freq;
  uint32_t internal_node_id;
  uint32_t max_addr_watch_pts;
  // HSA_AMD_AGENT_INFO_MEMORY_WIDTH is deprecated, so exclude
  // uint32_t mem_max_freq; Not supported by get_info
  uint32_t compute_unit;
  uint32_t wavefront_size;
  uint32_t workgroup_max_size;
  uint32_t grid_max_size;
  uint32_t fbarrier_max_size;
  uint32_t max_waves_per_cu;
  uint32_t simds_per_cu;
  uint32_t shader_engs;
  uint32_t shader_arrs_per_sh_eng;
  hsa_isa_t agent_isa;
  hsa_dim3_t grid_max_dim;
  uint16_t workgroup_max_dim[3];
  uint16_t bdf_id;
  bool fast_f16;
};

// This structure holds ISA information acquired through hsa info
// related calls, and is later used for reference when displaying the
// information.
struct isa_info_t {
  char *name_str;
  uint32_t workgroup_max_size;
  hsa_dim3_t grid_max_dim;
  uint64_t grid_max_size;
  uint32_t fbarrier_max_size;
  uint16_t workgroup_max_dim[3];
  bool def_rounding_modes[3];
  bool base_rounding_modes[3];
  bool mach_models[2];
  bool profiles[2];
  bool fast_f16;
};

static hsa_status_t AcquireAgentInfo(hsa_agent_t agent, agent_info_t *agent_i) {
  hsa_status_t err;
  // Get agent name and vendor
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_i->name);

  // Get UUID, an Ascii string, of a ROCm device
  err = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID,
                           &agent_i->uuid);

  // Get device's vendor name
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_VENDOR_NAME,
                           &agent_i->vendor_name);

  // Get device marketing name
  err = hsa_agent_get_info(agent,
                           (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME,
                           &agent_i->device_mkt_name);

  // Get agent feature
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE,
                           &agent_i->agent_feature);

  // Get profile supported by the agent
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE,
                           &agent_i->agent_profile);

  // Get floating-point rounding mode
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
                           &agent_i->float_rounding_mode);

  // Get max number of queue
  err =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUES_MAX, &agent_i->max_queue);

  // Get queue min size
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                           &agent_i->queue_min_size);

  // Get queue max size
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                           &agent_i->queue_max_size);

  // Get queue type
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_TYPE,
                           &agent_i->queue_type);

  // Get agent node
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &agent_i->node);

  // Get device type
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agent_i->device_type);

  if (HSA_DEVICE_TYPE_GPU == agent_i->device_type) {
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &agent_i->agent_isa);
  }

  // Get cache size
  err =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_CACHE_SIZE, agent_i->cache_size);

  // Get chip id
  err = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CHIP_ID,
                           &agent_i->chip_id);

  // Get asic revision
  err = hsa_agent_get_info(agent,
                           (hsa_agent_info_t)HSA_AMD_AGENT_INFO_ASIC_REVISION,
                           &agent_i->asic_revision);

  // Get cacheline size
  err = hsa_agent_get_info(agent,
                           (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
                           &agent_i->cacheline_size);

  // Get Max clock frequency
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY,
      &agent_i->max_clock_freq);

  // Internal Driver node ID
  err = hsa_agent_get_info(agent,
                           (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_NODE_ID,
                           &agent_i->internal_node_id);

  // Max number of watch points on mem. addr. ranges to generate exeception
  // events
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS,
      &agent_i->max_addr_watch_pts);

  // Get Agent BDFID
  err = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,
                           &agent_i->bdf_id);

  // Get Max Memory Clock
  // Not supported by hsa_agent_get_info
  //  err = hsa_agent_get_info(agent,d
  //              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY,
  //                                                      &agent_i->mem_max_freq);
  //

  // Get Num SIMDs per CU
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU,
      &agent_i->simds_per_cu);

  // Get Num Shader Engines
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES,
      &agent_i->shader_engs);

  // Get Num Shader Arrays per Shader engine
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE,
      &agent_i->shader_arrs_per_sh_eng);

  // Get number of Compute Unit
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      &agent_i->compute_unit);

  // Check if the agent is kernel agent
  if (agent_i->agent_feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
    // Get flaf of fast_f16 operation
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_FAST_F16_OPERATION,
                             &agent_i->fast_f16);

    // Get wavefront size
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE,
                             &agent_i->wavefront_size);

    // Get max total number of work-items in a workgroup
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE,
                             &agent_i->workgroup_max_size);

    // Get max number of work-items of each dimension of a work-group
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                             &agent_i->workgroup_max_dim);

    // Get max number of a grid per dimension
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_DIM,
                             &agent_i->grid_max_dim);

    // Get max total number of work-items in a grid
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_SIZE,
                             &agent_i->grid_max_size);

    // Get max number of fbarriers per work group
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_FBARRIER_MAX_SIZE,
                             &agent_i->fbarrier_max_size);

    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
        &agent_i->max_waves_per_cu);
  }
  return err;
}

static hsa_status_t getAgentArchInfo(hsa_agent_t agent, void *data) {

  hsa_status_t err;
  agent_info_t agent_i;
  std::tuple<std::string, int> *arch_info =
      reinterpret_cast<std::tuple<std::string, int> *>(data);

  err = AcquireAgentInfo(agent, &agent_i);
  if (std::string(agent_i.name).rfind("gfx", 0) == 0) {
    // name length
    uint32_t name_len;
    err = hsa_isa_get_info_alt(agent_i.agent_isa, HSA_ISA_INFO_NAME_LENGTH,
                               &name_len);

    // create buffer
    isa_info_t isa_i;
    isa_i.name_str = new char[name_len];
    if (isa_i.name_str == nullptr) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    // fill data
    err = hsa_isa_get_info_alt(agent_i.agent_isa, HSA_ISA_INFO_NAME,
                               isa_i.name_str);

    // set output tuple
    std::get<0>(*arch_info) = isa_i.name_str;
    std::get<1>(*arch_info) = agent_i.wavefront_size;
  }

  return HSA_STATUS_SUCCESS;
}

inline std::tuple<std::string, int> getArchInfo() {

  hsa_status_t err;
  std::tuple<std::string, int> arch_info = std::make_tuple("", 0);
  err = hsa_init();
  err = hsa_iterate_agents(getAgentArchInfo, &arch_info);

  return arch_info;
}

#endif
