////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2017-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef OPENSRC_HSA_RUNTIME_INC_HSA_VEN_AMD_AQLPROFILE_H_
#define OPENSRC_HSA_RUNTIME_INC_HSA_VEN_AMD_AQLPROFILE_H_

#include <stdint.h>
#include "hsa.h"

#define HSA_AQLPROFILE_VERSION_MAJOR 2
#define HSA_AQLPROFILE_VERSION_MINOR 0

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

////////////////////////////////////////////////////////////////////////////////
// Library version
uint32_t hsa_ven_amd_aqlprofile_version_major();
uint32_t hsa_ven_amd_aqlprofile_version_minor();

///////////////////////////////////////////////////////////////////////
// Library API:
// The library provides helper methods for instantiation of
// the profile context object and for populating of the start
// and stop AQL packets. The profile object contains a profiling
// events list and needed for profiling buffers descriptors,
// a command buffer and an output data buffer. To check if there
// was an error the library methods return a status code. Also
// the library provides methods for querying required buffers
// attributes, to validate the event attributes and to get profiling
// output data.
//
// Returned status:
//     hsa_status_t – HSA status codes are used from hsa.h header
//
// Supported profiling features:
//
// Supported profiling events
typedef enum {
  HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC = 0,
  HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE = 1,
} hsa_ven_amd_aqlprofile_event_type_t;

// Supported performance counters (PMC) blocks
// The block ID is the same for a block instances set, for example
// each block instance from the TCC block set, TCC0, TCC1, …, TCCN
// will have the same block ID HSA_VEN_AMD_AQLPROFILE_BLOCKS_TCC.
typedef enum {
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC = 0,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF = 1,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS = 2,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM = 3,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE = 4,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI = 5,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ = 6,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS = 7,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM = 8,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX = 9,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA = 10,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA = 11,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC = 12,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP = 13,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD = 14,
  // Memory related blocks
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB = 15,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB = 16,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM = 17,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ = 18,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2 = 19,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR = 20,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC = 21,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2 = 22,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA = 23,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB = 24,
  // System blocks
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA = 25,
  // GFX10 added blocks
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A = 26,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C = 27,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A = 28,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C = 29,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR = 30,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS = 31,

  // UMC & MMEA System Blocks
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC = 32,
  HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MMEA = 33,

  HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER
} hsa_ven_amd_aqlprofile_block_name_t;

// PMC event object structure
// ‘counter_id’ value is specified in GFXIPs perfcounter user guides
// which is the counters select value, “Performance Counters Selection”
// chapter.
typedef struct {
  hsa_ven_amd_aqlprofile_block_name_t block_name;
  uint32_t block_index;
  uint32_t counter_id;
} hsa_ven_amd_aqlprofile_event_t;

// Check if event is valid for the specific GPU
hsa_status_t hsa_ven_amd_aqlprofile_validate_event(
    hsa_agent_t agent,                            // HSA handle for the profiling GPU
    const hsa_ven_amd_aqlprofile_event_t* event,  // [in] Pointer on validated event
    bool* result);                                // [out] True if the event valid, False otherwise

// Profiling parameters
// All parameters are generic and if not applicable for a specific
// profile configuration then error status will be returned.
typedef enum {
  /*
  * Select the target compute unit (wgp) for profiling.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET = 0,
  /*
  * VMID Mask
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK = 1,
  /*
  * Legacy. Deprecated.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK = 2,
  /*
  * Legacy. Deprecated.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK = 3,
  /*
  * Legacy. Deprecated.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2 = 4,
  /*
  * Shader engine mask for selection.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK = 5,
  /*
  * Legacy. Deprecated.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE = 6,
  /*
  * Legacy. Deprecated.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT = 7,
  /*
  * Set SIMD Mask (GFX9) or SIMD ID for collection (Navi)
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION = 8,
  /*
  * Set true for occupancy collection only.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE = 9,
  /*
  * ATT collection max data size, in MB. Shared among shader engines.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE = 10,
  /*
  * Mask of which compute units to generate perfcounters. GFX9 only.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_MASK = 240,
  /*
  * Select collection period for perfcounters. GFX9 only.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL = 241,
  /*
  * Select perfcounter ID (SQ block) for collection. GFX9 only.
  */
  HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME = 242,
} hsa_ven_amd_aqlprofile_parameter_name_t;

// Profile parameter object
typedef struct {
  hsa_ven_amd_aqlprofile_parameter_name_t parameter_name;
  uint32_t value;
} hsa_ven_amd_aqlprofile_parameter_t;

typedef enum {
  HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_0 = 0,
  HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_1,
  HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_2,
  HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_3
} hsa_ven_amd_aqlprofile_att_marker_channel_t;

//
// Profile context object:
// The library provides a profile object structure which contains
// the events array, a buffer for the profiling start/stop commands
// and a buffer for the output data.
// The buffers are specified by the buffer descriptors and allocated
// by the application. The buffers allocation attributes, the command
// buffer size, the PMC output buffer size as well as profiling output
// data can be get using the generic get profile info helper _get_info.
//
// Buffer descriptor
typedef struct {
  void* ptr;
  uint32_t size;
} hsa_ven_amd_aqlprofile_descriptor_t;

// Profile context object structure, contains profiling events list and
// needed for profiling buffers descriptors, a command buffer and
// an output data buffer
typedef struct {
  hsa_agent_t agent;                                     // GFXIP handle
  hsa_ven_amd_aqlprofile_event_type_t type;              // Events type
  const hsa_ven_amd_aqlprofile_event_t* events;          // Events array
  uint32_t event_count;                                  // Events count
  const hsa_ven_amd_aqlprofile_parameter_t* parameters;  // Parameters array
  uint32_t parameter_count;                              // Parameters count
  hsa_ven_amd_aqlprofile_descriptor_t output_buffer;     // Output buffer
  hsa_ven_amd_aqlprofile_descriptor_t command_buffer;    // PM4 commands
} hsa_ven_amd_aqlprofile_profile_t;

//
// AQL packets populating methods:
// The helper methods to populate provided by the application START and
// STOP AQL packets which the application is required to submit before and
// after profiled GPU task packets respectively.
//
// AQL Vendor Specific packet which carries a PM4 command
typedef struct {
  uint16_t header;
  uint16_t pm4_command[27];
  hsa_signal_t completion_signal;
} hsa_ext_amd_aql_pm4_packet_t;

// Method to populate the provided AQL packet with profiling start commands
// Only 'pm4_command' fields of the packet are set and the application
// is responsible to set Vendor Specific header type a completion signal
hsa_status_t hsa_ven_amd_aqlprofile_start(
    hsa_ven_amd_aqlprofile_profile_t* profile,        // [in/out] profile contex object
    hsa_ext_amd_aql_pm4_packet_t* aql_start_packet);  // [out] profile start AQL packet

// Method to populate the provided AQL packet with profiling stop commands
// Only 'pm4_command' fields of the packet are set and the application
// is responsible to set Vendor Specific header type and a completion signal
hsa_status_t hsa_ven_amd_aqlprofile_stop(
    const hsa_ven_amd_aqlprofile_profile_t* profile,  // [in] profile contex object
    hsa_ext_amd_aql_pm4_packet_t* aql_stop_packet);   // [out] profile stop AQL packet

// Method to populate the provided AQL packet with profiling read commands
// Only 'pm4_command' fields of the packet are set and the application
// is responsible to set Vendor Specific header type and a completion signal
hsa_status_t hsa_ven_amd_aqlprofile_read(
    const hsa_ven_amd_aqlprofile_profile_t* profile,  // [in] profile contex object
    hsa_ext_amd_aql_pm4_packet_t* aql_read_packet);   // [out] profile stop AQL packet

// Legacy devices, PM4 profiling packet size
const unsigned HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE = 192;
// Legacy devices, converting the profiling AQL packet to PM4 packet blob
hsa_status_t hsa_ven_amd_aqlprofile_legacy_get_pm4(
    const hsa_ext_amd_aql_pm4_packet_t* aql_packet,  // [in] AQL packet
    void* data);                                     // [out] PM4 packet blob

// Method to add a marker (correlation ID) into the ATT buffer.
hsa_status_t hsa_ven_amd_aqlprofile_att_marker(
    hsa_ven_amd_aqlprofile_profile_t* profile,            // [in/out] profile contex object
    hsa_ext_amd_aql_pm4_packet_t* aql_marker_packet,      // [out] profile marker AQL packet
    uint32_t data,                                        // [in] Data to be inserted
    hsa_ven_amd_aqlprofile_att_marker_channel_t channel); // [in] Comm channel

//
// Get profile info:
// Generic method for getting various profile info including profile buffers
// attributes like the command buffer size and the profiling PMC results.
// It’s implied that all counters are 64bit values.
//
// Profile generic output data:
typedef struct {
  uint32_t sample_id;  // PMC sample or trace buffer index
  union {
    struct {
      hsa_ven_amd_aqlprofile_event_t event;  // PMC event
      uint64_t result;                       // PMC result
    } pmc_data;
    hsa_ven_amd_aqlprofile_descriptor_t trace_data;  // Trace output data descriptor
  };
} hsa_ven_amd_aqlprofile_info_data_t;

// ID query type
typedef struct {
  const char* name;
  uint32_t id;
  uint32_t instance_count;
} hsa_ven_amd_aqlprofile_id_query_t;

// Profile attributes
typedef enum {
  HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE = 0,  // get_info returns uint32_t value
  HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE = 1,        // get_info returns uint32_t value
  HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA = 2,             // get_info returns PMC uint64_t value
                                                        // in info_data object
  HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA = 3,           // get_info returns trace buffer ptr/size
                                                        // in info_data object
  HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS = 4,       // get_info returns number of block counter
  HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID = 5,             // get_info returns block id, instances
                                                        // by name string using _id_query_t
  HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD = 6,           // get_info returns size/pointer for
                                                        // counters enable command buffer
  HSA_VEN_AMD_AQLPROFILE_INFO_DISABLE_CMD = 7,          // get_info returns size/pointer for
                                                        // counters disable command buffer
} hsa_ven_amd_aqlprofile_info_type_t;


// Definition of output data iterator callback
typedef hsa_status_t (*hsa_ven_amd_aqlprofile_data_callback_t)(
    hsa_ven_amd_aqlprofile_info_type_t info_type,   // [in] data type, PMC or trace data
    hsa_ven_amd_aqlprofile_info_data_t* info_data,  // [in] info_data object
    void* callback_data);                           // [in/out] data passed to the callback

// Method for getting the profile info
hsa_status_t hsa_ven_amd_aqlprofile_get_info(
    const hsa_ven_amd_aqlprofile_profile_t* profile,  // [in] profile context object
    hsa_ven_amd_aqlprofile_info_type_t attribute,     // [in] requested profile attribute
    void* value);                                     // [in/out] returned value

// Method for iterating the events output data
hsa_status_t hsa_ven_amd_aqlprofile_iterate_data(
    const hsa_ven_amd_aqlprofile_profile_t* profile,  // [in] profile context object
    hsa_ven_amd_aqlprofile_data_callback_t callback,  // [in] callback to iterate the output data
    void* data);                                      // [in/out] data passed to the callback

// Return error string
hsa_status_t hsa_ven_amd_aqlprofile_error_string(
    const char** str);  // [out] pointer on the error string

/**
 * @brief Callback for iteration of all possible event coordinate IDs and coordinate names.
*/
typedef hsa_status_t(*hsa_ven_amd_aqlprofile_eventname_callback_t)(int id, const char* name);
/**
 * @brief Iterate over all possible event coordinate IDs and their names.
*/
hsa_status_t hsa_ven_amd_aqlprofile_iterate_event_ids(hsa_ven_amd_aqlprofile_eventname_callback_t);

/**
 * @brief Iterate over all event coordinates for a given agent_t and event_t.
 * @param position A counting sequence indicating callback number.
 * @param id Coordinate ID as in _iterate_event_ids.
 * @param extent Coordinate extent indicating maximum allowed instances.
 * @param coordinate The coordinate, in the range [0,extent-1].
 * @param name Coordinate name as in _iterate_event_ids.
 * @param userdata Userdata returned from _iterate_event_coord function.
*/
typedef hsa_status_t(*hsa_ven_amd_aqlprofile_coordinate_callback_t)(
  int position,
  int id,
  int extent,
  int coordinate,
  const char* name,
  void* userdata
);

/**
 * @brief Iterate over all event coordinates for a given agent_t and event_t.
 * @param[in] agent HSA agent.
 * @param[in] event The event ID and block ID to iterate for.
 * @param[in] sample_id aqlprofile_info_data_t.sample_id returned from _aqlprofile_iterate_data.
 * @param[in] callback Callback function to return the coordinates.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
*/
hsa_status_t hsa_ven_amd_aqlprofile_iterate_event_coord(
  hsa_agent_t agent,
  hsa_ven_amd_aqlprofile_event_t event,
  uint32_t sample_id,
  hsa_ven_amd_aqlprofile_coordinate_callback_t callback,
  void* userdata
);

/**
 * @brief Extension version.
 */
#define hsa_ven_amd_aqlprofile_VERSION_MAJOR 1
#define hsa_ven_amd_aqlprofile_LIB(suff) "libhsa-amd-aqlprofile" suff ".so"

#ifdef HSA_LARGE_MODEL
static const char kAqlProfileLib[] = hsa_ven_amd_aqlprofile_LIB("64");
#else
static const char kAqlProfileLib[] = hsa_ven_amd_aqlprofile_LIB("");
#endif

/**
 * @brief Extension function table.
 */
typedef struct hsa_ven_amd_aqlprofile_1_00_pfn_s {
  uint32_t (*hsa_ven_amd_aqlprofile_version_major)();
  uint32_t (*hsa_ven_amd_aqlprofile_version_minor)();

  hsa_status_t (*hsa_ven_amd_aqlprofile_error_string)(
      const char** str);

  hsa_status_t (*hsa_ven_amd_aqlprofile_validate_event)(
      hsa_agent_t agent,
      const hsa_ven_amd_aqlprofile_event_t* event,
      bool* result);

  hsa_status_t (*hsa_ven_amd_aqlprofile_start)(
      hsa_ven_amd_aqlprofile_profile_t* profile,
      hsa_ext_amd_aql_pm4_packet_t* aql_start_packet);

  hsa_status_t (*hsa_ven_amd_aqlprofile_stop)(
      const hsa_ven_amd_aqlprofile_profile_t* profile,
      hsa_ext_amd_aql_pm4_packet_t* aql_stop_packet);

  hsa_status_t (*hsa_ven_amd_aqlprofile_read)(
      const hsa_ven_amd_aqlprofile_profile_t* profile,
      hsa_ext_amd_aql_pm4_packet_t* aql_read_packet);

  hsa_status_t (*hsa_ven_amd_aqlprofile_legacy_get_pm4)(
      const hsa_ext_amd_aql_pm4_packet_t* aql_packet,
      void* data);

  hsa_status_t (*hsa_ven_amd_aqlprofile_get_info)(
      const hsa_ven_amd_aqlprofile_profile_t* profile,
      hsa_ven_amd_aqlprofile_info_type_t attribute,
      void* value);

  hsa_status_t (*hsa_ven_amd_aqlprofile_iterate_data)(
      const hsa_ven_amd_aqlprofile_profile_t* profile,
      hsa_ven_amd_aqlprofile_data_callback_t callback,
      void* data);

  hsa_status_t (*hsa_ven_amd_aqlprofile_iterate_event_ids)(
      hsa_ven_amd_aqlprofile_eventname_callback_t
  );

  hsa_status_t (*hsa_ven_amd_aqlprofile_iterate_event_coord)(
      hsa_agent_t agent,
      hsa_ven_amd_aqlprofile_event_t event,
      uint32_t sample_id,
      hsa_ven_amd_aqlprofile_coordinate_callback_t callback,
      void* userdata
  );

  hsa_status_t (*hsa_ven_amd_aqlprofile_att_marker)(
      hsa_ven_amd_aqlprofile_profile_t* profile,
      hsa_ext_amd_aql_pm4_packet_t* aql_packet,
      uint32_t data,
      hsa_ven_amd_aqlprofile_att_marker_channel_t channel
  );
} hsa_ven_amd_aqlprofile_1_00_pfn_t;

typedef hsa_ven_amd_aqlprofile_1_00_pfn_t hsa_ven_amd_aqlprofile_pfn_t;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // OPENSRC_HSA_RUNTIME_INC_HSA_VEN_AMD_AQLPROFILE_H_
