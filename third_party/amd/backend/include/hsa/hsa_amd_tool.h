#ifndef HSA_RUNTIME_AMD_TOOL_EVENTS_H_
#define HSA_RUNTIME_AMD_TOOL_EVENTS_H_

// Insert license header

#include <stddef.h>
#include <stdint.h>
#include "hsa.h"


typedef enum {
  HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_NONE = 0,
  HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_USE_ONCE =
      (1 << 0),  // This scratch allocation is only valid for 1 dispatch.
  HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT =
      (1 << 1),  // Used alternate scratch instead of main scratch
} hsa_amd_event_scratch_alloc_flag_t;

typedef enum {
  HSA_AMD_TOOL_EVENT_MIN = 0,

  // Scratch memory tracking
  HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_START,
  HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_END,
  HSA_AMD_TOOL_EVENT_SCRATCH_FREE_START,
  HSA_AMD_TOOL_EVENT_SCRATCH_FREE_END,
  HSA_AMD_TOOL_EVENT_SCRATCH_ASYNC_RECLAIM_START,
  HSA_AMD_TOOL_EVENT_SCRATCH_ASYNC_RECLAIM_END,

  // Add new events above ^
  HSA_AMD_TOOL_EVENT_MAX
} hsa_amd_tool_event_kind_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
} hsa_amd_tool_event_none_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
  uint64_t dispatch_id;  // Dispatch ID of the AQL packet that needs more scratch memory
} hsa_amd_event_scratch_alloc_start_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
  uint64_t dispatch_id;  // Dispatch ID of the AQL packet that needs more scratch memory
  size_t size;           // Amount of scratch allocated - in bytes
  size_t num_slots;      // limit of number of waves
} hsa_amd_event_scratch_alloc_end_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
} hsa_amd_event_scratch_free_start_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
} hsa_amd_event_scratch_free_end_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
} hsa_amd_event_scratch_async_reclaim_start_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
} hsa_amd_event_scratch_async_reclaim_end_t;

typedef union {
  const hsa_amd_tool_event_none_t* none;
  const hsa_amd_event_scratch_alloc_start_t* scratch_alloc_start;
  const hsa_amd_event_scratch_alloc_end_t* scratch_alloc_end;
  const hsa_amd_event_scratch_free_start_t* scratch_free_start;
  const hsa_amd_event_scratch_free_end_t* scratch_free_end;
  const hsa_amd_event_scratch_async_reclaim_start_t* scratch_async_reclaim_start;
  const hsa_amd_event_scratch_async_reclaim_end_t* scratch_async_reclaim_end;
} hsa_amd_tool_event_t;

typedef hsa_status_t (*hsa_amd_tool_event)(hsa_amd_tool_event_t);


#endif