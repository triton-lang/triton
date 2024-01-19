#pragma once

#include <stdint.h>

#include <hc_defines.h>

#define GRID_LAUNCH_VERSION 20

// Extern definitions
namespace hc{
class completion_future;
class accelerator_view;
}


// 3 dim structure for groups and grids.
typedef struct gl_dim3
{
  int x,y,z;
  gl_dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};
} gl_dim3;

typedef enum gl_barrier_bit {
    barrier_bit_queue_default,
    barrier_bit_none,
    barrier_bit_wait,
} gl_barrier_bit;


// grid_launch_parm contains information used to launch the kernel.
typedef struct grid_launch_parm
{
  //! Grid dimensions
  gl_dim3      grid_dim;

  //! Group dimensions
  gl_dim3      group_dim;

  //! Amount of dynamic group memory to use with the kernel launch.
  //! This memory is in addition to the amount used statically in the kernel.
  unsigned int  dynamic_group_mem_bytes;

  //! Control setting of barrier bit on per-packet basis:
  //! See gl_barrier_bit description.  
  //! Placeholder, is not used to control packet dispatch yet
  enum gl_barrier_bit barrier_bit;

  //! Value of packet fences to apply to launch.
  //! The correspond to the value of bits 9:14 in the AQL packet,
  //! see HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE and hsa_fence_scope_t.
  unsigned int  launch_fence;

  //! Pointer to the accelerator_view where the kernel should execute.
  //! If NULL, the default view on the default accelerator is used.
  hc::accelerator_view  *av;

  //! Pointer to the completion_future used to track the status of the command.
  //! If NULL, the command does not write status.  In this case, 
  //! synchronization can be enforced with queue-level waits or 
  //! waiting on younger commands.
  hc::completion_future *cf;

  grid_launch_parm() = default;
} grid_launch_parm;


extern void init_grid_launch(grid_launch_parm *gl);
