/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  amd_detail/amd_hip_cooperative_groups.h
 *
 *  \brief Device side implementation of `Cooperative Group` feature.
 *
 *  Defines new types and device API wrappers related to `Cooperative Group`
 *  feature, which the programmer can directly use in his kernel(s) in order to
 *  make use of this feature.
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/hip_cooperative_groups_helper.h>
#endif

namespace cooperative_groups {

/** \brief The base type of all cooperative group types.
 *
 *  \details Holds the key properties of a constructed cooperative group types
 *           object, like the group type, its size, etc.
 *
 *  \note  Cooperative groups feature is implemented on Linux, under development
 *         on Microsoft Windows.
 */
class thread_group {
 protected:
  __hip_uint32_t _type;         //! Type of the thread_group.
  __hip_uint32_t _num_threads;  //! Total number of threads in the thread_group.
  __hip_uint64_t _mask;         //! Lanemask for coalesced and tiled partitioned group types,
                                //! LSB represents lane 0, and MSB represents lane 63

  //! Construct a thread group, and set thread group type and other essential
  //! thread group properties. This generic thread group is directly constructed
  //! only when the group is supposed to contain only the calling thread
  //! (through the API - `this_thread()`), and in all other cases, this thread
  //! group object is a sub-object of some other derived thread group object.
  __CG_QUALIFIER__ thread_group(internal::group_type type,
                                __hip_uint32_t num_threads = static_cast<__hip_uint64_t>(0),
                                __hip_uint64_t mask = static_cast<__hip_uint64_t>(0)) {
    _type = type;
    _num_threads = num_threads;
    _mask = mask;
  }

  struct _tiled_info {
    bool is_tiled;
    unsigned int num_threads;
    unsigned int meta_group_rank;
    unsigned int meta_group_size;
  };

  struct _coalesced_info {
    lane_mask member_mask;
    unsigned int num_threads;
    struct _tiled_info tiled_info;
  } coalesced_info;

  friend __CG_QUALIFIER__ thread_group this_thread();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend class thread_block;

 public:
  //! Total number of threads in the thread_group, and this serves the purpose
  //! for all derived cooperative group types because their `num_threads` is directly
  //! saved during the construction.
  __CG_QUALIFIER__ __hip_uint32_t num_threads() const { return _num_threads; }
  //! Total number of threads in the group (alias of num_threads())
  __CG_QUALIFIER__ __hip_uint32_t size() const { return num_threads(); }
  //! Returns the type of the group.
  __CG_QUALIFIER__ unsigned int cg_type() const { return _type; }
  //! Rank of the calling thread within [0, \link num_threads() num_threads() \endlink).
  __CG_QUALIFIER__ __hip_uint32_t thread_rank() const;
  //! Returns true if the group has not violated any API constraints.
  __CG_QUALIFIER__ bool is_valid() const;

  /** \brief   Synchronizes the threads in the group.
   *
   *  \details Causes all threads in the group to wait at this synchronization point,
   *           and for all shared and global memory accesses by the threads to complete,
   *           before running synchronization. This guarantees the visibility of accessed data
   *           for all threads in the group.
   *
   * \note     There are potential read-after-write (RAW), write-after-read (WAR), or
   *           write-after-write (WAW) hazards, when threads in the group access the
   *           same addresses in shared or global memory. The data hazards can
   *           be avoided with synchronization of the group.
   */
  __CG_QUALIFIER__ void sync() const;
};
/**
 *  @defgroup CooperativeG Cooperative Groups
 *  @ingroup API
 *  @{
 *  This section describes the cooperative groups functions of HIP runtime API.
 *
 *  The cooperative groups provides flexible thread parallel programming algorithms, threads
 *  cooperate and share data to perform collective computations.
 *
 *  \note  Cooperative groups feature is implemented on Linux, under development
 *         on Microsoft Windows.
 *
 */

/** \brief   The multi-grid cooperative group type.
 *
 *  \details Represents an inter-device cooperative group type, where the
 *           participating threads within the group span across multiple
 *           devices, running the (same) kernel on these devices.
 *  \note    The multi-grid cooperative group type is implemented on Linux,
 *           under development on Microsoft Windows.
 */
class multi_grid_group : public thread_group {
  //! Only these friend functions are allowed to construct an object of this class
  //! and access its resources.
  friend __CG_QUALIFIER__ multi_grid_group this_multi_grid();

 protected:
  //! Construct multi-grid thread group (through the API this_multi_grid())
  explicit __CG_QUALIFIER__ multi_grid_group(__hip_uint32_t size)
      : thread_group(internal::cg_multi_grid, size) {}

 public:
  //! Number of invocations participating in this multi-grid group. In other
  //! words, the number of GPUs.
  __CG_QUALIFIER__ __hip_uint32_t num_grids() { return internal::multi_grid::num_grids(); }

  //! Rank of this invocation. In other words, an ID number within the range
  //! [0, num_grids()) of the GPU that kernel is running on.
  __CG_QUALIFIER__ __hip_uint32_t grid_rank() { return internal::multi_grid::grid_rank(); }
  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ __hip_uint32_t thread_rank() const {
    return internal::multi_grid::thread_rank();
  }
  //! @copydoc thread_group::is_valid
  __CG_QUALIFIER__ bool is_valid() const { return internal::multi_grid::is_valid(); }
  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync() const { internal::multi_grid::sync(); }
};

/** \addtogroup CooperativeGConstruct Construct functions of Cooperative groups
 * \ingroup CooperativeG
 *  @{ */

/** \brief   User-exposed API interface to construct grid cooperative group type
 *           object - `multi_grid_group`.
 *
 *  \details User is not allowed to construct an object of type
 *           `multi_grid_group` directly. Instead, they should construct it
 *           through this API function.
 *  \note    This multi-grid cooperative API type is implemented on Linux, under
 *           development on Microsoft Windows.
 */
__CG_QUALIFIER__ multi_grid_group this_multi_grid() {
  return multi_grid_group(internal::multi_grid::num_threads());
}
// Doxygen end group CooperativeGConstruct
/** @} */

/** \brief   The grid cooperative group type.
 *
 *  \details Represents an inter-workgroup cooperative group type, where the
 *           participating threads within the group spans across multiple
 *           workgroups running the (same) kernel on the same device.
 *  \note    This is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
class grid_group : public thread_group {
  //! Only these friend functions are allowed to construct an object of this class
  //! and access its resources.
  friend __CG_QUALIFIER__ grid_group this_grid();

 protected:
  //! Construct grid thread group (through the API this_grid())
  explicit __CG_QUALIFIER__ grid_group(__hip_uint32_t size)
      : thread_group(internal::cg_grid, size) {}

 public:
  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ __hip_uint32_t thread_rank() const { return internal::grid::thread_rank(); }
  //! @copydoc thread_group::is_valid
  __CG_QUALIFIER__ bool is_valid() const { return internal::grid::is_valid(); }
  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync() const { internal::grid::sync(); }
  __CG_QUALIFIER__ dim3 group_dim() const { return internal::grid::grid_dim(); }
};

/** \ingroup CooperativeGConstruct
 *  \brief   User-exposed API interface to construct grid cooperative group type
 *           object - `grid_group`.
 *
 *  \details User is not allowed to construct an object of type `grid_group`
 *           directly. Instead, they should construct it through this
 *           API function.
 *  \note    This function is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
__CG_QUALIFIER__ grid_group this_grid() { return grid_group(internal::grid::num_threads()); }

/** \brief   The workgroup (thread-block in CUDA terminology) cooperative group
 *           type.
 *
 *  \details Represents an intra-workgroup cooperative group type, where the
 *           participating threads within the group are the same threads that
 *           participated in the currently executing `workgroup`.
 *  \note    This function is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
class thread_block : public thread_group {
  //! Only these friend functions are allowed to construct an object of thi
  //! class and access its resources
  friend __CG_QUALIFIER__ thread_block this_thread_block();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_block& parent,
                                                       unsigned int tile_size);

 protected:
  // Construct a workgroup thread group (through the API this_thread_block())
  explicit __CG_QUALIFIER__ thread_block(__hip_uint32_t size)
      : thread_group(internal::cg_workgroup, size) {}

  __CG_QUALIFIER__ thread_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);
    // Invalid tile size, assert
    if (!tile_size || (tile_size > warpSize) || !pow2) {
      __hip_assert(false && "invalid tile size");
    }

    auto block_size = num_threads();
    auto rank = thread_rank();
    auto partitions = (block_size + tile_size - 1) / tile_size;
    auto tail = (partitions * tile_size) - block_size;
    auto partition_size = tile_size - tail * (rank >= (partitions - 1) * tile_size);
    thread_group tiledGroup = thread_group(internal::cg_tiled_group, partition_size);

    tiledGroup.coalesced_info.tiled_info.num_threads = tile_size;
    tiledGroup.coalesced_info.tiled_info.is_tiled = true;
    tiledGroup.coalesced_info.tiled_info.meta_group_rank = rank / tile_size;
    tiledGroup.coalesced_info.tiled_info.meta_group_size = partitions;
    return tiledGroup;
  }

 public:
  //! Returns 3-dimensional block index within the grid.
  __CG_STATIC_QUALIFIER__ dim3 group_index() { return internal::workgroup::group_index(); }
  //! Returns 3-dimensional thread index within the block.
  __CG_STATIC_QUALIFIER__ dim3 thread_index() { return internal::workgroup::thread_index(); }
  //! @copydoc thread_group::thread_rank
  __CG_STATIC_QUALIFIER__ __hip_uint32_t thread_rank() {
    return internal::workgroup::thread_rank();
  }
  //! @copydoc thread_group::num_threads
  __CG_STATIC_QUALIFIER__ __hip_uint32_t num_threads() {
    return internal::workgroup::num_threads();
  }
  //! @copydoc thread_group::size
  __CG_STATIC_QUALIFIER__ __hip_uint32_t size() { return num_threads(); }
  //! @copydoc thread_group::is_valid
  __CG_STATIC_QUALIFIER__ bool is_valid() { return internal::workgroup::is_valid(); }
  //! @copydoc thread_group::sync
  __CG_STATIC_QUALIFIER__ void sync() { internal::workgroup::sync(); }
  //! Returns the group dimensions.
  __CG_QUALIFIER__ dim3 group_dim() { return internal::workgroup::block_dim(); }
};

/** \ingroup CooperativeGConstruct
 *  \brief   User-exposed API interface to construct workgroup cooperative
 *           group type object - `thread_block`.
 *
 *  \details User is not allowed to construct an object of type `thread_block`
 *           directly. Instead, they should construct it through this API
 *           function.
 *  \note    This function is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
__CG_QUALIFIER__ thread_block this_thread_block() {
  return thread_block(internal::workgroup::num_threads());
}

/** \brief   The tiled_group cooperative group type
 *
 *  \details Represents one tiled thread group in a wavefront.
 *           This group type also supports sub-wave level intrinsics.
 *  \note    This is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
class tiled_group : public thread_group {
 private:
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ tiled_group tiled_partition(const tiled_group& parent,
                                                      unsigned int tile_size);

  __CG_QUALIFIER__ tiled_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);

    if (!tile_size || (tile_size > warpSize) || !pow2) {
      __hip_assert(false && "invalid tile size");
    }

    if (num_threads() <= tile_size) {
      return *this;
    }

    tiled_group tiledGroup = tiled_group(tile_size);
    tiledGroup.coalesced_info.tiled_info.is_tiled = true;
    return tiledGroup;
  }

 protected:
  explicit __CG_QUALIFIER__ tiled_group(unsigned int tileSize)
      : thread_group(internal::cg_tiled_group, tileSize) {
    coalesced_info.tiled_info.num_threads = tileSize;
    coalesced_info.tiled_info.is_tiled = true;
  }

 public:
  //! @copydoc thread_group::num_threads
  __CG_QUALIFIER__ unsigned int num_threads() const {
    return (coalesced_info.tiled_info.num_threads);
  }

  //! @copydoc thread_group::size
  __CG_QUALIFIER__ unsigned int size() const { return num_threads(); }

  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ unsigned int thread_rank() const {
    return (internal::workgroup::thread_rank() & (coalesced_info.tiled_info.num_threads - 1));
  }

  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync() const { internal::tiled_group::sync(); }
};

template <unsigned int size, class ParentCGTy> class thread_block_tile;

/** \brief   The coalesced_group cooperative group type
 *
 *  \details Represents an active thread group in a wavefront.
 *           This group type also supports sub-wave level intrinsics.
 *  \note    This is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
class coalesced_group : public thread_group {
 private:
  friend __CG_QUALIFIER__ coalesced_group coalesced_threads();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ coalesced_group tiled_partition(const coalesced_group& parent,
                                                          unsigned int tile_size);
  friend __CG_QUALIFIER__ coalesced_group binary_partition(const coalesced_group& cgrp, bool pred);
  template <unsigned int fsize, class fparent> friend __CG_QUALIFIER__ coalesced_group
  binary_partition(const thread_block_tile<fsize, fparent>& tgrp, bool pred);

  __CG_QUALIFIER__ coalesced_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);

    if (!tile_size || !pow2) {
      return coalesced_group(0);
    }

    // If a tiled group is passed to be partitioned further into a coalesced_group.
    // prepare a mask for further partitioning it so that it stays coalesced.
    if (coalesced_info.tiled_info.is_tiled) {
      unsigned int base_offset = (thread_rank() & (~(tile_size - 1)));
      unsigned int masklength =
          min(static_cast<unsigned int>(num_threads()) - base_offset, tile_size);
      lane_mask full_mask = (static_cast<int>(warpSize) == 32)
                                ? static_cast<lane_mask>((1u << 32) - 1)
                                : static_cast<lane_mask>(-1ull);
      lane_mask member_mask = full_mask >> (warpSize - masklength);

      member_mask <<= (__lane_id() & ~(tile_size - 1));
      coalesced_group coalesced_tile = coalesced_group(member_mask);
      coalesced_tile.coalesced_info.tiled_info.is_tiled = true;
      coalesced_tile.coalesced_info.tiled_info.meta_group_rank = thread_rank() / tile_size;
      coalesced_tile.coalesced_info.tiled_info.meta_group_size = num_threads() / tile_size;
      return coalesced_tile;
    }
    // Here the parent coalesced_group is not partitioned.
    else {
      lane_mask member_mask = 0;
      unsigned int tile_rank = 0;
      int lanes_to_skip = ((thread_rank()) / tile_size) * tile_size;

      for (unsigned int i = 0; i < warpSize; i++) {
        lane_mask active = coalesced_info.member_mask & (static_cast<lane_mask>(1) << i);
        // Make sure the lane is active
        if (active) {
          if (lanes_to_skip <= 0 && tile_rank < tile_size) {
            // Prepare a member_mask that is appropriate for a tile
            member_mask |= active;
            tile_rank++;
          }
          lanes_to_skip--;
        }
      }
      coalesced_group coalesced_tile = coalesced_group(member_mask);
      coalesced_tile.coalesced_info.tiled_info.meta_group_rank = thread_rank() / tile_size;
      coalesced_tile.coalesced_info.tiled_info.meta_group_size =
          (num_threads() + tile_size - 1) / tile_size;
      return coalesced_tile;
    }
    return coalesced_group(0);
  }

 protected:
  // Constructor
  explicit __CG_QUALIFIER__ coalesced_group(lane_mask member_mask)
      : thread_group(internal::cg_coalesced_group) {
    coalesced_info.member_mask = member_mask;  // Which threads are active
    coalesced_info.num_threads =
        __popcll(coalesced_info.member_mask);    // How many threads are active
    coalesced_info.tiled_info.is_tiled = false;  // Not a partitioned group
    coalesced_info.tiled_info.meta_group_rank = 0;
    coalesced_info.tiled_info.meta_group_size = 1;
  }

 public:
  //! @copydoc thread_group::num_threads
  __CG_QUALIFIER__ unsigned int num_threads() const { return coalesced_info.num_threads; }

  //! @copydoc thread_group::size
  __CG_QUALIFIER__ unsigned int size() const { return num_threads(); }

  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ unsigned int thread_rank() const {
    return internal::coalesced_group::masked_bit_count(coalesced_info.member_mask);
  }

  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync() const { internal::coalesced_group::sync(); }

  //! Returns the linear rank of the group within the set of tiles partitioned
  //! from a parent group (bounded by meta_group_size).
  __CG_QUALIFIER__ unsigned int meta_group_rank() const {
    return coalesced_info.tiled_info.meta_group_rank;
  }

  //! Returns the number of groups created when the parent group was partitioned.
  __CG_QUALIFIER__ unsigned int meta_group_size() const {
    return coalesced_info.tiled_info.meta_group_size;
  }

  /** \brief Shuffle operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle operation is a direct copy of ``var`` from ``srcRank``
   *           thread ID of group.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy. Only the srcRank thread ID of
   *                  group is copied to other threads.
   *  \param srcRank [in] The source thread ID of the group for copy.
   */
  template <class T> __CG_QUALIFIER__ T shfl(T var, int srcRank) const {
    srcRank = srcRank % static_cast<int>(num_threads());

    int lane = (num_threads() == warpSize) ? srcRank
               : (static_cast<int>(warpSize) == 64)
                   ? __fns64(coalesced_info.member_mask, 0, (srcRank + 1))
                   : __fns32(coalesced_info.member_mask, 0, (srcRank + 1));

    return __shfl(var, lane, warpSize);
  }

  /** \brief Shuffle down operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle down operation is copy of ``var`` from thread with
   *           thread ID of group relative higher with ``lane_delta`` to caller
   *           thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param lane_delta [in] The lane_delta is the relative thread ID difference
   *                         between caller thread ID and source of copy thread
   *                         ID. sourceID = (threadID + lane_delta) % size()
   */
  template <class T> __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const {
    // Note: The cuda implementation appears to use the remainder of lane_delta
    // and WARP_SIZE as the shift value rather than lane_delta itself.
    // This is not described in the documentation and is not done here.

    if (num_threads() == warpSize) {
      return __shfl_down(var, lane_delta, warpSize);
    }

    int lane;
    if (static_cast<int>(warpSize) == 64) {
      lane = __fns64(coalesced_info.member_mask, __lane_id(), lane_delta + 1);
    } else {
      lane = __fns32(coalesced_info.member_mask, __lane_id(), lane_delta + 1);
    }

    if (lane == -1) {
      lane = __lane_id();
    }

    return __shfl(var, lane, warpSize);
  }

  /** \brief Shuffle up operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle up operation is copy of ``var`` from thread with
   *           thread ID of group relative lower with ``lane_delta`` to caller
   *           thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param lane_delta [in] The lane_delta is the relative thread ID difference
   *                         between caller thread ID and source of copy thread
   *                         ID. sourceID = (threadID - lane_delta) % size()
   */
  template <class T> __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const {
    // Note: The cuda implementation appears to use the remainder of lane_delta
    // and WARP_SIZE as the shift value rather than lane_delta itself.
    // This is not described in the documentation and is not done here.

    if (num_threads() == warpSize) {
      return __shfl_up(var, lane_delta, warpSize);
    }

    int lane;
    if (static_cast<int>(warpSize) == 64) {
      lane = __fns64(coalesced_info.member_mask, __lane_id(), -(lane_delta + 1));
    } else if (static_cast<int>(warpSize) == 32) {
      lane = __fns32(coalesced_info.member_mask, __lane_id(), -(lane_delta + 1));
    }

    if (lane == -1) {
      lane = __lane_id();
    }

    return __shfl(var, lane, warpSize);
  }
#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)

  /** \brief Ballot function on group level.
   *
   *  \details Returns a bit mask with the Nth bit set to one if the specified
   *           predicate evaluates as true on the Nth thread.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ unsigned long long ballot(int pred) const {
    return internal::helper::adjust_mask(
        coalesced_info.member_mask,
        __ballot_sync<unsigned long long>(coalesced_info.member_mask, pred));
  }

  /** \brief Any function on group level.
   *
   *  \details Returns non-zero if a predicate evaluates true for any threads.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ int any(int pred) const {
    return __any_sync(static_cast<unsigned long long>(coalesced_info.member_mask), pred);
  }

  /** \brief All function on group level.
   *
   *  \details Returns non-zero if a predicate evaluates true for all threads.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ int all(int pred) const {
    return __all_sync(static_cast<unsigned long long>(coalesced_info.member_mask), pred);
  }

  /** \brief Match any function on group level.
   *
   *  \details Returns a bit mask containing a 1-bit for every participating
   *           thread if that thread has the same value in ``value`` as the
   *           caller thread.
   *
   *  \param value [in] The value to examine on the current thread in group.
   */
  template <typename T> __CG_QUALIFIER__ unsigned long long match_any(T value) const {
    return internal::helper::adjust_mask(
        coalesced_info.member_mask,
        __match_any_sync(static_cast<unsigned long long>(coalesced_info.member_mask), value));
  }

  /** \brief Match all function on group level.
   *
   *  \details Returns a bit mask containing a 1-bit for every participating
   *           thread if they all have the same value in ``value`` as the caller
   *           thread. The predicate ``pred`` is set to true if all
   *           participating threads have the same value in ``value``.
   *
   *  \param value [in] The value to examine on the current thread in group.
   *  \param pred [out] The predicate is set to true if all participating
   *                    threads in the thread group have the same value.
   */
  template <typename T> __CG_QUALIFIER__ unsigned long long match_all(T value, int& pred) const {
    return internal::helper::adjust_mask(
        coalesced_info.member_mask,
        __match_all_sync(static_cast<unsigned long long>(coalesced_info.member_mask), value,
                         &pred));
  }
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS
};

/** \ingroup CooperativeGConstruct
 *  \brief   User-exposed API to create coalesced groups.
 *
 *  \details A collective operation that groups all active lanes into a new
 *           thread group.
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
__CG_QUALIFIER__ coalesced_group coalesced_threads() {
  return cooperative_groups::coalesced_group(__builtin_amdgcn_read_exec());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/**
 *  Implementation of all publicly exposed base class APIs
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
__CG_QUALIFIER__ __hip_uint32_t thread_group::thread_rank() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      return (static_cast<const multi_grid_group*>(this)->thread_rank());
    }
    case internal::cg_grid: {
      return (static_cast<const grid_group*>(this)->thread_rank());
    }
    case internal::cg_workgroup: {
      return (static_cast<const thread_block*>(this)->thread_rank());
    }
    case internal::cg_tiled_group: {
      return (static_cast<const tiled_group*>(this)->thread_rank());
    }
    case internal::cg_coalesced_group: {
      return (static_cast<const coalesced_group*>(this)->thread_rank());
    }
    default: {
      __hip_assert(false && "invalid cooperative group type");
      return -1;
    }
  }
}

/**
 *  Implementation of all publicly exposed thread group API
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
__CG_QUALIFIER__ bool thread_group::is_valid() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      return (static_cast<const multi_grid_group*>(this)->is_valid());
    }
    case internal::cg_grid: {
      return (static_cast<const grid_group*>(this)->is_valid());
    }
    case internal::cg_workgroup: {
      return (static_cast<const thread_block*>(this)->is_valid());
    }
    case internal::cg_tiled_group: {
      return (static_cast<const tiled_group*>(this)->is_valid());
    }
    case internal::cg_coalesced_group: {
      return (static_cast<const coalesced_group*>(this)->is_valid());
    }
    default: {
      __hip_assert(false && "invalid cooperative group type");
      return false;
    }
  }
}

/**
 *  Implementation of all publicly exposed thread group sync API
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
__CG_QUALIFIER__ void thread_group::sync() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      static_cast<const multi_grid_group*>(this)->sync();
      break;
    }
    case internal::cg_grid: {
      static_cast<const grid_group*>(this)->sync();
      break;
    }
    case internal::cg_workgroup: {
      static_cast<const thread_block*>(this)->sync();
      break;
    }
    case internal::cg_tiled_group: {
      static_cast<const tiled_group*>(this)->sync();
      break;
    }
    case internal::cg_coalesced_group: {
      static_cast<const coalesced_group*>(this)->sync();
      break;
    }
    default: {
      __hip_assert(false && "invalid cooperative group type");
    }
  }
}

#endif

/** \addtogroup CooperativeGAPI User-exposed API of Cooperative groups
 * \ingroup CooperativeG
 *  @{ */

/** \brief   Returns the size of the group.
 *
 *  \details Total number of threads in the thread group, and this serves the
 *           purpose for all derived cooperative group types because their
 *           `size` is directly saved during the construction.
 *
 *  \tparam CGTy  The cooperative group class template parameter.
 *  \param g [in] The cooperative group for size returns.
 *
 *  \note    Implementation of publicly exposed `wrapper` API on top of basic
 *           cooperative group type APIs. This function is implemented on Linux
 *           and is under development on Microsoft Windows.
 */
template <class CGTy> __CG_QUALIFIER__ __hip_uint32_t group_size(CGTy const& g) {
  return g.num_threads();
}

/** \brief   Returns the rank of thread of the group.
 *
 *  \details Rank of the calling thread within [0, \link num_threads() num_threads() \endlink).
 *
 *  \tparam CGTy  The cooperative group class template parameter.
 *  \param g [in] The cooperative group for rank returns.
 *
 *  \note    Implementation of publicly exposed `wrapper` API on top of basic
 *           cooperative group type APIs. This function is implemented on Linux
 *           and is under development on Microsoft Windows.
 */
template <class CGTy> __CG_QUALIFIER__ __hip_uint32_t thread_rank(CGTy const& g) {
  return g.thread_rank();
}

/** \brief   Returns true if the group has not violated any API constraints.
 *
 *  \tparam CGTy  The cooperative group class template parameter.
 *  \param g [in] The cooperative group for validity check.
 *
 *  \note    Implementation of publicly exposed `wrapper` API on top of basic
 *           cooperative group type APIs. This function is implemented on Linux
 *           and is under development on Microsoft Windows.
 */
template <class CGTy> __CG_QUALIFIER__ bool is_valid(CGTy const& g) { return g.is_valid(); }

/** \brief   Synchronizes the threads in the group.
 *
 *  \tparam CGTy  The cooperative group class template parameter.
 *  \param g [in] The cooperative group for synchronization.
 *
 *  \note    Implementation of publicly exposed `wrapper` API on top of basic
 *           cooperative group type APIs. This function is implemented on Linux
 *           and is under development on Microsoft Windows.
 */
template <class CGTy> __CG_QUALIFIER__ void sync(CGTy const& g) { g.sync(); }

// Doxygen end group CooperativeGAPI
/** @} */

/**
 * template class tile_base
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
template <unsigned int tileSize> class tile_base {
 protected:
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;

 public:
  //! Rank of the thread within this tile
  _CG_STATIC_CONST_DECL_ unsigned int thread_rank() {
    return (internal::workgroup::thread_rank() & (numThreads - 1));
  }

  //! Number of threads within this tile
  __CG_STATIC_QUALIFIER__ unsigned int num_threads() { return numThreads; }

  //! Legacy member functions
  //! Number of threads within this tile (alias of num_threads())
  __CG_STATIC_QUALIFIER__ unsigned int size() { return num_threads(); }
};

/**
 * template class thread_block_tile_base
 *  \note  This class is implemented on Linux, under development
 *  on Microsoft Windows.
 */
template <unsigned int size> class thread_block_tile_base : public tile_base<size> {
  static_assert(is_valid_tile_size<size>::value,
                "Tile size is either not a power of 2 or greater than the wavefront size");
  using tile_base<size>::numThreads;

  template <unsigned int fsize, class fparent> friend __CG_QUALIFIER__ coalesced_group
  binary_partition(const thread_block_tile<fsize, fparent>& tgrp, bool pred);

#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
  __CG_QUALIFIER__ unsigned long long build_mask() const {
    unsigned long long mask = ~0ull >> (64 - numThreads);
    // thread_rank() gives thread id from 0..thread launch size.
    return mask << (((internal::workgroup::thread_rank() % warpSize) / numThreads) * numThreads);
  }
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS

 public:
  __CG_STATIC_QUALIFIER__ void sync() { internal::tiled_group::sync(); }

  template <class T> __CG_QUALIFIER__ T shfl(T var, int srcRank) const {
    return (__shfl(var, srcRank, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const {
    return (__shfl_down(var, lane_delta, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const {
    return (__shfl_up(var, lane_delta, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_xor(T var, unsigned int laneMask) const {
    return (__shfl_xor(var, laneMask, numThreads));
  }

#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
  __CG_QUALIFIER__ unsigned long long ballot(int pred) const {
    const auto mask = build_mask();
    return internal::helper::adjust_mask(mask, __ballot_sync(mask, pred));
  }

  __CG_QUALIFIER__ int any(int pred) const { return __any_sync(build_mask(), pred); }

  __CG_QUALIFIER__ int all(int pred) const { return __all_sync(build_mask(), pred); }

  template <typename T> __CG_QUALIFIER__ unsigned long long match_any(T value) const {
    const auto mask = build_mask();
    return internal::helper::adjust_mask(mask, __match_any_sync(mask, value));
  }

  template <typename T> __CG_QUALIFIER__ unsigned long long match_all(T value, int& pred) const {
    const auto mask = build_mask();
    return internal::helper::adjust_mask(mask, __match_all_sync(mask, value, &pred));
  }
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS
};

/** \brief   User exposed API that captures the state of the parent group pre-partition
 */
template <unsigned int tileSize, typename ParentCGTy> class parent_group_info {
 public:
  //! Returns the linear rank of the group within the set of tiles partitioned
  //! from a parent group (bounded by meta_group_size)
  __CG_STATIC_QUALIFIER__ unsigned int meta_group_rank() {
    return ParentCGTy::thread_rank() / tileSize;
  }

  //! Returns the number of groups created when the parent group was partitioned.
  __CG_STATIC_QUALIFIER__ unsigned int meta_group_size() {
    return (ParentCGTy::num_threads() + tileSize - 1) / tileSize;
  }
};

/** \brief   Group type - thread_block_tile
 *
 *  \details  Represents one tile of thread group.
 *  \note     This type is implemented on Linux, under development
 *            on Microsoft Windows.
 */
template <unsigned int tileSize, class ParentCGTy> class thread_block_tile_type
    : public thread_block_tile_base<tileSize>,
      public tiled_group,
      public parent_group_info<tileSize, ParentCGTy> {
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;
  typedef thread_block_tile_base<numThreads> tbtBase;

 protected:
  __CG_QUALIFIER__ thread_block_tile_type() : tiled_group(numThreads) {
    coalesced_info.tiled_info.num_threads = numThreads;
    coalesced_info.tiled_info.is_tiled = true;
  }

  __CG_QUALIFIER__ thread_block_tile_type(unsigned int meta_group_rank,
                                          unsigned int meta_group_size)
      : tiled_group(numThreads) {
    coalesced_info.tiled_info.num_threads = numThreads;
    coalesced_info.tiled_info.is_tiled = true;
    coalesced_info.tiled_info.meta_group_rank = meta_group_rank;
    coalesced_info.tiled_info.meta_group_size = meta_group_size;
  }

 public:
  using tbtBase::num_threads;
  using tbtBase::size;
  using tbtBase::sync;
  using tbtBase::thread_rank;
};

// Partial template specialization
template <unsigned int tileSize> class thread_block_tile_type<tileSize, void>
    : public thread_block_tile_base<tileSize>, public tiled_group {
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;

  typedef thread_block_tile_base<numThreads> tbtBase;

 protected:
  __CG_QUALIFIER__ thread_block_tile_type(unsigned int meta_group_rank,
                                          unsigned int meta_group_size)
      : tiled_group(numThreads) {
    coalesced_info.tiled_info.num_threads = numThreads;
    coalesced_info.tiled_info.is_tiled = true;
    coalesced_info.tiled_info.meta_group_rank = meta_group_rank;
    coalesced_info.tiled_info.meta_group_size = meta_group_size;
  }

 public:
  using tbtBase::num_threads;
  using tbtBase::size;
  using tbtBase::sync;
  using tbtBase::thread_rank;

  //! Returns the linear rank of the group within the set of tiles partitioned
  //! from a parent group (bounded by meta_group_size)
  __CG_QUALIFIER__ unsigned int meta_group_rank() const {
    return coalesced_info.tiled_info.meta_group_rank;
  }

  //! Returns the number of groups created when the parent group was partitioned.
  __CG_QUALIFIER__ unsigned int meta_group_size() const {
    return coalesced_info.tiled_info.meta_group_size;
  }
  // Doxygen end group CooperativeG
  /**
   * @}
   */
};

__CG_QUALIFIER__ thread_group this_thread() {
  thread_group g(internal::group_type::cg_coalesced_group, 1, __ockl_activelane_u32());
  return g;
}

/** \ingroup CooperativeGConstruct
 *  \brief   User-exposed API to partition groups.
 *
 *  \details A collective operation that partitions the parent group into a
 *           one-dimensional, row-major, tiling of subgroups.
 */

__CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent, unsigned int tile_size) {
  if (parent.cg_type() == internal::cg_tiled_group) {
    const tiled_group* cg = static_cast<const tiled_group*>(&parent);
    return cg->new_tiled_group(tile_size);
  } else if (parent.cg_type() == internal::cg_coalesced_group) {
    const coalesced_group* cg = static_cast<const coalesced_group*>(&parent);
    return cg->new_tiled_group(tile_size);
  } else {
    const thread_block* tb = static_cast<const thread_block*>(&parent);
    return tb->new_tiled_group(tile_size);
  }
}

// Thread block type overload
__CG_QUALIFIER__ thread_group tiled_partition(const thread_block& parent, unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

__CG_QUALIFIER__ tiled_group tiled_partition(const tiled_group& parent, unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

// If a coalesced group is passed to be partitioned, it should remain coalesced
__CG_QUALIFIER__ coalesced_group tiled_partition(const coalesced_group& parent,
                                                 unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

namespace impl {
template <unsigned int size, class ParentCGTy> class thread_block_tile_internal;

template <unsigned int size, class ParentCGTy> class thread_block_tile_internal
    : public thread_block_tile_type<size, ParentCGTy> {
 protected:
  template <unsigned int tbtSize, class tbtParentT> __CG_QUALIFIER__ thread_block_tile_internal(
      const thread_block_tile_internal<tbtSize, tbtParentT>& g)
      : thread_block_tile_type<size, ParentCGTy>(g.meta_group_rank(), g.meta_group_size()) {}

  __CG_QUALIFIER__ thread_block_tile_internal(const thread_block& g)
      : thread_block_tile_type<size, ParentCGTy>() {}
};
}  // namespace impl

/** \brief    Group type - thread_block_tile
 *
 *  \details  Represents one tiled thread group in a wavefront.
 *            This group type also supports sub-wave level intrinsics.
 *
 *  \note     This type is implemented on Linux, under development
 *            on Microsoft Windows.
 */
template <unsigned int size, class ParentCGTy> class thread_block_tile
    : public impl::thread_block_tile_internal<size, ParentCGTy> {
 protected:
  __CG_QUALIFIER__ thread_block_tile(const ParentCGTy& g)
      : impl::thread_block_tile_internal<size, ParentCGTy>(g) {}

 public:
  __CG_QUALIFIER__ operator thread_block_tile<size, void>() const {
    return thread_block_tile<size, void>(*this);
  }

#ifdef DOXYGEN_SHOULD_INCLUDE_THIS

  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ unsigned int thread_rank() const;

  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync();

  //! Returns the linear rank of the group within the set of tiles partitioned
  //! from a parent group (bounded by meta_group_size)
  __CG_QUALIFIER__ unsigned int meta_group_rank() const;

  //! Returns the number of groups created when the parent group was partitioned.
  __CG_QUALIFIER__ unsigned int meta_group_size() const;

  /** \brief Shuffle operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle operation is a direct copy of ``var`` from ``srcRank``
   *           thread ID of group.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy. Only the srcRank thread ID of
   *                  group is copied to other threads.
   *  \param srcRank [in] The source thread ID of the group for copy.
   */
  template <class T> __CG_QUALIFIER__ T shfl(T var, int srcRank) const;

  /** \brief Shuffle down operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle down operation is copy of ``var`` from thread with
   *           thread ID of group relative higher with ``lane_delta`` to caller
   *           thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param lane_delta [in] The lane_delta is the relative thread ID difference
   *                         between caller thread ID and source of copy thread
   *                         ID. sourceID = (threadID + lane_delta) % size()
   */
  template <class T> __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const;

  /** \brief Shuffle up operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle up operation is copy of ``var`` from thread with
   *           thread ID of group relative lower with ``lane_delta`` to caller
   *           thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param lane_delta [in] The lane_delta is the relative thread ID difference
   *                         between caller thread ID and source of copy thread
   *                         ID. sourceID = (threadID - lane_delta) % size()
   */
  template <class T> __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const;

  /** \brief Shuffle xor operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle xor operation is copy of var from thread with thread ID
   *           of group based on laneMask XOR of the caller thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param laneMask [in] The laneMask is the mask for XOR operation.
   *                       sourceID = threadID ^ laneMask
   */
  template <class T> __CG_QUALIFIER__ T shfl_xor(T var, unsigned int laneMask) const;

  /** \brief Ballot function on group level.
   *
   *  \details Returns a bit mask with the Nth bit set to one if the Nth thread
   *           predicate evaluates true.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ unsigned long long ballot(int pred) const;

  /** \brief Any function on group level.
   *
   *  \details Returns non-zero if a predicate evaluates true for any threads.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ int any(int pred) const;

  /** \brief All function on group level.
   *
   *  \details Returns non-zero if a predicate evaluates true for all threads.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ int all(int pred) const;

  /** \brief Match any function on group level.
   *
   *  \details Returns a bit mask containing a 1-bit for every participating
   *           thread if that thread has the same value in ``value`` as the
   *           caller thread.
   *
   *  \param value [in] The value to examine on the current thread in group.
   */
  template <typename T> __CG_QUALIFIER__ unsigned long long match_any(T value) const;

  /** \brief Match all function on group level.
   *
   *  \details Returns a bit mask containing a 1-bit for every participating
   *           thread if they all have the same value in ``value`` as the caller
   *           thread. The predicate ``pred`` is set to true if all
   *           participating threads have the same value in ``value``.
   *
   *  \param value [in] The value to examine on the current thread in group.
   *  \param pred [out] The predicate is set to true if all participating
   *                    threads in the thread group have the same value.
   */
  template <typename T> __CG_QUALIFIER__ unsigned long long match_all(T value, int& pred) const;

#endif
};

template <unsigned int size> class thread_block_tile<size, void>
    : public impl::thread_block_tile_internal<size, void> {
  template <unsigned int, class ParentCGTy> friend class thread_block_tile;

 protected:
 public:
  template <class ParentCGTy>
  __CG_QUALIFIER__ thread_block_tile(const thread_block_tile<size, ParentCGTy>& g)
      : impl::thread_block_tile_internal<size, void>(g) {}
};

template <unsigned int size, class ParentCGTy = void> class thread_block_tile;

namespace impl {
template <unsigned int size, class ParentCGTy> struct tiled_partition_internal;

template <unsigned int size> struct tiled_partition_internal<size, thread_block>
    : public thread_block_tile<size, thread_block> {
  __CG_QUALIFIER__ tiled_partition_internal(const thread_block& g)
      : thread_block_tile<size, thread_block>(g) {}
};

// ParentCGTy = thread_block_tile<ParentSize, GrandParentCGTy> specialization
template <unsigned int size, unsigned int ParentSize, class GrandParentCGTy>
struct tiled_partition_internal<size, thread_block_tile<ParentSize, GrandParentCGTy> >
    : public thread_block_tile<size, thread_block_tile<ParentSize, GrandParentCGTy> > {
  static_assert(size <= ParentSize, "Sub tile size must be <= parent tile size in tiled_partition");

  __CG_QUALIFIER__ tiled_partition_internal(const thread_block_tile<ParentSize, GrandParentCGTy>& g)
      : thread_block_tile<size, thread_block_tile<ParentSize, GrandParentCGTy> >(g) {}
};

}  // namespace impl

/** \ingroup CooperativeGConstruct
 *  \brief   Create a partition.
 *
 *  \details This constructs a templated class derived from thread_group. The
 *           template defines the tile size of the new thread group at compile
 *           time.
 *
 *  \tparam size       The new size of the partition.
 *  \tparam ParentCGTy The cooperative group class template parameter of the input group.
 *
 *  \param g [in] The coalesced group for split.
 */
template <unsigned int size, class ParentCGTy>
__CG_QUALIFIER__ thread_block_tile<size, ParentCGTy> tiled_partition(const ParentCGTy& g) {
  static_assert(is_valid_tile_size<size>::value,
                "Tiled partition with size > wavefront size. Currently not supported ");
  return impl::tiled_partition_internal<size, ParentCGTy>(g);
}

#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)

/** \ingroup CooperativeGConstruct
 *  \brief Binary partition.
 *
 *  \details This splits the input thread group into two partitions determined by predicate.
 *
 *  \param cgrp [in] The coalesced group for split.
 *  \param pred [in] The predicate used during the group split up.
 */
__CG_QUALIFIER__ coalesced_group binary_partition(const coalesced_group& cgrp, bool pred) {
  auto mask = __ballot_sync<unsigned long long>(cgrp.coalesced_info.member_mask, pred);

  if (pred) {
    return coalesced_group(mask);
  } else {
    return coalesced_group(cgrp.coalesced_info.member_mask ^ mask);
  }
}

/** \ingroup CooperativeGConstruct
 *  \brief Binary partition.
 *
 *  \details This splits the input thread group into two partitions determined by predicate.
 *
 *  \tparam size   The size of the input thread block tile group.
 *  \tparam parent The cooperative group class template parameter of the input group.
 *
 *  \param tgrp [in] The thread block tile group for split.
 *  \param pred [in] The predicate used during the group split up.
 */
template <unsigned int size, class parent>
__CG_QUALIFIER__ coalesced_group binary_partition(const thread_block_tile<size, parent>& tgrp,
                                                  bool pred) {
  auto mask = __ballot_sync<unsigned long long>(tgrp.build_mask(), pred);

  if (pred) {
    return coalesced_group(mask);
  } else {
    return coalesced_group(tgrp.build_mask() ^ mask);
  }
}
#endif
}  // namespace cooperative_groups

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H
