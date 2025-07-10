#ifndef SYNCLOG_HPP
#define SYNCLOG_HPP

#include <stdio.h>

const char *opNames[] = {"barrier.sync",
                         "cp.async.bulk.commit_group",
                         "cp.async.commit_group",
                         "cp.async.bulk.wait_group",
                         "cp.async.wait_group",
                         "cp.async.bulk.tensor",
                         "cp.reduce.async.bulk.tensor",
                         "fence.proxy",
                         "tensormap.cp_fenceproxy",
                         "fence.mbarrier_init",
                         "wgmma.fence",
                         "wgmma.commit_group",
                         "wgmma.wait_group",
                         "mbarrier.init",
                         "mbarrier.wait",
                         "mbarrier.arrive",
                         "mbarrier.inval",
                         "mbarrier.expect_tx",
                         "mbarrier.test_wait",
                         "mbarrier.try_wait",
                         "tcgen05.wait",
                         "tcgen05.commit",
                         "cp.async.mbarrier.arrive"};

void printSynclog(uint32_t *synclog_buffer) {
  printf("synclog start\n");
  for (size_t i = 1; i < synclog_buffer[0];) {
    {
      uint32_t headerNumber = synclog_buffer[i];
      const char *opName = opNames[headerNumber];
      printf("%s ", opName);
      uint32_t numArgs = synclog_buffer[i + 1];
      i += 2;
      for (size_t j = 0; j < numArgs; j++) {
        uint32_t arg = synclog_buffer[i + j];
        printf("arg%lu=%u ", j, arg);
      }
      i += numArgs;
      uint32_t time_lo = synclog_buffer[i];
      uint32_t time_hi = synclog_buffer[i + 1];
      uint32_t threadIdx_x = synclog_buffer[i + 2];
      uint32_t threadIdx_y = synclog_buffer[i + 3];
      uint32_t threadIdx_z = synclog_buffer[i + 4];
      uint32_t blockIdx_x = synclog_buffer[i + 5];
      uint32_t blockIdx_y = synclog_buffer[i + 6];
      uint32_t blockIdx_z = synclog_buffer[i + 7];
      uint32_t ctaRank = synclog_buffer[i + 8];
      printf("time=%lu thread=%u,%u,%u block=%u,%u,%u ctaRank=%u\n",
             (uint64_t)time_hi << 32 | time_lo, threadIdx_x, threadIdx_y,
             threadIdx_z, blockIdx_x, blockIdx_y, blockIdx_z, ctaRank);
      i += 9;
    }
  }
  printf("synclog end\n");
}

#endif
