/* Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#ifndef DIALECT_NVWS_IR_SEMAPHOREPENDINGCOUNT_H_
#define DIALECT_NVWS_IR_SEMAPHOREPENDINGCOUNT_H_

#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include <optional>

namespace mlir::triton::nvws {

// Result of analyzing the pending count for one semaphore.
//
// The pending count is computed from partitioned semaphore.release ops:
// - each distinct releasing partition contributes independently
// - multiple releases from the same partition do not add extra pending-count
//   slots; they must agree on their async contribution
// - a partitioned semaphore.release must therefore carry exactly one partition
//   id; releasing the same semaphore from multiple partitions requires
//   multiple semaphore.release ops, one per partition
//
// Examples:
// - one waiting partition, two other partitions releasing the same semaphore:
//   pendingCount is 2, so the wait completes only after both releases arrive
// - three waiting partitions, one releasing partition: each waiting partition
//   still observes pendingCount 1 on that semaphore stage
// - two releases from the same partition with identical async_ops are counted
//   once; if they imply different contributions, the analysis reports an error
//
// If no partitioned releases are present, the default pending count is 1.
struct SemaphorePendingCountAnalysis {
  int pendingCount = 1;
  std::optional<unsigned> invalidPartitionArity;
  std::optional<AsyncOp> unsupportedAsyncOp;
  std::optional<int> inconsistentPartitionId;
  int expectedContribution = 0;
  int actualContribution = 0;

  bool hasError() const {
    return invalidPartitionArity.has_value() ||
           unsupportedAsyncOp.has_value() ||
           inconsistentPartitionId.has_value();
  }
};

// Analyze the per-partition arrival contribution for one semaphore.
//
// Each supported async kind in semaphore.release contributes one arrival.  The
// analysis sums the per-partition contribution across distinct partitions,
// requiring repeated releases from the same partition to agree.
SemaphorePendingCountAnalysis
analyzeSemaphorePendingCount(SemaphoreCreateOp op);

} // namespace mlir::triton::nvws

#endif // DIALECT_NVWS_IR_SEMAPHOREPENDINGCOUNT_H_
