#include "test/include/Analysis/TestAxisInfo.h"

namespace mlir {
namespace test {
void registerTestAlignmentPass() { PassRegistration<TestAxisInfoPass>(); }
} // namespace test
} // namespace mlir
