#ifndef PROTON_PROFILER_ROC_PC_SAMPLING_H_
#define PROTON_PROFILER_ROC_PC_SAMPLING_H_

#include "Utility/Singleton.h"

namespace proton {

class RocprofPCSampling : public Singleton<RocprofPCSampling> {
public:
  RocprofPCSampling();
  ~RocprofPCSampling();
};

} // namespace proton

#endif // PROTON_PROFILER_ROC_PC_SAMPLING_H_
