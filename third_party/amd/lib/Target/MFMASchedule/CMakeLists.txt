add_triton_library(TritonAMDMFMASchedule
  MFMASchedule.cpp
  MFMAVALUSchedule.cpp

  LINK_LIBS PUBLIC
  LLVMCore
)
