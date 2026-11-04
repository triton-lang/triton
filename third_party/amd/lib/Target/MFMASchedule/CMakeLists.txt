add_triton_library(TritonAMDMFMASchedule
  MFMAMemSchedule.cpp
  MFMAVALUSchedule.cpp

  LINK_LIBS PUBLIC
  LLVMCore
)
