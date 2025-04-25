add_subdirectory(TraceDataIO)

add_proton_library(ProtonData
	Data.cpp
	TraceData.cpp
	TreeData.cpp
)
