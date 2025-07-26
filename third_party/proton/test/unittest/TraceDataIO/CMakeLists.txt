set(PROTON_TEST_UTIL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../util/")
add_compile_definitions(PROTON_TEST_UTIL_PATH="${PROTON_TEST_UTIL_PATH}")

add_triton_ut(
	NAME TraceDataIO
	SRCS ByteSpanTest.cpp DecoderTest.cpp CircularLayoutParserTest.cpp ChromeTraceWriterTest.cpp
	LIBS ProtonTraceDataIO
)

target_include_directories(TraceDataIO
PRIVATE
    "${JSON_INCLUDE_DIR}"
	"${PROTON_COMMON_DIR}/include"
    "${PROTON_SRC_DIR}/include"
)
