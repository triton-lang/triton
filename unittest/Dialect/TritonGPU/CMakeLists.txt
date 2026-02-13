add_triton_ut(
  NAME TestSwizzling
  SRCS SwizzleTest.cpp
)
add_triton_ut(
  NAME Dialect
  SRCS DialectTest.cpp
)
add_triton_ut(
  NAME LinearLayoutConversions
  SRCS LinearLayoutConversionsTest.cpp
)

add_triton_ut(
  NAME DumpLayoutTest
  SRCS DumpLayoutTest.cpp
)
