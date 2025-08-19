# CMake generated Testfile for 
# Source directory: /Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU
# Build directory: /Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU/TestSwizzling[1]_include.cmake")
include("/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU/Dialect[1]_include.cmake")
include("/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU/LinearLayoutConversions[1]_include.cmake")
include("/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU/DumpLayoutTest[1]_include.cmake")
add_test([=[TestSwizzling]=] "/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU/TestSwizzling")
set_tests_properties([=[TestSwizzling]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/andrew/zzCoding-play/triton/cmake/AddTritonUnitTest.cmake;12;add_test;/Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU/CMakeLists.txt;1;add_triton_ut;/Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU/CMakeLists.txt;0;")
add_test([=[Dialect]=] "/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU/Dialect")
set_tests_properties([=[Dialect]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/andrew/zzCoding-play/triton/cmake/AddTritonUnitTest.cmake;12;add_test;/Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU/CMakeLists.txt;9;add_triton_ut;/Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU/CMakeLists.txt;0;")
add_test([=[LinearLayoutConversions]=] "/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU/LinearLayoutConversions")
set_tests_properties([=[LinearLayoutConversions]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/andrew/zzCoding-play/triton/cmake/AddTritonUnitTest.cmake;12;add_test;/Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU/CMakeLists.txt;18;add_triton_ut;/Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU/CMakeLists.txt;0;")
add_test([=[DumpLayoutTest]=] "/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Dialect/TritonGPU/DumpLayoutTest")
set_tests_properties([=[DumpLayoutTest]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/andrew/zzCoding-play/triton/cmake/AddTritonUnitTest.cmake;12;add_test;/Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU/CMakeLists.txt;27;add_triton_ut;/Users/andrew/zzCoding-play/triton/unittest/Dialect/TritonGPU/CMakeLists.txt;0;")
