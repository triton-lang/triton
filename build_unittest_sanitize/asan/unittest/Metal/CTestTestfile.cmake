# CMake generated Testfile for 
# Source directory: /Users/andrew/zzCoding-play/triton/unittest/Metal
# Build directory: /Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Metal
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Metal/ArgumentBinding[1]_include.cmake")
include("/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Metal/RuntimeTest[1]_include.cmake")
include("/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Metal/PipelineConcurrency[1]_include.cmake")
add_test([=[ArgumentBinding]=] "/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Metal/ArgumentBinding")
set_tests_properties([=[ArgumentBinding]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/andrew/zzCoding-play/triton/cmake/AddTritonUnitTest.cmake;12;add_test;/Users/andrew/zzCoding-play/triton/unittest/Metal/CMakeLists.txt;1;add_triton_ut;/Users/andrew/zzCoding-play/triton/unittest/Metal/CMakeLists.txt;0;")
add_test([=[RuntimeTest]=] "/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Metal/RuntimeTest")
set_tests_properties([=[RuntimeTest]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/andrew/zzCoding-play/triton/cmake/AddTritonUnitTest.cmake;12;add_test;/Users/andrew/zzCoding-play/triton/unittest/Metal/CMakeLists.txt;7;add_triton_ut;/Users/andrew/zzCoding-play/triton/unittest/Metal/CMakeLists.txt;0;")
add_test([=[PipelineConcurrency]=] "/Users/andrew/zzCoding-play/triton/build_unittest_sanitize/asan/unittest/Metal/PipelineConcurrency")
set_tests_properties([=[PipelineConcurrency]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/andrew/zzCoding-play/triton/cmake/AddTritonUnitTest.cmake;12;add_test;/Users/andrew/zzCoding-play/triton/unittest/Metal/CMakeLists.txt;13;add_triton_ut;/Users/andrew/zzCoding-play/triton/unittest/Metal/CMakeLists.txt;0;")
