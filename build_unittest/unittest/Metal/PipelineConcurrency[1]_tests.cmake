add_test([=[MetalPipeline.ConcurrentCreationProducesSinglePipeline]=]  /Users/andrew/zzCoding-play/triton/build_unittest/unittest/Metal/PipelineConcurrency [==[--gtest_filter=MetalPipeline.ConcurrentCreationProducesSinglePipeline]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[MetalPipeline.ConcurrentCreationProducesSinglePipeline]=]  PROPERTIES WORKING_DIRECTORY /Users/andrew/zzCoding-play/triton/build_unittest/unittest/Metal SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  PipelineConcurrency_TESTS MetalPipeline.ConcurrentCreationProducesSinglePipeline)
