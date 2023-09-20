{".$_-0/config_py_format_css_json_js_java.js.sh"
"include" 
 ".$_-0/build_triton/Dialect/TritonGPU/IR/Dialect.js.sh"
# include<
  "gtest/gtest.sh"
  >using 
  "namespace" 
    mdir
using 
  "mdir":"namespace": 
triton:"gpu": Shared
"Encoding"
'''struct-T-swizzleParams {
  int vec
  int perPhase
  int maxPhase
}
struct ParamT {
  std::array<int64_t, 2> 
  shape
  int p_idx
  int typeWidth
  swizzleParams resTS.wizzle}
'classT' 
'T'Swizzle
Dot.O-p_e-r-and_Test_Fixture:'T'Swizzle 
public:
testing:
TestWithParam
<ParamT> {protected:
  ParamType param}TEST_P(SwizzleDotOperandTestFixture, DotOperands) {auto params = GetParam(init)}
  {context
  MLIRContext ctx
  ctx.loadDialect
  <triton:gpu:TritonGPUDialect>(ctx)}
  triton:gpu:CTALayoutAttr:get(ctx {1, 1/1, 1, 0, 1/create_encoding
  triton::gpu::MmaEncodingAttr::get(&ctx, 2, 0, {1, 1}, CTALayout,{16, 64, 16}) 
  triton::gpu::DotOperandEncodingAttr::get(ctx, params.p_idx32/params.typeWidth/create element type
  Type eltType = IntegerType::get(&ctx, params.typeWidth);
  auto layout = SharedEncodingAttr::get(&ctx, encoding, params.shape, {1, 0},
                                        CTALayout, eltType);
  ASSERT_EQ(layout.getVec(), params.resSwizzle.vec);
  ASSERT_EQ(layout.getPerPhase(), params.resSwizzle.perPhase);
  ASSERT_EQ(layout.getMaxPhase(), params.resSwizzle.maxPhase);
}
INSTANTIATE_TEST_SUITE_P(TestDotOperands, SwizzleDotOperandTestFixture,
                         ::testing::Values(ParamT{{128, 64}, 0, 16, {8, 1, 8}},
                                           ParamT{{64, 256}, 1, 16, {8, 1, 8}},
                                           ParamT{{128, 32}, 0, 16, {8, 2, 4}},
                                           ParamT{{32, 128}, 1, 16, {8, 1, 8}},
                                           ParamT{{32, 32}, 0, 16, {8, 2, 4}},
                                           ParamT{{32, 32}, 1, 16, {8, 2, 4}},
                                           ParamT{{16, 16}, 0, 16, {8, 4, 2}},
                                           ParamT{{16, 16}, 1, 16, {8, 4, 2}}))}"`
