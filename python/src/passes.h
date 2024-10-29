#define ADD_PASS_WRAPPER_0(name, builder)                                      \
  m.def(name, [](mlir::PassManager &pm) { pm.addPass(builder()); })

#define ADD_PASS_WRAPPER_1(name, builder, ty0)                                 \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder(val0)); })

#define ADD_PASS_WRAPPER_2(name, builder, ty0, ty1)                            \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1) {                  \
    pm.addPass(builder(val0, val1));                                           \
  })

#define ADD_PASS_WRAPPER_3(name, builder, ty0, ty1, ty2)                       \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2) {        \
    pm.addPass(builder(val0, val1, val2));                                     \
  })

#define ADD_PASS_WRAPPER_4(name, builder, ty0, ty1, ty2, ty3)                  \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3) { pm.addPass(builder(val0, val1, val2, val3)); })

#define ADD_PASS_OPTION_WRAPPER_1(name, builder, ty0)                          \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder({val0})); })

#define ADD_PASS_OPTION_WRAPPER_2(name, builder, ty0, ty1)                     \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1) {                  \
    pm.addPass(builder({val0, val1}));                                         \
  })

#define ADD_PASS_OPTION_WRAPPER_3(name, builder, ty0, ty1, ty2)                \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2) {        \
    pm.addPass(builder({val0, val1, val2}));                                   \
  })

#define ADD_PASS_OPTION_WRAPPER_4(name, builder, ty0, ty1, ty2, ty3)           \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3) { pm.addPass(builder({val0, val1, val2, val3})); })
