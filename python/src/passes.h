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

#define ADD_PASS_WRAPPER_5(name, builder, ty0, ty1, ty2, ty3, ty4)             \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2, ty3 val3,      \
           ty4 val4) { pm.addPass(builder(val0, val1, val2, val3, val4)); })

#define ADD_PASS_WRAPPER_6(name, builder, ty0, ty1, ty2, ty3, ty4, ty5)        \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3, ty4 val4, ty5 val5) {                               \
    pm.addPass(builder(val0, val1, val2, val3, val4, val5));                   \
  })

#define ADD_PASS_WRAPPER_7(name, builder, ty0, ty1, ty2, ty3, ty4, ty5, ty6)   \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3, ty4 val4, ty5 val5, ty6 val6) {                     \
    pm.addPass(builder(val0, val1, val2, val3, val4, val5, val6));             \
  })

#define ADD_PASS_WRAPPER_8(name, builder, ty0, ty1, ty2, ty3, ty4, ty5, ty6,   \
                           ty7)                                                \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3, ty4 val4, ty5 val5, ty6 val6, ty7 val7) {           \
    pm.addPass(builder(val0, val1, val2, val3, val4, val5, val6, val7));       \
  })

  #define ADD_PASS_WRAPPER_9(name, builder, ty0, ty1, ty2, ty3, ty4, ty5, ty6, \
                           ty7, ty8)                                           \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3, ty4 val4, ty5 val5, ty6 val6, ty7 val7, ty8 val8) { \
    pm.addPass(builder(val0, val1, val2, val3, val4, val5, val6, val7, val8)); \
  })

#define ADD_PASS_WRAPPER_9(name, builder, ty0, ty1, ty2, ty3, ty4, ty5, ty6,   \
                           ty7, ty8)                                           \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3, ty4 val4, ty5 val5, ty6 val6, ty7 val7, ty8 val8) { \
    pm.addPass(builder(val0, val1, val2, val3, val4, val5, val6, val7, val8)); \
  })

#define ADD_PASS_WRAPPER_10(name, builder, ty0, ty1, ty2, ty3, ty4, ty5, ty6,  \
                           ty7, ty8, ty9)                                      \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3, ty4 val4, ty5 val5, ty6 val6, ty7 val7, ty8 val8,   \
                 ty9 val9) { \
    pm.addPass(builder(val0, val1, val2, val3, val4, val5, val6, val7, val8,   \
                       val9)); \
  })

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
