#pragma once
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

namespace mlir::triton {

namespace {
struct Descriptor {
  Value base;
  ValueRange shape;
  ValueRange strides;
  Value paddingOption;
  Value roundF32ToTF32;
};
} // namespace

bool hasATensorDescriptorType(mlir::TypeRange types);

mlir::TypeConverter createDescTypeConverter();

FuncArgRenamer createFunArgRenamer();

void populateMakeTensorDescriptorPattern(RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter);

void populateLoadTensorDescriptorPattern(RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter);

void populateStoreTensorDescriptorPattern(RewritePatternSet &patterns,
                                          mlir::TypeConverter &converter);

void populateGatherDescriptorPattern(RewritePatternSet &patterns,
                                     mlir::TypeConverter &converter);

void populateScatterTensorDescriptorPattern(RewritePatternSet &patterns,
                                            mlir::TypeConverter &converter);

void populateReduceTensorDescriptorPattern(RewritePatternSet &patterns,
                                           mlir::TypeConverter &converter);
} // namespace mlir::triton
