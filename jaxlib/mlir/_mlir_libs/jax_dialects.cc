#include "jaxlib/mlir/_mlir_libs/jax_dialects.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/Passes.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arith, arith, mlir::arith::ArithDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Vector, vector, mlir::vector::VectorDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Math, math, mlir::math::MathDialect)

MLIR_CAPI_EXPORTED void mlirJAXRegisterAllPasses() { mlir::registerTransformsPasses(); }

}
