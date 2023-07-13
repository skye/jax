#ifndef JAX_DIALECTS_H
#define JAX_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Arith, arith);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Math, math);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Vector, vector);

MLIR_CAPI_EXPORTED void mlirJAXRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // JAX_DIALECTS_H
