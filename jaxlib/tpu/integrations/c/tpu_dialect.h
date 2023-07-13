#ifndef JAXLIB_TPU_INTEGRATIONS_C_TPU_DIALECT_H_
#define JAXLIB_TPU_INTEGRATIONS_C_TPU_DIALECT_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "jaxlib/tpu/integrations/c/tpu_passes.capi.h.inc"

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TPU, tpu);

MLIR_CAPI_EXPORTED bool mlirTPUAttributeIsATiledLayoutAttr(MlirAttribute attr);

/// Encodes the tiles as an ArrayAttr of DenseI64ArrayAttrs.
MLIR_CAPI_EXPORTED MlirAttribute
mlirTPUTiledLayoutAttrGetTiles(MlirAttribute attr);

MLIR_CAPI_EXPORTED void mlirTPUAnalyzePotentialCommunication(
    MlirOperation op, bool* has_communication, bool* has_custom_barrier);

#ifdef __cplusplus
}
#endif

#endif  // JAXLIB_TPU_INTEGRATIONS_C_TPU_DIALECT_H_
