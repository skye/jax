/* Copyright 2023 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_HLO_TPU_TPU_DIALECT_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_HLO_TPU_TPU_DIALECT_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "xla/layout.h"
#include "jaxlib/tpu/layout.h"
#include "jaxlib/tpu/tpu_enums.h.inc"

namespace mlir::tpu {
class TPUDialect;
}  // namespace mlir::tpu

#define GET_ATTRDEF_CLASSES
#include "jaxlib/tpu/tpu_attr_defs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "jaxlib/tpu/tpu_type_defs.h.inc"

#define GET_OP_CLASSES
#include "jaxlib/tpu/tpu_dialect.h.inc"
#include "jaxlib/tpu/tpu_ops.h.inc"

namespace mlir {
namespace tpu {

std::pair<bool, bool> mightCommunicateBetweenChips(Operation* op);

std::unique_ptr<OperationPass<func::FuncOp>>
createLogicalToPhysicalDeviceIdPass(int64_t total_devices);

// In Mosaic, we often strip tiled layouts from memrefs, for compatibility with
// vector ops. This functions inverts the layout erasure applied to the value.
MemRefType getMemRefType(Value value);

}  // namespace tpu
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_HLO_TPU_TPU_DIALECT_H_
