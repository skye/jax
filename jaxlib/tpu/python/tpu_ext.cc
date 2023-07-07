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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "mlir-c/AffineMap.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/lib/Bindings/Python/IRModule.h"
#include "jaxlib/tpu/tpu_dialect.h"

PYBIND11_MODULE(_tpu_ext, m) {
  m.def("register_tpu_dialect",
        [](MlirContext ctx) {
          unwrap(ctx)->loadDialect<mlir::tpu::TPUDialect>();
        },
        py::arg("context"));
  // TODO(apaszke): All of those should be upstreamed to MLIR Python bindings.
  m.def("private_replace_all_uses_with",
        [](MlirOperation op, std::vector<MlirValue> vals) {
          std::vector<mlir::Value> values(vals.size());
          std::transform(vals.begin(), vals.end(), values.begin(),
                         [](MlirValue v) { return unwrap(v); });
          unwrap(op)->replaceAllUsesWith(llvm::ArrayRef<mlir::Value>(values));
        });
  m.def("private_replace_all_uses_except",
        [](MlirValue old, MlirValue new_val, MlirOperation except) {
          unwrap(old).replaceAllUsesExcept(unwrap(new_val), unwrap(except));
        });
  m.def("private_set_operand",
        [](MlirOperation op, int idx, MlirValue new_operand) {
          unwrap(op)->setOperand(idx, unwrap(new_operand));
        });
  m.def("private_set_operands",
        [](MlirOperation op, std::vector<MlirValue> new_operands) {
          std::vector<mlir::Value> values(new_operands.size());
          std::transform(new_operands.begin(), new_operands.end(),
                         values.begin(), [](MlirValue v) { return unwrap(v); });
          unwrap(op)->setOperands(values);
        });
  m.def("private_has_no_memory_space", [](MlirType ty) {
    return !llvm::cast<mlir::MemRefType>(unwrap(ty)).getMemorySpace();
  });
  m.def("private_is_identity", [](MlirAttribute attr) {
    return llvm::cast<mlir::AffineMapAttr>(unwrap(attr)).isIdentity();
  });
  m.def("private_insert_argument",
        [](int index, mlir::python::PyBlock py_block,
           MlirType py_type) -> MlirValue {
          auto type = unwrap(py_type);
          return wrap(
              unwrap(py_block.get())
                  ->insertArgument(index, type,
                                   mlir::UnknownLoc::get(type.getContext())));
        });
  m.def("private_is_tiled_layout", [](MlirAttribute attr) {
    return llvm::dyn_cast<mlir::tpu::TiledLayoutAttr>(unwrap(attr)) != nullptr;
  });
  m.def("private_get_tiles", [](MlirAttribute py_attr) -> py::object {
    auto tiles =
        llvm::cast<mlir::tpu::TiledLayoutAttr>(unwrap(py_attr)).getTiles();
    py::tuple t(tiles.size());
    for (int64_t i = 0; i < tiles.size(); ++i) {
      auto dimensions = tiles[i].dimensions();
      py::tuple py_dimensions(dimensions.size());
      for (int64_t j = 0; j < dimensions.size(); ++j) {
        py_dimensions[j] = py::cast(dimensions[j]);
      }
      t[i] = py_dimensions;
    }
    return t;
  });
  m.def("private_set_arg_attr",
        [](MlirOperation op, unsigned i, std::string name, MlirAttribute attr) {
          llvm::dyn_cast<mlir::func::FuncOp>(unwrap(op))
              .setArgAttr(i, name, unwrap(attr));
        });
  m.def("private_has_communication", [](MlirOperation op) {
    return mlir::tpu::mightCommunicateBetweenChips(unwrap(op));
  });
  m.def("private_move_all_regions", [](MlirOperation src, MlirOperation dst) {
    auto src_ptr = unwrap(src);
    auto dst_ptr = unwrap(dst);
    CHECK(src_ptr->getNumRegions() == dst_ptr->getNumRegions())
        << "Region numbers do not match in src operation and dst operations";
    for (int i = 0; i < src_ptr->getNumRegions(); ++i) {
      dst_ptr->getRegion(i).takeBody(src_ptr->getRegion(i));
    }
  });
}
