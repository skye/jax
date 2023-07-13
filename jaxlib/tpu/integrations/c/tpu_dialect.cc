#include "jaxlib/tpu/integrations/c/tpu_dialect.h"

#include "jaxlib/tpu/tpu_dialect.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TPU, tpu, mlir::tpu::TPUDialect);

bool mlirTPUAttributeIsATiledLayoutAttr(MlirAttribute attr) {
  return llvm::isa<mlir::tpu::TiledLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirTPUTiledLayoutAttrGetTiles(MlirAttribute attr) {
  auto layout_attr = llvm::cast<mlir::tpu::TiledLayoutAttr>(unwrap(attr));
  std::vector<mlir::Attribute> tile_attrs;
  tile_attrs.reserve(layout_attr.getTiles().size());
  mlir::MLIRContext *ctx = layout_attr.getContext();
  for (auto &tile : layout_attr.getTiles()) {
    auto d = tile.dimensions();
    tile_attrs.push_back(mlir::DenseI64ArrayAttr::get(
        ctx, llvm::ArrayRef<int64_t>(d.begin(), d.end())));
  }
  return wrap(mlir::ArrayAttr::get(ctx, tile_attrs));
}

void mlirTPUAnalyzePotentialCommunication(MlirOperation op,
                                          bool *has_communication,
                                          bool *has_custom_barrier) {
  auto result = mlir::tpu::mightCommunicateBetweenChips(unwrap(op));
  *has_communication = result.first;
  *has_custom_barrier = result.second;
}

using namespace mlir::tpu;

#include "jaxlib/tpu/integrations/c/tpu_passes.capi.cc.inc"
}
