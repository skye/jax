// Registers MLIR dialects used by JAX.
// This module is called by mlir/__init__.py during initialization.

#include "mlir-c/Dialect/Func.h"
#include "jaxlib/mlir/_mlir_libs/jax_dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_site_initialize_0, m) {
  m.doc() = "Registers MLIR dialects used by JAX.";

#define REGISTER_DIALECT(name) \
    MlirDialectHandle name##_dialect = mlirGetDialectHandle__##name##__(); \
    mlirDialectHandleInsertDialect(name##_dialect, registry);

  mlirJAXRegisterAllPasses();

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    REGISTER_DIALECT(arith)
    REGISTER_DIALECT(func)
    REGISTER_DIALECT(math)
    REGISTER_DIALECT(vector)
  });
}
