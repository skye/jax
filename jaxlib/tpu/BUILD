# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO(sharadmv): describe this package.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

# Mosaic is still an experimental project. If you want to use it, please get in touch
# with us. We also depend on MLIR and LLVM, both of which only provide loose SLA guarantees.

cc_library(
    name = "type_util",
    srcs = ["type_util.cc"],
    hdrs = ["type_util.h"],
    compatible_with = ["//buildenv/target:libtpu"],
    deps = [
        "@llvm-project//mlir:IR",
        "@xla//xla:xla_data_proto_cc",
    ],
)

################################################################################
# TPU dialect

cc_library(
    name = "tpu_dialect",
    srcs = [
        "communication.cc",
        "layout.cc",
        "tpu_dialect.cc",
        "tpu_ops.cc",
    ],
    hdrs = [
        "layout.h",
        "tpu_dialect.h",
    ],
    deps = [
        ":tpu_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
        "@xla//xla:shape_util",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "tpu_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tpu_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "tpu_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "tpu_dialect.cc.inc",
        ),
        (
            ["-gen-enum-decls"],
            "tpu_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "tpu_enums.cc.inc",
        ),
        (
            ["-gen-attrdef-decls"],
            "tpu_attr_defs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "tpu_attr_defs.cc.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "tpu_type_defs.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "tpu_type_defs.cc.inc",
        ),
        (
            [
                "-gen-pass-decls",
                "-name=TPU",
            ],
            "tpu_passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=TPU",
            ],
            "integrations/c/tpu_passes.capi.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tpu.td",
    deps = [":tpu_td_files"],
)

td_library(
    name = "tpu_td_files",
    srcs = [
        "tpu.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

# C API targets

TPU_CAPI_SOURCES = [
    "integrations/c/tpu_dialect.cc",
]

TPU_CAPI_HEADERS = [
    "integrations/c/tpu_dialect.h",
]

cc_library(
    name = "tpu_dialect_capi",
    srcs = TPU_CAPI_SOURCES,
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_dialect",
        ":tpu_inc_gen",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "tpu_dialect_capi_headers",
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_inc_gen",
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "tpu_dialect_capi_objects",
    srcs = TPU_CAPI_SOURCES,
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_dialect",
        ":tpu_inc_gen",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = True,
)

py_library(
    name = "tpu_support",
    deps = [
        "//jaxlib/tpu/python:tpu_dialect",
    ],
)
