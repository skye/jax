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
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup")

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
    compatible_with = ["//buildenv/target:libtpu"],
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
    compatible_with = ["//buildenv/target:libtpu"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

py_library(
    name = "tpu_support",
    deps = [
        # "//third_party/py/jax/jaxlib/tpu/python:tpu_dialect",
    ],
)
