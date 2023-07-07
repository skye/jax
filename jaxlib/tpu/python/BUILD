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
load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup")
load("//jaxlib:jax.bzl", "pybind_extension")

gentbl_filegroup(
    name = "tpu_python_gen_raw",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=tpu",
            ],
            "_tpu_gen_raw.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = ":tpu_python.td",
    td_srcs = [
        "@llvm-project//mlir:include/mlir/Bindings/Python/Attributes.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@xla//xla/mlir_hlo/tpu:tpu_td_files",
    ],
)

genrule(
    name = "tpu_python_gen",
    srcs = ["_tpu_gen_raw.py"],
    outs = ["_tpu_gen.py"],
    cmd = "cat $(location _tpu_gen_raw.py) | sed -e 's/^from \\./from mlir\\.dialects\\./g' > $@",
)

py_library(
    name = "tpu_dialect",
    srcs = [
        "_tpu_gen.py",
        "_tpu_ops_ext.py",
        "tpu.py",
    ],
    deps = [
        ":_tpu_ext",
        "//third_party/py/mlir",
    ],
)

pybind_extension(
    name = "_tpu_ext",
    srcs = ["tpu_ext.cc"],
    deps = [
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeadersAndDeps",
        "@xla//xla/mlir_hlo/tpu:tpu_dialect",
    ],
)

py_library(
    name = "tpu_passes",
    srcs = [
        "apply_vector_layout.py",
        "infer_memref_layout.py",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:flagsaver",
        "//third_party/py/jax/jaxlib/tpu/python:tpu_dialect",
        "//third_party/py/mlir:arithmetic_dialect",
        "//third_party/py/mlir:func_dialect",
        "//third_party/py/mlir:ir",
        "//third_party/py/mlir:math_dialect",
        "//third_party/py/mlir:pass_manager",
        "//third_party/py/mlir:scf_dialect",
        "//third_party/py/mlir:stablehlo_dialect",
        "//third_party/py/mlir:vector_dialect",
        "//third_party/py/numpy",
    ],
)
