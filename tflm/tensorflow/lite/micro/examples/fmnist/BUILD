# Description:
#   TensorFlow Lite for Microcontrollers "fmnist" example.
load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "model",
    srcs = [
        "//tensorflow/lite/micro/examples/fmnist/models:generated_fmnist_float_model_cc",
        "//tensorflow/lite/micro/examples/fmnist/models:generated_fmnist_int8_model_cc",
        "//tensorflow/lite/micro/examples/fmnist/models:generated_fmnist_pruned_int8_model_cc",
    ],
    hdrs = [
        "//tensorflow/lite/micro/examples/fmnist/models:generated_fmnist_float_model_hdr",
        "//tensorflow/lite/micro/examples/fmnist/models:generated_fmnist_int8_model_hdr",
        "//tensorflow/lite/micro/examples/fmnist/models:generated_fmnist_pruned_int8_model_hdr",
    ],
)

tflm_cc_test(
    name = "fmnist_test",
    srcs = [
        "fmnist_test.cc",
    ],
    deps = [
        ":model",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
