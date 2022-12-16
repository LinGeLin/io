licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//third_party/tf:tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "genome_ops",
    srcs = [
        "kernels/fastq.cc",
        "ops/read_fastq.cc",
    ],
    copts = tf_io_copts(),
    includes = [
        ".",
    ],
    linkstatic = True,
    deps = [
        "//tensorflow_io/core:dataset_ops",
        "@nucleus//:fastq_reader",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)
