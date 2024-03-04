licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name="tf_head",
    srcs = [
        "tf_head/c_api_macros.h",
        "tf_head/filesystem_interface.h",
        "tf_head/tf_file_statistics.h",
        "tf_head/tf_status.h",
        "tf_head/tsl_status.h",
        "tf_head/logging.h",
    ]
)

cc_library(
    name = "filesystem_plugins_header",
    srcs = [
        "filesystem_plugins.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "@local_config_tf//:stub/libtensorflow_framework.lib",
        ],
        "//conditions:default": [
            "@local_config_tf//:stub/libtensorflow_framework.so",
        ],
    }),
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:tf_c_header_lib",
        "@local_config_tf//:tf_tsl_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "filesystem_plugins",
    srcs = [
        "filesystem_plugins.h",
        "filesystem_plugins.cc",
        "tf_head/c_api_macros.h",
        "tf_head/filesystem_interface.h",
        "tf_head/tf_file_statistics.h",
        "tf_head/tf_status.h",
        "tf_head/tsl_status.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "@local_config_tf//:stub/libtensorflow_framework.lib",
        ],
        "//conditions:default": [
            "@local_config_tf//:stub/libtensorflow_framework.so",
        ],
    }),
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        ":filesystem_plugins_header",
        "//tensorflow_io/core/filesystems/az",
        "//tensorflow_io/core/filesystems/hdfs",
        "//tensorflow_io/core/filesystems/http",
        "//tensorflow_io/core/filesystems/s3",
    ] + select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "//tensorflow_io/core/filesystems/oss",
        ],
    }),
    alwayslink = 1,
)
