/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("IO>ListFeatherColumns")
    .Input("filename: string")
    .Input("memory: string")
    .Output("columns: string")
    .Output("dtypes: string")
    .Output("shapes: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      c->set_output(2, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("IO>ArrowS3Dataset")
    .Input("aws_access_key: string")
    .Input("aws_secret_key: string")
    .Input("aws_endpoint_override: string")
    .Input("parquet_files: string")
    .Input("column_names: string")
    .Input("filter: string")
    .Input("same_schema: bool")
    .Input("columns: int32")
    .Input("batch_size: int64")
    .Input("batch_mode: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset from s3 parqeut files

aws_access_key: S3 access key.
aws_secret_key: S3 secret_key.
aws_endpoint_override: S3 endpoint override
parquet_files: One or more parqeut file path on s3
column_names: Select columns to read by names
)doc");

}  // namespace tensorflow
