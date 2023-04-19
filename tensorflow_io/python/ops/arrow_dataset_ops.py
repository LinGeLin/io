# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Arrow Dataset."""

from functools import partial
import io
from itertools import chain
import os
import socket
import threading
import tempfile

import tensorflow as tf
from tensorflow import dtypes
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure as structure_lib
from tensorflow_io.python.ops import core_ops

if hasattr(tf, "nest"):
    from tensorflow import nest  # pylint: disable=ungrouped-imports
else:
    from tensorflow.python.data.util import nest  # pylint: disable=ungrouped-imports

class ArrowBaseDataset(dataset_ops.DatasetV2):
    """Base class for Arrow Datasets to provide columns used in record batches
    and corresponding output tensor types, shapes and classes.
    """

    batch_modes_supported = ("keep_remainder", "drop_remainder", "auto")

    def __init__(
        self,
        make_variant_fn,
        columns,
        output_types,
        output_shapes=None,
        batch_size=None,
        batch_mode="keep_remainder",
    ):
        self._columns = columns
        self._structure = structure_lib.convert_legacy_structure(
            output_types,
            output_shapes
            or nest.map_structure(lambda _: tf.TensorShape(None), output_types),
            nest.map_structure(lambda _: tf.Tensor, output_types),
        )
        self._batch_size = tf.convert_to_tensor(
            batch_size or 0, dtype=dtypes.int64, name="batch_size"
        )
        if batch_mode not in self.batch_modes_supported:
            raise ValueError(
                "Unsupported batch_mode: '{}', must be one of {}".format(
                    batch_mode, self.batch_modes_supported
                )
            )
        self._batch_mode = tf.convert_to_tensor(
            batch_mode, dtypes.string, name="batch_mode"
        )
        if batch_size is not None or batch_mode == "auto":
            spec_batch_size = batch_size if batch_mode == "drop_remainder" else None
            # pylint: disable=protected-access
            self._structure = nest.map_structure(
                lambda component_spec: component_spec._batch(spec_batch_size),
                self._structure,
            )
        variant_tensor = make_variant_fn(
            columns=self._columns,
            batch_size=self._batch_size,
            batch_mode=self._batch_mode,
            **self._flat_structure,
        )
        super().__init__(variant_tensor)

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._structure

    @property
    def columns(self):
        return self._columns

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch_mode(self):
        return self._batch_mode

class ArrowS3Dataset(ArrowBaseDataset):
    """An Arrow Dataset for reading record batches from an input stream.
    Currently supported input streams are a socket client or stdin.
    """

    def __init__(
        self,
        aws_access_key,
        aws_secret_key,
        aws_endpoint_override,
        parquet_files,
        column_names,
        columns,
        output_types,
        output_shapes=None,
        batch_size=None,
        batch_mode="keep_remainder",
        filter="",
        same_schema=True,
    ):
        """Create an ArrowDataset from an input stream.

        Args:
            aws_access_key: S3 access key
            aws_secret_key: S3 secret key
            aws_endpoint_override: S3 endpoint override
            parquet_files: A list of parquet files path on s3
            column_names: A list of column names to be used in the dataset
            columns: A list of column indices to be used in the Dataset
            output_types: Tensor dtypes of the output tensors
            output_shapes: TensorShapes of the output tensors or None to
                            infer partial
            batch_size: Batch size of output tensors, setting a batch size here
                        will create batched tensors from Arrow memory and can be more
                        efficient than using tf.data.Dataset.batch().
                        NOTE: batch_size does not need to be set if batch_mode='auto'
            batch_mode: Mode of batching, supported strings:
                        "keep_remainder" (default, keeps partial batch data),
                        "drop_remainder" (discard partial batch data),
                        "auto" (size to number of records in Arrow record batch)
            filter : filter for reade row
            same_schema : Whether the input files have the same view（default true）
        """
        aws_access_key = tf.convert_to_tensor(
            aws_access_key, dtype=dtypes.string, name="aws_access_key"
        )
        aws_secret_key = tf.convert_to_tensor(
            aws_secret_key, dtype=dtypes.string, name="aws_secret_key"
        )
        aws_endpoint_override = tf.convert_to_tensor(
            aws_endpoint_override, dtype=dtypes.string, name="aws_endpoint_override"
        )
        parquet_files = tf.convert_to_tensor(
            parquet_files, dtype=dtypes.string, name="parquet_files"
        )
        column_names = tf.convert_to_tensor(
            column_names, dtype=dtypes.string, name="column_names"
        )
        filter = tf.convert_to_tensor(filter, dtype=dtypes.string, name="filter")
        same_schema = tf.convert_to_tensor(
            same_schema, dtype=dtypes.bool, name="same_schema"
        )
        super().__init__(
            partial(
                core_ops.io_arrow_s3_dataset,
                aws_access_key,
                aws_secret_key,
                aws_endpoint_override,
                parquet_files,
                column_names,
                filter,
                same_schema,
            ),
            columns,
            output_types,
            output_shapes,
            batch_size,
            batch_mode,
        )


def list_feather_columns(filename, **kwargs):

    """list_feather_columns"""
    if not tf.executing_eagerly():
        raise NotImplementedError("list_feather_columns only support eager mode")
    memory = kwargs.get("memory", "")
    columns, dtypes_, shapes = core_ops.io_list_feather_columns(filename, memory=memory)
    entries = zip(tf.unstack(columns), tf.unstack(dtypes_), tf.unstack(shapes))
    return {
        column.numpy().decode(): tf.TensorSpec(
            shape.numpy(), dtype.numpy().decode(), column.numpy().decode()
        )
        for (column, dtype, shape) in entries
    }
