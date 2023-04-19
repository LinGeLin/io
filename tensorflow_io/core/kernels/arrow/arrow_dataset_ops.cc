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

#include <numeric>

#include "arrow/api.h"
#include "arrow/io/stdio.h"
#include "arrow/ipc/api.h"
#include "arrow/result.h"
#include "parquet/arrow/reader.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow_io/core/kernels/arrow/arrow_kernels.h"
#include "tensorflow_io/core/kernels/arrow/arrow_stream_client.h"
#include "tensorflow_io/core/kernels/arrow/arrow_util.h"
#include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {
namespace data {

enum ArrowBatchMode {
  BATCH_KEEP_REMAINDER,
  BATCH_DROP_REMAINDER,
  BATCH_AUTO,
};

Status GetBatchModeStr(ArrowBatchMode batch_mode, tstring* batch_mode_str) {
  switch (batch_mode) {
    case ArrowBatchMode::BATCH_KEEP_REMAINDER:
      *batch_mode_str = "keep_remainder";
      break;
    case ArrowBatchMode::BATCH_DROP_REMAINDER:
      *batch_mode_str = "drop_remainder";
      break;
    case ArrowBatchMode::BATCH_AUTO:
      *batch_mode_str = "auto";
      break;
    default:
      return errors::Internal("Unsupported batch mode: " +
                              std::to_string(batch_mode));
  }
  return Status::OK();
}

Status GetBatchMode(string batch_mode_str, ArrowBatchMode* batch_mode) {
  if (batch_mode_str == "keep_remainder") {
    *batch_mode = ArrowBatchMode::BATCH_KEEP_REMAINDER;
  } else if (batch_mode_str == "drop_remainder") {
    *batch_mode = ArrowBatchMode::BATCH_DROP_REMAINDER;
  } else if (batch_mode_str == "auto") {
    *batch_mode = ArrowBatchMode::BATCH_AUTO;
  } else {
    return errors::Internal("Unsupported batch mode: " + batch_mode_str);
  }
  return Status::OK();
}

// Base class for defining a Dataset over Arrow record batches with an
// iterator that iterates over rows of the batch to get Tensors
class ArrowDatasetBase : public DatasetBase {
 public:
  ArrowDatasetBase(OpKernelContext* ctx, const std::vector<int32>& columns,
                   const int64 batch_size, const ArrowBatchMode batch_mode,
                   const DataTypeVector& output_types,
                   const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        columns_(columns),
        batch_size_(batch_size),
        batch_mode_(batch_mode),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

 protected:
  // Abstract base class for iterating over rows of Arrow record
  // batches. Implementations will define how record batches are
  // initialized and consumed.
  template <typename DatasetType>
  class ArrowBaseIterator : public DatasetIterator<DatasetType> {
   public:
    ArrowBaseIterator(
        const typename DatasetIterator<DatasetType>::Params& params)
        : DatasetIterator<DatasetType>(params) {}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      // If in initial state, setup and read first batch
      if (current_batch_ == nullptr && current_row_idx_ == 0) {
        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      }

      std::vector<Tensor>* result_tensors = out_tensors;
      auto partial_batches =
          std::vector<std::shared_ptr<std::vector<Tensor>>>();
      int64 partial_batch_size = 0;
      bool have_result = false;

      // Loop until have_result or end_of_sequence
      do {
        // Try to go to next batch if consumed all rows in current batch
        if (current_batch_ != nullptr &&
            current_row_idx_ >= current_batch_->num_rows()) {
          TF_RETURN_IF_ERROR(NextStreamLocked(ctx->env()));
        }

        // Check if reached end of stream
        if (current_batch_ == nullptr) {
          // Finalize the iterator state
          ResetStreamsLocked();

          // Return partial batch if drop_remainder flag not set
          if (partial_batch_size > 0 &&
              this->dataset()->batch_mode_ !=
                  ArrowBatchMode::BATCH_DROP_REMAINDER) {
            // Copy partial batched tensors to output tensors
            TF_RETURN_IF_ERROR(AppendPartialTensors(
                ctx, partial_batch_size, partial_batches, out_tensors));
            have_result = true;
            // No more results, so end the sequence
          } else {
            *end_of_sequence = true;
          }
        } else {
          // Calc the batch size, will be 0 if not batching
          int64 batch_size =
              this->dataset()->batch_mode_ == ArrowBatchMode::BATCH_AUTO
                  ?
                  // Auto batch size is number of rows in current record batch
                  current_batch_->num_rows()
                  :
                  // Use set batch size minus any partials already read
                  this->dataset()->batch_size_ - partial_batch_size;

          // Prepare a partial batch to save, either current record batch is too
          // small or continuing to fill previous partial batch
          if (batch_size != 0 &&
              (partial_batch_size > 0 ||
               current_row_idx_ + batch_size > current_batch_->num_rows())) {
            int64 rows_remaining =
                current_batch_->num_rows() - current_row_idx_;
            batch_size = std::min(batch_size, rows_remaining);
            partial_batches.push_back(std::make_shared<std::vector<Tensor>>());
            result_tensors = partial_batches.back().get();
            partial_batch_size += batch_size;
          }

          // Assign Tensors for each column in the current row
          result_tensors->reserve(this->dataset()->columns_.size());
          for (size_t i = 0; i < this->dataset()->columns_.size(); ++i) {
            int32 col = this->dataset()->columns_[i];
            DataType output_type = this->dataset()->output_types_[i];
            std::shared_ptr<arrow::Array> arr = current_batch_->column(col);

            // Get the TensorShape for the column batch
            TensorShape output_shape = TensorShape({});
            TF_RETURN_IF_ERROR(ArrowUtil::AssignShape(
                arr, current_row_idx_, batch_size, &output_shape));

            if (output_shape.dims() == 1) {
              auto&& output_shape_in = this->dataset()->output_shapes_[i];
              if (output_shape_in.dim_size(output_shape_in.dims() - 1) == 1) {
                output_shape.AddDim(1);
              }
            }

            // Allocate a new tensor and assign Arrow data to it
            Tensor tensor(ctx->allocator({}), output_type, output_shape);
            TF_RETURN_IF_ERROR(
                ArrowUtil::AssignTensor(arr, current_row_idx_, &tensor));
            result_tensors->emplace_back(std::move(tensor));
          }

          // If not batching or have a full batch, then have a result to return
          if (partial_batch_size == 0 ||
              partial_batch_size == this->dataset()->batch_size_) {
            have_result = true;

            // If have partial batches, copy partial tensors to output tensors
            if (!partial_batches.empty()) {
              TF_RETURN_IF_ERROR(AppendPartialTensors(
                  ctx, partial_batch_size, partial_batches, out_tensors));
            }
          }

          // Increment to next row or batch
          current_row_idx_ += batch_size == 0 ? 1 : batch_size;
          *end_of_sequence = false;
        }

      } while (!(have_result || *end_of_sequence));

      return Status::OK();
    }

   private:
    Status AppendPartialTensors(
        IteratorContext* ctx, int64 batch_size,
        const std::vector<std::shared_ptr<std::vector<Tensor>>>& partials,
        std::vector<Tensor>* out_tensors) {
      int64 batch_index = 0;

      // If only one partial batch, can just move to output
      if (partials.size() == 1) {
        *out_tensors = std::move(*partials.at(0).get());
        return Status::OK();
      }

      // Copy all partial tensors to a single output tensor
      for (auto it_partial = partials.begin(); it_partial != partials.end();
           it_partial++) {
        int64 partial_batch_size = 0;
        for (size_t i = 0; i < (*it_partial)->size(); ++i) {
          const Tensor& element = (*it_partial)->at(i);
          partial_batch_size = element.dim_size(0);

          // Allocate tensor sized to batch on first iteration
          if (it_partial == partials.begin()) {
            TensorShape shape = element.shape();
            shape.set_dim(0, batch_size);
            Tensor output(ctx->allocator({}), element.dtype(), shape);
            out_tensors->emplace_back(std::move(output));
          }

          // Copy partial batch to the output batch
          TF_RETURN_IF_ERROR(
              CopyElementsToParent(element, &out_tensors->at(i), batch_index));
        }
        batch_index += partial_batch_size;
      }
      return Status::OK();
    }

    template <typename T>
    Status HandleElementsToParent(const Tensor& element, Tensor* parent,
                                  int64 index) {
      // TODO: look into removing this loop, move tensor instead of copy
      for (int64 i = 0; i < element.dim_size(0); ++i) {
        parent->flat_outer_dims<T>().chip(index + i, 0) =
            element.flat_outer_dims<T>().chip(i, 0);
      }
      return Status::OK();
    }

    Status CopyElementsToParent(const Tensor& element, Tensor* parent,
                                int64 index) {
#define HANDLE_TYPE(T)                                                   \
  case DataTypeToEnum<T>::value: {                                       \
    return HandleElementsToParent<T>(std::move(element), parent, index); \
  }

      switch (element.dtype()) {
        TF_CALL_ALL_TYPES(HANDLE_TYPE);
        TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION == 3
        TF_CALL_uint32(HANDLE_TYPE);
        TF_CALL_uint64(HANDLE_TYPE);
#endif
#undef HANDLE_TYPE
        default:
          return errors::Unimplemented(
              "CopyElementsToParent Unhandled data type: ", element.dtype());
      }
    }

   protected:
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return errors::Unimplemented("SaveInternal is currently not supported");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented(
          "RestoreInternal is currently not supported");
    }

    // Setup Arrow record batch consumer and initialze current_batch_
    virtual Status SetupStreamsLocked(Env* env)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

    // Get the next Arrow record batch, if available. If not then
    // current_batch_ will be set to nullptr to indicate no further batches.
    virtual Status NextStreamLocked(Env* env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      current_batch_ = nullptr;
      current_row_idx_ = 0;
      return Status::OK();
    }

    // Reset the Arrow record batch consumer when done with batches.
    virtual void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      // This is the final state of the iterator after end_of_sequence=true
      current_batch_ = nullptr;
      current_row_idx_ = 1;
    }

    // Check columns of batch in stream are expected data type
    Status CheckBatchColumnTypes(std::shared_ptr<arrow::RecordBatch> batch) {
      for (size_t i = 0; i < this->dataset()->columns_.size(); ++i) {
        int32 col = this->dataset()->columns_[i];
        DataType dt = this->dataset()->output_types_[i];
        std::shared_ptr<arrow::Array> arr = batch->column(col);
        TF_RETURN_IF_ERROR(ArrowUtil::CheckArrayType(arr->type(), dt));
      }
      return Status::OK();
    }

    mutex mu_;
    std::shared_ptr<arrow::RecordBatch> current_batch_ TF_GUARDED_BY(mu_) =
        nullptr;
    int64_t current_row_idx_ TF_GUARDED_BY(mu_) = 0;
  };

  const std::vector<int32> columns_;
  const int64 batch_size_;
  const ArrowBatchMode batch_mode_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

// Abstract base class to define an Arrow OpKernel with output_types and
// output_shapes attributes, and list of column indices. Implementations
// will define how to create the Arrow Dataset.
class ArrowOpKernelBase : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  ArrowOpKernelBase(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    for (const DataType& dt : output_types_) {
      std::shared_ptr<arrow::DataType> arrow_type;
      OP_REQUIRES_OK(ctx, ArrowUtil::GetArrowType(dt, &arrow_type));
    }
    for (const PartialTensorShape& pts : output_shapes_) {
      OP_REQUIRES(ctx, -1 <= pts.dims() && pts.dims() <= 2,
                  errors::InvalidArgument("Output shape must be a scalar, "
                                          "vector, matrix or unknown"));
    }
  }

 private:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* columns_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("columns", &columns_tensor));
    OP_REQUIRES(
        ctx, columns_tensor->dims() <= 1,
        errors::InvalidArgument("`columns` must be a scalar or a vector."));

    std::vector<int32> columns;
    columns.reserve(columns_tensor->NumElements());
    for (int32 i = 0; i < static_cast<int32>(columns_tensor->NumElements());
         ++i) {
      columns.push_back(columns_tensor->flat<int32>()(i));
    }

    int64 batch_size;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size", &batch_size));

    tstring batch_mode_str;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "batch_mode", &batch_mode_str));
    ArrowBatchMode batch_mode;
    OP_REQUIRES_OK(ctx, GetBatchMode(batch_mode_str, &batch_mode));

    ArrowDatasetBase* arrow_output;
    MakeArrowDataset(ctx, columns, batch_size, batch_mode, output_types_,
                     output_shapes_, &arrow_output);
    *output = arrow_output;
  }

 protected:
  // Define to construct an implementation of ArrowDatasetBase
  virtual void MakeArrowDataset(
      OpKernelContext* ctx, const std::vector<int32>& columns,
      const int64 batch_size, const ArrowBatchMode batch_mode,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) = 0;

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class ArrowS3DatasetOp : public ArrowOpKernelBase {
 public:
  explicit ArrowS3DatasetOp(OpKernelConstruction* ctx)
      : ArrowOpKernelBase(ctx) {}

  virtual void MakeArrowDataset(
      OpKernelContext* ctx, const std::vector<int32>& columns,
      const int64 batch_size, const ArrowBatchMode batch_mode,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      ArrowDatasetBase** output) override {
    tstring aws_access_key;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "aws_access_key",
                                                     &aws_access_key));

    tstring aws_secret_key;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "aws_secret_key",
                                                     &aws_secret_key));

    tstring aws_endpoint_override;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<tstring>(ctx, "aws_endpoint_override",
                                                &aws_endpoint_override));

    const Tensor* parquet_files_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("parquet_files", &parquet_files_tensor));
    OP_REQUIRES(
        ctx, parquet_files_tensor->dims() <= 1,
        errors::InvalidArgument("`parquet_files` must be a scalar or vector."));
    std::vector<string> parquet_files;
    parquet_files.reserve(parquet_files_tensor->NumElements());
    for (int i = 0; i < parquet_files_tensor->NumElements(); ++i) {
      parquet_files.push_back(parquet_files_tensor->flat<tstring>()(i));
    }

    const Tensor* column_names_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("column_names", &column_names_tensor));
    OP_REQUIRES(
        ctx, column_names_tensor->dims() <= 1,
        errors::InvalidArgument("`column_names` must be a scalar or vector."));
    std::vector<string> column_names;
    column_names.reserve(column_names_tensor->NumElements());
    for (int i = 0; i < column_names_tensor->NumElements(); ++i) {
      column_names.push_back(column_names_tensor->flat<tstring>()(i));
    }

    std::vector<int32> column_cols(column_names.size());
    std::iota(column_cols.begin(), column_cols.end(), 0);

    tstring filter;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "filter", &filter));

    bool same_schema = true;
    OP_REQUIRES_OK(
        ctx, data::ParseScalarArgument<bool>(ctx, "same_schema", &same_schema));
    *output = new Dataset(ctx, aws_access_key, aws_secret_key,
                          aws_endpoint_override, parquet_files, column_names,
                          filter, same_schema, column_cols, batch_size,
                          batch_mode, output_types_, output_shapes_);
  }

 private:
  class Dataset : public ArrowDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::string& aws_access_key,
            const std::string& aws_secret_key,
            const std::string& aws_endpoint_override,
            const std::vector<std::string>& parquet_files,
            const std::vector<std::string>& column_names,
            const std::string& filter, const bool same_schema,
            const std::vector<int32> columns, const int64 batch_size,
            const ArrowBatchMode batch_mode, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : ArrowDatasetBase(ctx, columns, batch_size, batch_mode, output_types,
                           output_shapes),
          aws_access_key_(aws_access_key),
          aws_secret_key_(aws_secret_key),
          aws_endpoint_override_(aws_endpoint_override),
          parquet_files_(parquet_files),
          column_names_(column_names),
          filter_(filter),
          same_schema_(same_schema) {}

    string DebugString() const override { return "ArrowS3DatasetOp::Dataset"; }
    Status InputDatasets(std::vector<const DatasetBase*>* inputs) const {
      return Status::OK();
    }
    Status CheckExternalState() const override { return Status::OK(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* aws_access_key = nullptr;
      tstring access_key = aws_access_key_;
      TF_RETURN_IF_ERROR(b->AddScalar(access_key, &aws_access_key));
      Node* aws_secret_key = nullptr;
      tstring secret_key = aws_secret_key_;
      TF_RETURN_IF_ERROR(b->AddScalar(secret_key, &aws_secret_key));
      Node* aws_endpoint_override = nullptr;
      tstring endpoint_override = aws_endpoint_override_;
      TF_RETURN_IF_ERROR(
          b->AddScalar(endpoint_override, &aws_endpoint_override));
      Node* parquet_files = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(parquet_files_, &parquet_files));
      Node* column_names = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(column_names_, &column_names));
      Node* same_schema = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(same_schema_, &same_schema));
      Node* columns = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(columns_, &columns));
      Node* filter = nullptr;
      tstring filter_str = filter_;
      TF_RETURN_IF_ERROR(b->AddScalar(filter_str, &filter));
      Node* batch_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
      Node* batch_mode = nullptr;
      tstring batch_mode_str;
      TF_RETURN_IF_ERROR(GetBatchModeStr(batch_mode_, &batch_mode_str));
      TF_RETURN_IF_ERROR(b->AddScalar(batch_mode_str, &batch_mode));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          {aws_access_key, aws_secret_key, aws_endpoint_override, parquet_files,
           column_names, filter, same_schema, columns, batch_size, batch_mode},
          output));
      return Status::OK();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::ArrowS3")}));
    }

   private:
    class Iterator : public ArrowBaseIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : ArrowBaseIterator<Dataset>(params) {}

     private:
      Status SetupStreamsLocked(Env* env)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        if (!s3fs_) {
          arrow::fs::EnsureS3Initialized();
          auto s3Options = arrow::fs::S3Options::FromAccessKey(
              dataset()->aws_access_key_, dataset()->aws_secret_key_);
          s3Options.endpoint_override = dataset()->aws_endpoint_override_;
          s3fs_ = arrow::fs::S3FileSystem::Make(s3Options).ValueOrDie();
        }

        auto filter_expr_ptr =
            const_cast<arrow::compute::Expression*>(&(dataset()->filter_expr_));

        // filter
        if (!dataset()->filter_.empty()) {
          TF_RETURN_IF_ERROR(
              ArrowUtil::ParseExpression(dataset()->filter_, *filter_expr_ptr));
        }

        TF_RETURN_IF_ERROR(ReadFile(current_file_idx_));

        // If Filter is enabled, the entire file may not meet the filter
        while (record_batches_.empty() &&
               ++current_file_idx_ < dataset()->parquet_files_.size()) {
          TF_RETURN_IF_ERROR(ReadFile(current_file_idx_));
        }

        if (!background_worker_) {
          background_worker_ =
              std::make_shared<BackgroundWorker>(env, "download_next_worker");
        }

        if (current_batch_idx_ < record_batches_.size()) {
          current_batch_ = record_batches_[current_batch_idx_];
        }

        if (current_file_idx_ + 1 < dataset()->parquet_files_.size()) {
          background_worker_->Schedule(std::bind(&Iterator::ReadFile, this,
                                                 current_file_idx_ + 1, true));
        }
        return Status::OK();
      }

      Status NextStreamLocked(Env* env)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::NextStreamLocked(env);
        if (++current_batch_idx_ < record_batches_.size()) {
          current_batch_ = record_batches_[current_batch_idx_];
        } else if (++current_file_idx_ < dataset()->parquet_files_.size()) {
          current_batch_idx_ = 0;

          {
            mutex_lock lk(cv_mu_);
            while (!background_thread_finished_) {
              cv_.wait(lk);
            }
            if (!background_res_.ok()) {
              return background_res_;
            }
          }

          record_batches_.swap(next_record_batches_);

          // If Filter is enabled, the entire file may not meet the filter
          while (record_batches_.empty() &&
                 ++current_file_idx_ < dataset()->parquet_files_.size()) {
            TF_RETURN_IF_ERROR(ReadFile(current_file_idx_));
          }

          if (!record_batches_.empty()) {
            current_batch_ = record_batches_[current_batch_idx_];
          } else {
            current_batch_ = nullptr;
          }

          background_thread_finished_ = false;
          if (current_file_idx_ + 1 < dataset()->parquet_files_.size()) {
            background_worker_->Schedule(std::bind(
                &Iterator::ReadFile, this, current_file_idx_ + 1, true));
          }
        }
        return Status::OK();
      }

      void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
        ArrowBaseIterator<Dataset>::ResetStreamsLocked();
        current_file_idx_ = 0;
        current_batch_idx_ = 0;
        record_batches_.clear();
        next_record_batches_.clear();
      }

      Status ReadFile(int file_index, bool background = false)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        Status res = Status::OK();
        do {
          auto access_file_result =
              s3fs_->OpenInputFile(dataset()->parquet_files_[file_index]);
          if (!access_file_result.ok()) {
            res =
                errors::InvalidArgument(access_file_result.status().ToString());
            break;
          }

          auto access_file = access_file_result.ValueOrDie();

          parquet::ArrowReaderProperties properties;
          properties.set_use_threads(true);
          properties.set_pre_buffer(true);
          parquet::ReaderProperties parquet_properties =
              parquet::default_reader_properties();

          std::shared_ptr<parquet::arrow::FileReaderBuilder> builder =
              std::make_shared<parquet::arrow::FileReaderBuilder>();
          builder->Open(access_file, parquet_properties);

          std::unique_ptr<parquet::arrow::FileReader> reader;
          builder->properties(properties)->Build(&reader);

          if (column_indices_.empty() || !dataset()->same_schema_) {
            column_indices_.clear();
            std::shared_ptr<arrow::Schema> schema;
            reader->GetSchema(&schema);
            // check column name exist
            std::string err_column_names;
            for (const auto& name : dataset()->column_names_) {
              int fieldIndex = schema->GetFieldIndex(name);
              column_indices_.push_back(fieldIndex);
              if (-1 == fieldIndex) {
                err_column_names = err_column_names + " " + name;
              }
            }

            if (err_column_names.length() != 0) {
              res = errors::InvalidArgument(
                  "these column names don't exist: ", err_column_names,
                  " when read file: ", dataset()->parquet_files_[file_index]);
              break;
            }
          }
          // Read file columns and build a table
          std::shared_ptr<::arrow::Table> table;
          arrow::Status arrow_status =
              reader->ReadTable(column_indices_, &table);
          if (!arrow_status.ok()) {
            res = errors::Internal(arrow_status.ToString());
            break;
          }
          // Convert the table to a sequence of batches
          std::shared_ptr<arrow::RecordBatchReader> batch_reader =
              std::make_shared<arrow::TableBatchReader>(table);
          std::shared_ptr<arrow::RecordBatch> batch = nullptr;

          // filter
          if (!dataset()->filter_.empty()) {
            auto scanner_builder =
                arrow::dataset::ScannerBuilder::FromRecordBatchReader(
                    batch_reader);
            scanner_builder->Filter(dataset()->filter_expr_);
            auto scanner_result = scanner_builder->Finish();
            if (!scanner_result.ok()) {
              res = errors::Internal(scanner_result.status().ToString());
              break;
            }
            auto scanner = scanner_result.ValueOrDie();
            auto batch_reader_result = scanner->ToRecordBatchReader();
            if (!batch_reader_result.ok()) {
              res = errors::Internal(batch_reader_result.status().ToString());
              break;
            }
            batch_reader = batch_reader_result.ValueOrDie();
          }

          arrow_status = batch_reader->ReadNext(&batch);
          if (!arrow_status.ok()) {
            res = errors::Internal(arrow_status.ToString());
            break;
          }
          res = CheckBatchColumnTypes(batch);
          if (!res.ok()) {
            break;
          }
          next_record_batches_.clear();
          while (batch != nullptr) {
            if (batch->num_rows() != 0) {
              if (!background) {
                record_batches_.emplace_back(batch);
              } else {
                next_record_batches_.emplace_back(batch);
              }
            }
            arrow_status = batch_reader->ReadNext(&batch);
            if (!arrow_status.ok()) {
              res = errors::Internal(arrow_status.ToString());
              break;
            }
          }
        } while (0);

        if (background) {
          mutex_lock lk(cv_mu_);
          background_thread_finished_ = true;
          background_res_ = res;
          cv_.notify_all();
        }

        return res;
      }

      size_t current_file_idx_ TF_GUARDED_BY(mu_) = 0;
      size_t current_batch_idx_ TF_GUARDED_BY(mu_) = 0;
      std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_
          TF_GUARDED_BY(mu_);
      std::vector<std::shared_ptr<arrow::RecordBatch>> next_record_batches_
          TF_GUARDED_BY(mu_);
      std::shared_ptr<arrow::fs::S3FileSystem> s3fs_ TF_GUARDED_BY(mu_) =
          nullptr;
      std::vector<int> column_indices_ TF_GUARDED_BY(mu_);
      std::shared_ptr<BackgroundWorker> background_worker_ = nullptr;
      mutex cv_mu_;
      condition_variable cv_;
      bool background_thread_finished_ = false;
      Status background_res_ = Status::OK();
    };

    const std::string aws_access_key_;
    const std::string aws_secret_key_;
    const std::string aws_endpoint_override_;
    const std::vector<std::string> parquet_files_;
    const std::vector<std::string> column_names_;
    const std::string filter_;
    const bool same_schema_;
    arrow::compute::Expression filter_expr_;
  };
};  // class ArrowS3DatasetOp

REGISTER_KERNEL_BUILDER(Name("IO>ArrowS3Dataset").Device(DEVICE_CPU),
                        ArrowS3DatasetOp);

}  // namespace data
}  // namespace tensorflow
