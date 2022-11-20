#include "arrow/buffer.h"
#include "arrow/util/future.h"
#include "arrow/util/logging.h"
#include "arrow/util/key_value_metadata.h"
#include "rados_random_access_file.h"
#include "iostream"


namespace tensorflow {
namespace data {

arrow::Status RadosRandomAccessFile::Init() {
    int err;
    uint64_t object_size;
    time_t last_modify_time;
    err = rados_stat(rados_ioctx_, rados_oid_.c_str(), &object_size, &last_modify_time);
    if (err < 0) {
        return arrow::Status::Invalid("Failed to get object's size");
    }
    content_length_ = object_size;
    return arrow::Status::OK();
}

arrow::Status RadosRandomAccessFile::CheckClosed() const {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed stream");
    }
    return arrow::Status::OK();
}

arrow::Status RadosRandomAccessFile::CheckPosition(int64_t position, const char* action) const {
    if (position < 0) {
      return arrow::Status::Invalid("Cannot ", action, " from negative position");
    }
    if (position > content_length_) {
      return arrow::Status::IOError("Cannot ", action, " past end of file");
    }
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> RadosRandomAccessFile::ReadMetadata() {
    return nullptr;
}

arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> RadosRandomAccessFile::ReadMetadataAsync(const arrow::io::IOContext& io_context) {
    // return nullptr;
    auto metadata = std::make_shared<const arrow::KeyValueMetadata>();
    return metadata;
}

arrow::Status RadosRandomAccessFile::Close() {
    closed_ = true;
    return arrow::Status::OK();
}

bool RadosRandomAccessFile::closed() const {
    return closed_;
}

arrow::Result<int64_t> RadosRandomAccessFile::Tell() const {
    RETURN_NOT_OK(CheckClosed());
    return pos_;
}

arrow::Result<int64_t> RadosRandomAccessFile::GetSize() {
    RETURN_NOT_OK(CheckClosed());
    return content_length_;
}

arrow::Status RadosRandomAccessFile::Seek(int64_t position) {
    RETURN_NOT_OK(CheckClosed());
    RETURN_NOT_OK(CheckPosition(position, "seek"));

    pos_ = position;
    return arrow::Status::OK();
}

arrow::Result<int64_t> RadosRandomAccessFile::ReadAt(int64_t position, int64_t nbytes, void* out) {
    RETURN_NOT_OK(CheckClosed());
    RETURN_NOT_OK(CheckPosition(position, "read"));

    nbytes = std::min(nbytes, content_length_ - position);
    if (nbytes == 0) {
        return 0;
    }
    int64_t count = rados_read(rados_ioctx_, rados_oid_.c_str(), (char*)out, nbytes, position);

    if (count < 0) {
        return arrow::Status::IOError("failed to read data");
    }

    return count;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> RadosRandomAccessFile::ReadAt(int64_t position, int64_t nbytes) {
    RETURN_NOT_OK(CheckClosed());
    RETURN_NOT_OK(CheckPosition(position, "read"));

    nbytes = std::min(nbytes, content_length_ - position);
    ARROW_ASSIGN_OR_RAISE(auto buf, AllocateResizableBuffer(nbytes, io_context_.pool()));
    if (nbytes > 0) {
        ARROW_ASSIGN_OR_RAISE(int64_t bytes_read,
                            ReadAt(position, nbytes, buf->mutable_data()));
        DCHECK_LE(bytes_read, nbytes);
        RETURN_NOT_OK(buf->Resize(bytes_read));
    }
    return std::move(buf);
}

arrow::Result<int64_t> RadosRandomAccessFile::Read(int64_t nbytes, void* out) {
    ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(pos_, nbytes, out));
    pos_ += bytes_read;
    return bytes_read;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> RadosRandomAccessFile::Read(int64_t nbytes) {
    ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
    pos_ += buffer->size();
    return std::move(buffer);
}


// RadosFileSystem

RadosFileSystem::~RadosFileSystem() {
    rados_ioctx_destroy(rados_ioctx_);
    rados_shutdown(rados_cluster_);
}

arrow::Status RadosFileSystem::Init() {
    int err;
    err = rados_create(&rados_cluster_, rados_options_.rados_id_.c_str());
    if (err < 0) {
        return arrow::Status::Invalid("cannot create a cluster handle");
    }

    err = rados_conf_set(rados_cluster_, "mon_host", rados_options_.rados_mon_host_.c_str());
    if (err < 0) {
        return arrow::Status::Invalid("Failed to set mon host");
    }

    err = rados_conf_set(rados_cluster_, "keyring", rados_options_.rados_keyring_.c_str());
    if (err < 0) {
        return arrow::Status::Invalid("Failed to set keyring");
    }

    err = rados_connect(rados_cluster_);
    if (err < 0) {
        return arrow::Status::Invalid("cannot connect to cluster");
    }


    err = rados_ioctx_create(rados_cluster_, rados_options_.rados_pool_name_.c_str(), &rados_ioctx_);
    if (err < 0) {
        return arrow::Status::Invalid("cannot open rados pool");
    }

    // rados_ioctx_set_namespace(rados_cluster_, rados_options_.rados_namespace_.c_str());

    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<RadosRandomAccessFile>> RadosFileSystem::OpenInputFile(const std::string& rados_oid) {
    auto ptr = std::make_shared<RadosRandomAccessFile>(rados_ioctx_, rados_oid, io_context_);
    RETURN_NOT_OK(ptr->Init());
    return ptr;
}

arrow::Result<std::shared_ptr<RadosFileSystem>> RadosFileSystem::Make(const RadosOptions& options, 
                                                               const arrow::io::IOContext& io_context) {
    std::shared_ptr<RadosFileSystem> ptr(new RadosFileSystem(options, io_context));
    RETURN_NOT_OK(ptr->Init());
    return ptr;
}

}   // namespace tensorflow
}   // namespace data