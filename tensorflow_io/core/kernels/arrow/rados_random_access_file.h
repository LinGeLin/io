#ifndef __RADOS_RANDOM_ACCESS_FILE_H__
#define __RADOS_RANDOM_ACCESS_FILE_H__

#include "librados.h"
#include "arrow/io/api.h"

namespace tensorflow {
namespace data{

class RadosRandomAccessFile final : public arrow::io::RandomAccessFile {
public:
    RadosRandomAccessFile(const rados_ioctx_t& rados_ioctx,
                          const std::string& rados_oid,
                          const arrow::io::IOContext& io_context,
                          int64_t size = -1)
                          : rados_ioctx_(rados_ioctx),
                            rados_oid_(rados_oid),
                            io_context_(io_context),
                            content_length_(size) {}
    arrow::Status Init();

    arrow::Status CheckClosed() const;

    arrow::Status CheckPosition(int64_t position, const char* action) const;

    arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override;

    arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(const arrow::io::IOContext& io_context) override;

    arrow::Status Close() override;

    bool closed() const override;

    arrow::Result<int64_t> Tell() const override;

    arrow::Result<int64_t> GetSize() override;

    arrow::Status Seek(int64_t position) override;

    arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override;

    arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override;

    arrow::Result<int64_t> Read(int64_t nbytes, void* out) override;

    arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override;

protected:
    const rados_ioctx_t rados_ioctx_;
    const std::string rados_oid_;    
    const arrow::io::IOContext io_context_;

    bool closed_ = false;
    int64_t pos_ = 0;
    int64_t content_length_ = -1;
};  //class RadosRandomAccessFile


struct RadosOptions {
    RadosOptions(const std::string& rados_id,
                const std::string& rados_keyring,
                const std::string& rados_mon_host,
                const std::string& rados_pool_name = "iceberg",
                const std::string& rados_namespace = "")
                :   rados_id_(rados_id),
                    rados_keyring_(rados_keyring),
                    rados_mon_host_(rados_mon_host),
                    rados_pool_name_(rados_pool_name),
                    rados_namespace_(rados_namespace) {}
    std::string rados_id_;
    std::string rados_keyring_;
    std::string rados_mon_host_;
    std::string rados_pool_name_;
    std::string rados_namespace_;
};

class RadosFileSystem {
public:
    ~RadosFileSystem();
    arrow::Status Init();
    arrow::Result<std::shared_ptr<RadosRandomAccessFile>> OpenInputFile(const std::string& rados_oid);
    static arrow::Result<std::shared_ptr<RadosFileSystem>> Make(const RadosOptions& options,
                                                         const arrow::io::IOContext& = arrow::io::default_io_context());
protected:
    explicit RadosFileSystem(const RadosOptions& options, const arrow::io::IOContext& io_context) : 
                            rados_options_(options), io_context_(io_context) {}

private:
    RadosOptions rados_options_;
    const arrow::io::IOContext& io_context_;

    rados_t rados_cluster_;
    rados_ioctx_t rados_ioctx_;
};  // class RadosFileSystem

}   // namespace tensorflow
}   // namespace data

#endif