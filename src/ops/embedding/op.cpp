#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // Validate input tensors
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "index must be int64 type");
    CHECK_ARGUMENT(index->ndim() == 1, "index must be 1-D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight must be 2-D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "output must be 2-D tensor");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "output and weight must have same dtype");

    size_t num_indices = index->shape()[0];
    size_t embed_dim = weight->shape()[1];

    CHECK_ARGUMENT(out->shape()[0] == num_indices, "output shape[0] must match index length");
    CHECK_ARGUMENT(out->shape()[1] == embed_dim, "output shape[1] must match weight shape[1]");

    // Ensure tensors are contiguous for efficient access
    CHECK_ARGUMENT(index->isContiguous(), "index must be contiguous");
    CHECK_ARGUMENT(weight->isContiguous(), "weight must be contiguous");
    CHECK_ARGUMENT(out->isContiguous(), "output must be contiguous");

    auto device = out->deviceType();
    CHECK_ARGUMENT(device == index->deviceType() && device == weight->deviceType(),
                   "all tensors must be on the same device");

    std::byte *out_ptr = out->data();
    const std::byte *index_ptr = index->data();
    const std::byte *weight_ptr = weight->data();

    switch (device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out_ptr, index_ptr, weight_ptr, out->dtype(), num_indices, embed_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
