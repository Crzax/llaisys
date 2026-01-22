#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // Validate input tensors
    CHECK_ARGUMENT(in->ndim() == 2, "input must be 2-D tensor");
    CHECK_ARGUMENT(weight->ndim() == 1, "weight must be 1-D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "output must be 2-D tensor");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "output and input must have same dtype");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "output and weight must have same dtype");

    size_t M = in->shape()[0];
    size_t N = in->shape()[1];

    CHECK_ARGUMENT(out->shape()[0] == M, "output shape[0] must match input shape[0]");
    CHECK_ARGUMENT(out->shape()[1] == N, "output shape[1] must match input shape[1]");
    CHECK_ARGUMENT(weight->shape()[0] == N, "weight shape must match input shape[1]");

    // Ensure tensors are contiguous for efficient access
    CHECK_ARGUMENT(in->isContiguous(), "input must be contiguous");
    CHECK_ARGUMENT(weight->isContiguous(), "weight must be contiguous");
    CHECK_ARGUMENT(out->isContiguous(), "output must be contiguous");

    auto device = out->deviceType();
    CHECK_ARGUMENT(device == in->deviceType() && device == weight->deviceType(),
                   "all tensors must be on the same device");

    std::byte *out_ptr = out->data();
    const std::byte *in_ptr = in->data();
    const std::byte *weight_ptr = weight->data();

    switch (device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out_ptr, in_ptr, weight_ptr, eps, out->dtype(), M, N);
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
