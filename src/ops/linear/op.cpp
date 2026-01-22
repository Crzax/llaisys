#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Validate input tensors
    CHECK_ARGUMENT(in->ndim() == 2, "input must be 2-D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight must be 2-D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "output must be 2-D tensor");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "output and input must have same dtype");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "output and weight must have same dtype");

    size_t M = in->shape()[0];     // batch size
    size_t K = in->shape()[1];     // input features
    size_t N = weight->shape()[0]; // output features

    CHECK_ARGUMENT(weight->shape()[1] == K, "weight shape[1] must match input shape[1]");
    CHECK_ARGUMENT(out->shape()[0] == M, "output shape[0] must match input shape[0]");
    CHECK_ARGUMENT(out->shape()[1] == N, "output shape[1] must match weight shape[0]");

    // Validate bias if provided
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1, "bias must be 1-D tensor");
        CHECK_ARGUMENT(bias->shape()[0] == N, "bias shape[0] must match output features");
        CHECK_ARGUMENT(bias->dtype() == out->dtype(), "bias and output must have same dtype");
        CHECK_ARGUMENT(bias->isContiguous(), "bias must be contiguous");
    }

    // Ensure tensors are contiguous for efficient access
    CHECK_ARGUMENT(in->isContiguous(), "input must be contiguous");
    CHECK_ARGUMENT(weight->isContiguous(), "weight must be contiguous");
    CHECK_ARGUMENT(out->isContiguous(), "output must be contiguous");

    auto device = out->deviceType();
    CHECK_ARGUMENT(device == in->deviceType() && device == weight->deviceType(),
                   "all tensors must be on the same device");
    if (bias != nullptr) {
        CHECK_ARGUMENT(device == bias->deviceType(), "bias must be on the same device");
    }

    std::byte *out_ptr = out->data();
    const std::byte *in_ptr = in->data();
    const std::byte *weight_ptr = weight->data();
    const std::byte *bias_ptr = bias ? bias->data() : nullptr;

    switch (device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out_ptr, in_ptr, weight_ptr, bias_ptr, out->dtype(), M, N, K);
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
