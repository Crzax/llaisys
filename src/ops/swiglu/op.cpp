#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->isContiguous(), "out must be contiguous");
    ASSERT(gate->isContiguous(), "gate must be contiguous");
    ASSERT(up->isContiguous(), "up must be contiguous");
    ASSERT(out->ndim() == 2, "out must be 2D [seqlen, intermediate_size]");
    ASSERT(gate->ndim() == 2, "gate must be 2D [seqlen, intermediate_size]");
    ASSERT(up->ndim() == 2, "up must be 2D [seqlen, intermediate_size]");

    auto out_shape = out->shape();
    auto gate_shape = gate->shape();
    auto up_shape = up->shape();
    size_t seqlen = out_shape[0];
    size_t intermedia_size = out_shape[1];

    ASSERT(seqlen == gate_shape[0], "gate's seqlen must equal to out's");
    ASSERT(seqlen == up_shape[0], "up's seqlen must equal to out's");
    ASSERT(intermedia_size == gate_shape[1], "gate's intermedia_size must equal to out's");
    ASSERT(intermedia_size == up_shape[1], "up's intermedia_size must equal to out's");

    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(),
                           seqlen, intermedia_size);
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
