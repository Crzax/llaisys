#include "op.hpp"
#include "../../utils/check.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(in->isContiguous(), "Input tensor must be contiguous");
    ASSERT(out->isContiguous(), "Output tensor must be contiguous");
    ASSERT(pos_ids->isContiguous(), "Pos_ids tensor must be contiguous");

    // Check shapes
    // in: [seqlen, nhead, head_dim]
    // pos_ids: [seqlen]
    auto in_shape = in->shape();
    auto out_shape = out->shape();
    auto pos_shape = pos_ids->shape();

    ASSERT(in->ndim() == 3, "Input must be 3D [seqlen, nhead, head_dim]");
    CHECK_SAME_SHAPE(in_shape, out_shape);

    size_t seqlen = in_shape[0];
    size_t nhead = in_shape[1];
    size_t head_dim = in_shape[2];

    ASSERT(pos_ids->ndim() == 1, "Pos_ids must be 1D");
    ASSERT(pos_shape[0] == seqlen, "Pos_ids dim 0 must match seqlen");

    CHECK_SAME_DTYPE(in->dtype(), out->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Pos_ids must be int64");

    switch (in->deviceType()) {

    case LLAISYS_DEVICE_CPU:
        cpu::rope(out->data(), in->data(),
                  reinterpret_cast<const int64_t *>(pos_ids->data()),
                  theta, in->dtype(),
                  seqlen, nhead, head_dim);
        return;
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
