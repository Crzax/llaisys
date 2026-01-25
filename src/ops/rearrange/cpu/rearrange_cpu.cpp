#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void rearrange_(T *out, const T *in, const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &out_strides,
                const std::vector<ptrdiff_t> &in_strides) {
    size_t ndim = shape.size();
    size_t numel = 1;
    for (auto s : shape)
        numel *= s;

    if (numel == 0)
        return;
    if (ndim == 0) {
        *out = *in;
        return;
    }

    std::vector<size_t> index(ndim, 0);
    T *out_ptr = out;
    const T *in_ptr = in;

    for (size_t i = 0; i < numel; ++i) {
        *out_ptr = *in_ptr;

        for (ptrdiff_t d = ndim - 1; d >= 0; --d) {
            index[d]++;
            if (index[d] < shape[d]) {
                in_ptr += in_strides[d];
                out_ptr += out_strides[d];
                break;
            }
            index[d] = 0;
            in_ptr -= (shape[d] - 1) * in_strides[d];
            out_ptr -= (shape[d] - 1) * out_strides[d];
        }
    }
}

void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides, const std::vector<ptrdiff_t> &in_strides,
               llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rearrange_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                          shape, out_strides, in_strides);
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
                                    reinterpret_cast<const llaisys::bf16_t *>(in), shape, out_strides, in_strides);
        break;
    case LLAISYS_DTYPE_F16:
        rearrange_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
                                    reinterpret_cast<const llaisys::fp16_t *>(in), shape, out_strides, in_strides);
        break;
    case LLAISYS_DTYPE_I64:
        rearrange_<int64_t>(reinterpret_cast<int64_t *>(out),
                            reinterpret_cast<const int64_t *>(in), shape, out_strides, in_strides);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
