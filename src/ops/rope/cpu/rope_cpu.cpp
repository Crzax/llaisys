#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <cmath>

template <typename T>
void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                 size_t seqlen, size_t nhead, size_t head_dim) {
    size_t dim = head_dim;
    size_t half_dim = dim / 2;

    for (size_t s = 0; s < seqlen; ++s) {
        int64_t pos = pos_ids[s];
        for (size_t h = 0; h < nhead; ++h) {
            size_t offset = (s * nhead + h) * dim;
            const T *in_vec = in + offset;
            T *out_vec = out + offset;

            for (size_t i = 0; i < half_dim; ++i) {
                float freq = static_cast<float>(pos) / std::pow(theta, 2.0f * i / dim);
                float sin_val = std::sin(freq);
                float cos_val = std::cos(freq);

                float a = llaisys::utils::cast<float>(in_vec[i]);
                float b = llaisys::utils::cast<float>(in_vec[i + half_dim]);

                float a_prime = a * cos_val - b * sin_val;
                float b_prime = b * cos_val + a * sin_val;

                out_vec[i] = llaisys::utils::cast<T>(a_prime);
                out_vec[i + half_dim] = llaisys::utils::cast<T>(b_prime);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_kernel(reinterpret_cast<float *>(out),
                           reinterpret_cast<const float *>(in),
                           pos_ids, theta, seqlen, nhead, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_kernel(reinterpret_cast<llaisys::bf16_t *>(out),
                           reinterpret_cast<const llaisys::bf16_t *>(in),
                           pos_ids, theta, seqlen, nhead, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_kernel(reinterpret_cast<llaisys::fp16_t *>(out),
                           reinterpret_cast<const llaisys::fp16_t *>(in),
                           pos_ids, theta, seqlen, nhead, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
