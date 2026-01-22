#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, const float eps, size_t M, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        // Compute the sum of X_j ^ 2
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float x = llaisys::utils::cast<float>(in[i * N + j]);
                sum += x * x;
            } else {
                sum += in[i * N + j] * in[i * N + j];
            }
        }
        // Compute Y
        float divided = sqrt(sum / N + eps);
        for (size_t j = 0; j < N; ++j) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * N + j] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in[i * N + j]) *
                                                         llaisys::utils::cast<float>(weight[j]) / divided);
            } else {
                out[i * N + j] = in[i * N + j] * weight[j] / divided;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float eps,
              llaisysDataType_t type, size_t M, size_t N) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         eps,
                         M, N);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         eps,
                         M, N);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         eps,
                         M, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
