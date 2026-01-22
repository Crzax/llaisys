#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
    // Matrix multiplication: out[M, N] = in[M, K] @ weight[N, K]^T
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // Use float32 accumulator for better precision
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += llaisys::utils::cast<float>(in[i * K + k]) *
                           llaisys::utils::cast<float>(weight[j * K + k]);
                }
                // Add bias if provided
                if (bias != nullptr) {
                    sum += llaisys::utils::cast<float>(bias[j]);
                }
                out[i * N + j] = llaisys::utils::cast<T>(sum);
            } else {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < K; ++k) {
                    sum += in[i * K + k] * weight[j * K + k];
                }
                // Add bias if provided
                if (bias != nullptr) {
                    sum += bias[j];
                }
                out[i * N + j] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       bias ? reinterpret_cast<const float *>(bias) : nullptr,
                       M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                       M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                       M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
