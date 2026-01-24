#include "self_attetion_cpu.hpp"
#include "../../../utils.hpp"
#include <limits>
#include <vector>
#include <cmath>

template <typename T>
void self_attetion_(T *atten_val, const T *q, const T *k, const T *v, const float scale,
                    const size_t seq_len, const size_t total_len, const size_t nh, const size_t nkvh, const size_t d, const size_t dv) {
    size_t group_size = nh / nkvh;
    size_t start_pos = total_len - seq_len;
    size_t q_s_stride = nh * d;
    size_t q_h_stride = d;
    size_t k_t_stride = nkvh * d;
    size_t k_h_stride = d;
    size_t v_t_stride = nkvh * dv;
    size_t v_h_stride = dv;
    size_t out_s_stride = nh * dv;
    size_t out_h_stride = dv;
    std::vector<float> scores(total_len);
    std::vector<float> val_acc(dv);
    for (size_t i = 0; i < seq_len; ++i) {
        size_t pos = start_pos + i;
        for (size_t h = 0; h < nh; ++h) {
            size_t kv_h = h / group_size;
            const T *q_vec = q + i * q_s_stride + h * q_h_stride;
            float max_score = -std::numeric_limits<float>::infinity();

            // 1. Attention Score Q @ K^T
            for (size_t t = 0; t < total_len; ++t) {
                if (t > pos) {
                    scores[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const T *k_vec = k + t * k_t_stride + kv_h * k_h_stride;
                float score = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    score += llaisys::utils::cast<float>(q_vec[j]) * llaisys::utils::cast<float>(k_vec[j]);
                }
                score *= scale;
                scores[t] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (size_t t = 0; t <= pos; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }

            // 3. Weighted Sum
            std::fill(val_acc.begin(), val_acc.end(), 0.0f);
            for (size_t t = 0; t <= pos; ++t) {
                const T *v_vec = v + t * v_t_stride + kv_h * v_h_stride;
                float prob = scores[t] / sum_exp;
                for (size_t j = 0; j < dv; ++j) {
                    val_acc[j] += prob * llaisys::utils::cast<float>(v_vec[j]);
                }
            }

            // Write output
            T *out_vec = atten_val + i * out_s_stride + h * out_h_stride;
            for (size_t j = 0; j < dv; ++j) {
                out_vec[j] = llaisys::utils::cast<T>(val_acc[j]);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attetion(std::byte *atten_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, const float scale,
                   const size_t seq_len, const size_t total_len, const size_t nh, const size_t nkvh, const size_t d, const size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attetion_(reinterpret_cast<float *>(atten_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                              scale, seq_len, total_len, nh, nkvh, d, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attetion_(reinterpret_cast<llaisys::bf16_t *>(atten_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                              scale, seq_len, total_len, nh, nkvh, d, dv);
    case LLAISYS_DTYPE_F16:
        return self_attetion_(reinterpret_cast<llaisys::fp16_t *>(atten_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                              scale, seq_len, total_len, nh, nkvh, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}