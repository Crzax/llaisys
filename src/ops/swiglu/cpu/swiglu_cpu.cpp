#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up,
             const size_t seqlen, const size_t intermedia_size) {
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t j = 0; j < intermedia_size; ++j) {
            size_t offset = i * intermedia_size + j;
            float gate_val = llaisys::utils::cast<float>(gate[offset]);
            out[offset] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(up[offset]) * gate_val / (1 + exp(-gate_val)));
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t type,
            const size_t seqlen, const size_t intermedia_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up),
                       seqlen, intermedia_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up),
                       seqlen, intermedia_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up),
                       seqlen, intermedia_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu