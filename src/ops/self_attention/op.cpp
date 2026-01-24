#include "op.hpp"
#include "../../utils/check.hpp"
#include "cpu/self_attetion_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous(), "attn_val tensor must be contiguous");
    ASSERT(q->isContiguous(), "q tensor must be contiguous");
    ASSERT(k->isContiguous(), "k tensor must be contiguous");
    ASSERT(v->isContiguous(), "v tensor must be contiguous");
    ASSERT(attn_val->ndim() == 3, "attn_val must be 3D [seqlen, nhead, dv]");
    ASSERT(q->ndim() == 3, "q must be 3D [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "k must be 3D [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "v must be 3D [total_len, nkvhead, dv]");

    auto val_shape = attn_val->shape();
    auto q_shape = q->shape();
    auto k_shape = k->shape();
    auto v_shape = v->shape();
    size_t seqlen = val_shape[0];
    size_t nh = val_shape[1];
    size_t dv = val_shape[2];
    size_t total_len = k_shape[0];
    size_t nkvh = k_shape[1];
    size_t d = k_shape[2];

    ASSERT(seqlen == q_shape[0], "attn_val's seq_len must equal to q's");
    ASSERT(nh == q_shape[1], "attn_val's nh must equal to q's");
    ASSERT(dv == v_shape[2], "attn_val's dv must equal to v's");
    ASSERT(total_len == v_shape[0], "k's total_len must equal to v's");
    ASSERT(nkvh == v_shape[1], "k's nkvh must equal to v's");
    ASSERT(d == q_shape[2], "k's d must equal to q's");

    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attetion(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), scale,
                                  seqlen, total_len, nh, nkvh, d, dv);
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
