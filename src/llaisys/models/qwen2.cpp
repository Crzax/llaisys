#include "llaisys/models/qwen2.h"
#include "../llaisys_tensor.hpp"
#include "../../tensor/tensor.hpp"

// Ops
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include "../../core/llaisys_core.hpp"

#include <vector>
#include <iostream>
#include <cmath>

using namespace llaisys;

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device_type;
    int device_id;

    // The handles exposed to C API
    LlaisysQwen2Weights weights_handles;

    // We store the actual LlaisysTensor objects that the handles point to
    // to manage their lifetime properly if needed, though here we just allocate them on heap
    // and store pointers in weights_handles.

    // To facilitate cleanup, we can keep a vector of allocated LlaisysTensor*
    std::vector<LlaisysTensor *> allocated_tensors;

    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;

    int64_t current_pos = 0;

    // Temporary tensors for inference (reused to avoid allocation overhead if possible,
    // but shapes change dynamic with seq len in prefill vs decode).
    // For simplicity, we might create them on the fly or reuse only fixed size ones.

    LlaisysQwen2Model(const LlaisysQwen2Meta *m, llaisysDeviceType_t dev, int dev_id) {
        meta = *m;
        device_type = dev;
        device_id = dev_id;

        // Initialize weights structure
        init_weights();

        // Initialize KV Cache
        init_cache();
    }

    ~LlaisysQwen2Model() {
        for (auto t : allocated_tensors) {
            delete t;
        }
        if (weights_handles.attn_norm_w)
            delete[] weights_handles.attn_norm_w;
        if (weights_handles.attn_q_w)
            delete[] weights_handles.attn_q_w;
        if (weights_handles.attn_q_b)
            delete[] weights_handles.attn_q_b;
        if (weights_handles.attn_k_w)
            delete[] weights_handles.attn_k_w;
        if (weights_handles.attn_k_b)
            delete[] weights_handles.attn_k_b;
        if (weights_handles.attn_v_w)
            delete[] weights_handles.attn_v_w;
        if (weights_handles.attn_v_b)
            delete[] weights_handles.attn_v_b;
        if (weights_handles.attn_o_w)
            delete[] weights_handles.attn_o_w;
        if (weights_handles.mlp_norm_w)
            delete[] weights_handles.mlp_norm_w;
        if (weights_handles.mlp_gate_w)
            delete[] weights_handles.mlp_gate_w;
        if (weights_handles.mlp_up_w)
            delete[] weights_handles.mlp_up_w;
        if (weights_handles.mlp_down_w)
            delete[] weights_handles.mlp_down_w;
    }

    LlaisysTensor *create_tensor(const std::vector<size_t> &shape) {
        LlaisysTensor *lt = new LlaisysTensor();
        lt->tensor = Tensor::create(shape, meta.dtype, device_type, device_id);
        allocated_tensors.push_back(lt);
        return lt;
    }

    void init_weights() {
        weights_handles.in_embed = create_tensor({meta.voc, meta.hs});
        weights_handles.out_embed = create_tensor({meta.voc, meta.hs});
        weights_handles.out_norm_w = create_tensor({meta.hs});

        weights_handles.attn_norm_w = new llaisysTensor_t[meta.nlayer];
        weights_handles.attn_q_w = new llaisysTensor_t[meta.nlayer];
        weights_handles.attn_q_b = new llaisysTensor_t[meta.nlayer];
        weights_handles.attn_k_w = new llaisysTensor_t[meta.nlayer];
        weights_handles.attn_k_b = new llaisysTensor_t[meta.nlayer];
        weights_handles.attn_v_w = new llaisysTensor_t[meta.nlayer];
        weights_handles.attn_v_b = new llaisysTensor_t[meta.nlayer];
        weights_handles.attn_o_w = new llaisysTensor_t[meta.nlayer];

        weights_handles.mlp_norm_w = new llaisysTensor_t[meta.nlayer];
        weights_handles.mlp_gate_w = new llaisysTensor_t[meta.nlayer];
        weights_handles.mlp_up_w = new llaisysTensor_t[meta.nlayer];
        weights_handles.mlp_down_w = new llaisysTensor_t[meta.nlayer];

        for (size_t i = 0; i < meta.nlayer; ++i) {
            weights_handles.attn_norm_w[i] = create_tensor({meta.hs});
            weights_handles.attn_q_w[i] = create_tensor({meta.nh * meta.dh, meta.hs});
            weights_handles.attn_q_b[i] = create_tensor({meta.nh * meta.dh});
            weights_handles.attn_k_w[i] = create_tensor({meta.nkvh * meta.dh, meta.hs});
            weights_handles.attn_k_b[i] = create_tensor({meta.nkvh * meta.dh});
            weights_handles.attn_v_w[i] = create_tensor({meta.nkvh * meta.dh, meta.hs});
            weights_handles.attn_v_b[i] = create_tensor({meta.nkvh * meta.dh});
            weights_handles.attn_o_w[i] = create_tensor({meta.hs, meta.nh * meta.dh});

            weights_handles.mlp_norm_w[i] = create_tensor({meta.hs});
            weights_handles.mlp_gate_w[i] = create_tensor({meta.di, meta.hs});
            weights_handles.mlp_up_w[i] = create_tensor({meta.di, meta.hs});
            weights_handles.mlp_down_w[i] = create_tensor({meta.hs, meta.di});
        }
    }

    void init_cache() {
        for (size_t i = 0; i < meta.nlayer; ++i) {
            k_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id));
            v_cache.push_back(Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id));
        }
    }
};

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    int dev_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;
    return new LlaisysQwen2Model(meta, device, dev_id);
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model)
        delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model)
        return nullptr;
    return &model->weights_handles;
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || !token_ids || ntoken == 0)
        return -1;

    // Determine the number of new tokens to process
    size_t start_pos = model->current_pos;
    size_t seq_len = ntoken - start_pos;

    if (seq_len <= 0)
        return -1; // Should not happen in normal generation loop

    // Create tensor for input_ids
    // Provide a view on host memory? Or copy?
    // Tensor::load takes void* and copies.
    // So create a tensor on device and load.
    tensor_t input_ids_dev = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);

    // We assume token_ids is contiguous int64 array.
    input_ids_dev->load(token_ids + start_pos);

    // Create position ids
    // Shape: [seq_len]
    // Values: start_pos, start_pos+1, ...
    std::vector<int64_t> pos_ids_host(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
        pos_ids_host[i] = start_pos + i;

    tensor_t pos_ids = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
    pos_ids->load(pos_ids_host.data());

    // Embedding
    // out: [seq_len, hs]
    tensor_t x = Tensor::create({seq_len, model->meta.hs}, model->meta.dtype, model->device_type, model->device_id);

    // Assuming LlaisysTensor* -> tensor is accessible via ->tensor
    llaisys::ops::embedding(x, input_ids_dev, model->weights_handles.in_embed->tensor);

    // Loop layers
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        tensor_t residual = x;

        // 1. RMS Norm
        tensor_t hidden = Tensor::create({seq_len, model->meta.hs}, model->meta.dtype, model->device_type, model->device_id);
        llaisys::ops::rms_norm(hidden, x, model->weights_handles.attn_norm_w[i]->tensor, model->meta.epsilon);

        // 2. Self Attention
        // QKV Projections
        // Q: [seq_len, nh * dh]
        // K: [seq_len, nkvh * dh]
        // V: [seq_len, nkvh * dh]
        tensor_t q = Tensor::create({seq_len, model->meta.nh * model->meta.dh}, model->meta.dtype, model->device_type, model->device_id);
        tensor_t k = Tensor::create({seq_len, model->meta.nkvh * model->meta.dh}, model->meta.dtype, model->device_type, model->device_id);
        tensor_t v = Tensor::create({seq_len, model->meta.nkvh * model->meta.dh}, model->meta.dtype, model->device_type, model->device_id);

        llaisys::ops::linear(q, hidden, model->weights_handles.attn_q_w[i]->tensor, model->weights_handles.attn_q_b[i]->tensor);
        llaisys::ops::linear(k, hidden, model->weights_handles.attn_k_w[i]->tensor, model->weights_handles.attn_k_b[i]->tensor);
        llaisys::ops::linear(v, hidden, model->weights_handles.attn_v_w[i]->tensor, model->weights_handles.attn_v_b[i]->tensor);

        // Reshape for rope: [seq_len, nhead, dh]
        q = q->view({seq_len, model->meta.nh, model->meta.dh});
        k = k->view({seq_len, model->meta.nkvh, model->meta.dh});
        v = v->view({seq_len, model->meta.nkvh, model->meta.dh});

        // RoPE
        // In-place or new? RoPE op signature: rope(out, in, pos, theta).
        // Can be in-place if out==in? Check op implementation or assume safe if different.
        // Let's create new tensors to be safe or use same if op supports it.
        // Assuming op supports in-place if passed same ptr? Or create new.
        // Let's create new to be safe.
        tensor_t q_rope = Tensor::create(q->shape(), model->meta.dtype, model->device_type, model->device_id);
        tensor_t k_rope = Tensor::create(k->shape(), model->meta.dtype, model->device_type, model->device_id);

        llaisys::ops::rope(q_rope, q, pos_ids, model->meta.theta);
        llaisys::ops::rope(k_rope, k, pos_ids, model->meta.theta);

        // Update KV Cache
        // k_cache[i] shape: [maxseq, nkvh, dh]
        // Slice assumes dim 0 (seq).
        tensor_t k_slot = model->k_cache[i]->slice(0, start_pos, start_pos + seq_len);
        tensor_t v_slot = model->v_cache[i]->slice(0, start_pos, start_pos + seq_len);

        // Copy k_rope -> k_slot
        llaisys::ops::rearrange(k_slot, k_rope);
        llaisys::ops::rearrange(v_slot, v);

        // Get full active KV
        tensor_t k_full = model->k_cache[i]->slice(0, 0, start_pos + seq_len);
        tensor_t v_full = model->v_cache[i]->slice(0, 0, start_pos + seq_len);

        // Attention
        tensor_t attn_out = Tensor::create({seq_len, model->meta.nh, model->meta.dh}, model->meta.dtype, model->device_type, model->device_id);
        float scale = 1.0f / std::sqrt((float)model->meta.dh);
        llaisys::ops::self_attention(attn_out, q_rope, k_full, v_full, scale);

        // Output projection
        // Reshape attn_out to [seq_len, nh * dh]
        attn_out = attn_out->view({seq_len, model->meta.nh * model->meta.dh});
        // Linear wants matrix.

        tensor_t sa_out = Tensor::create({seq_len, model->meta.hs}, model->meta.dtype, model->device_type, model->device_id);
        // attn_o_w: [hs, nh*dh] (transposed logic in linear: xW^T).
        // W shape in linear is [out_features, in_features].
        // attn_o_w matches.
        // bias? struct has none. pass empty tensor?
        // linear(out, in, w, b). b can be null/empty?
        // assignment 2 says "bias (optional)".
        // need a way to pass no bias. maybe nullptr? or empty tensor?
        // Tensor::create({0}...) ?
        // Let's try passing nullptr equivalent or create a empty tensor.
        // Assuming linear op handles empty/null bias tensor wrapper.
        // Actually llaisys::ops::linear signature expects tensor_t (shared_ptr).
        // If I pass initialized shared_ptr but points to nothing? Or empty tensor.
        // Let's create a dummy empty tensor? Or maybe just use nullptr (shared_ptr default).
        tensor_t no_bias; // nullptr
        llaisys::ops::linear(sa_out, attn_out, model->weights_handles.attn_o_w[i]->tensor, no_bias);

        // Residual Add
        llaisys::ops::add(x, residual, sa_out);

        // 3. MLP
        residual = x;
        hidden = Tensor::create({seq_len, model->meta.hs}, model->meta.dtype, model->device_type, model->device_id);
        llaisys::ops::rms_norm(hidden, x, model->weights_handles.mlp_norm_w[i]->tensor, model->meta.epsilon);

        // Gate, Up
        tensor_t gate = Tensor::create({seq_len, model->meta.di}, model->meta.dtype, model->device_type, model->device_id);
        tensor_t up = Tensor::create({seq_len, model->meta.di}, model->meta.dtype, model->device_type, model->device_id);

        llaisys::ops::linear(gate, hidden, model->weights_handles.mlp_gate_w[i]->tensor, no_bias);
        llaisys::ops::linear(up, hidden, model->weights_handles.mlp_up_w[i]->tensor, no_bias);

        // Swiglu
        // In place? signature: swiglu(out, gate, up).
        // It computes out = swiglu(gate, up).
        tensor_t act = Tensor::create({seq_len, model->meta.di}, model->meta.dtype, model->device_type, model->device_id);
        llaisys::ops::swiglu(act, gate, up);

        // Down
        tensor_t mlp_out = Tensor::create({seq_len, model->meta.hs}, model->meta.dtype, model->device_type, model->device_id);
        llaisys::ops::linear(mlp_out, act, model->weights_handles.mlp_down_w[i]->tensor, no_bias);

        // Residual Add
        llaisys::ops::add(x, residual, mlp_out);
    }

    // Final Norm
    tensor_t final_hidden = Tensor::create({seq_len, model->meta.hs}, model->meta.dtype, model->device_type, model->device_id);
    llaisys::ops::rms_norm(final_hidden, x, model->weights_handles.out_norm_w->tensor, model->meta.epsilon);

    // Logits
    // Only separate last token?
    // If we only care about the last token generation.
    // slice last row.
    tensor_t last_token_hidden = final_hidden->slice(0, seq_len - 1, seq_len);
    // shape: [1, hs]

    tensor_t logits = Tensor::create({1, model->meta.voc}, model->meta.dtype, model->device_type, model->device_id);
    llaisys::ops::linear(logits, last_token_hidden, model->weights_handles.out_embed->tensor, tensor_t());

    // Argmax
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, model->device_type, model->device_id);
    tensor_t max_val = Tensor::create({1}, model->meta.dtype, model->device_type, model->device_id);

    llaisys::ops::argmax(max_idx, max_val, logits);

    // Read back result
    int64_t next_token;
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        next_token = *reinterpret_cast<int64_t *>(max_idx->data());
    } else {
        core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());
        core::context().runtime().api()->memcpy_sync(
            &next_token,
            max_idx->data(),
            sizeof(int64_t),
            LLAISYS_MEMCPY_D2H);
    }

    // Update model state
    model->current_pos += seq_len;

    return next_token;
}
}
