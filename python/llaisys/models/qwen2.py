from typing import Sequence
import json
import ctypes
import numpy as np
import safetensors.torch
import torch
from pathlib import Path

from ..libllaisys import LIB_LLAISYS, DeviceType
from ..libllaisys import load_qwen2
from ..libllaisys.models.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, LlaisysQwen2Model

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        with open(config_path, "r") as f:
            config = json.load(f)

        self.eos_token_id = config.get("eos_token_id", 151643)

        # Prepare Meta
        meta = LlaisysQwen2Meta()
        dtype_map = {"float32": 13, "float16": 12, "bfloat16": 19}
        meta.dtype = dtype_map.get(config.get("torch_dtype", "float32"), 13)
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config.get("num_key_value_heads", meta.nh)
        meta.dh = meta.hs // meta.nh
        # intermediate_size
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 8192) # Default 8192?
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = self.eos_token_id

        # Create Model
        device_ids = (ctypes.c_int * 1)(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            device.value,
            device_ids,
            1
        )

        # Get Weights structure
        self.weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model).contents

        # Load Weights
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_data = f.get_tensor(key)
                    self._load_tensor(key, tensor_data)

    def _load_tensor(self, key, data):
        ptr = self._get_weight_ptr(key)
        if ptr:
            if not data.is_contiguous():
                data = data.contiguous()
            
            data_ptr = ctypes.c_void_p(data.data_ptr())
            LIB_LLAISYS.tensorLoad(ptr, data_ptr)

    def _get_weight_ptr(self, key):
        w = self.weights
        if key == "model.embed_tokens.weight": return w.in_embed
        if key == "lm_head.weight": return w.out_embed
        if key == "model.norm.weight": return w.out_norm_w
        
        parts = key.split('.')
        # model.layers.0.self_attn.q_proj.weight
        if len(parts) >= 5 and parts[0] == "model" and parts[1] == "layers":
             try:
                 layer_idx = int(parts[2])
                 module_part = parts[3]
                 
                 if module_part == "input_layernorm" and parts[4] == "weight":
                     return w.attn_norm_w[layer_idx]
                 
                 if module_part == "post_attention_layernorm" and parts[4] == "weight":
                     return w.mlp_norm_w[layer_idx]
                     
                 if module_part == "self_attn":
                     proj = parts[4] # q_proj
                     type_ = parts[5] # weight/bias
                     
                     if type_ == "weight":
                         if proj == "q_proj": return w.attn_q_w[layer_idx]
                         if proj == "k_proj": return w.attn_k_w[layer_idx]
                         if proj == "v_proj": return w.attn_v_w[layer_idx]
                         if proj == "o_proj": return w.attn_o_w[layer_idx]
                     elif type_ == "bias":
                         if proj == "q_proj": return w.attn_q_b[layer_idx]
                         if proj == "k_proj": return w.attn_k_b[layer_idx]
                         if proj == "v_proj": return w.attn_v_b[layer_idx]
                         # o_proj bias not in struct, ignore
                         
                 if module_part == "mlp":
                     proj = parts[4]
                     type_ = parts[5]
                     if type_ == "weight":
                         if proj == "gate_proj": return w.mlp_gate_w[layer_idx]
                         if proj == "up_proj": return w.mlp_up_w[layer_idx]
                         if proj == "down_proj": return w.mlp_down_w[layer_idx]

             except ValueError:
                 pass
        return None

    def __del__(self):
        if hasattr(self, "model") and self.model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        current_ids = list(inputs)
        
        for _ in range(max_new_tokens):
             ntoken = len(current_ids)
             inp_arr = (ctypes.c_int64 * ntoken)(*current_ids)
             
             next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, inp_arr, ntoken)
             
             current_ids.append(next_token)
             
             if next_token == self.eos_token_id:
                  break
        
        return current_ids
