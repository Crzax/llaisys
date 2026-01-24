#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attetion(std::byte *atten_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, const float scale,
                   const size_t seq_len, const size_t total_len, const size_t nh, const size_t nkvh, const size_t d, const size_t dv);
}