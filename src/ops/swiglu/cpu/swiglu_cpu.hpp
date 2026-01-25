#pragma onece
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, std::byte *gate, std::byte *up, llaisysDataType_t type,
            const size_t seqlen, const size_t intermedia_size);
}