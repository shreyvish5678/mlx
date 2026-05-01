// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_quantized_attention.h"

#define instantiate_quantized_attention(tname, type)                    \
  template [[host_name("steel_quantized_attention_" #tname              \
                       "_bq32_bk16_bd256_wm4_wn1")]]                   \
  [[kernel]] decltype(quantized_attention<type, 32, 16, 256, 4, 1>)     \
      quantized_attention<type, 32, 16, 256, 4, 1>;

instantiate_quantized_attention(float32, float)
instantiate_quantized_attention(float16, half)
instantiate_quantized_attention(bfloat16, bfloat16_t)
// clang-format on
