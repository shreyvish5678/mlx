// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_quantized_attention.h"

#define instantiate_quantized_attention(tname, type, bq, bd, wm)        \
  template [[host_name("steel_quantized_attention_" #tname              \
                       "_bq" #bq "_bk16_bd" #bd "_wm" #wm "_wn1")]]  \
  [[kernel]] decltype(quantized_attention<type, bq, 16, bd, wm, 1>)     \
      quantized_attention<type, bq, 16, bd, wm, 1>;

instantiate_quantized_attention(float32, float, 32, 256, 4)
instantiate_quantized_attention(float16, half, 32, 256, 4)
instantiate_quantized_attention(bfloat16, bfloat16_t, 32, 256, 4)
instantiate_quantized_attention(float32, float, 8, 512, 1)
instantiate_quantized_attention(float16, half, 8, 512, 1)
instantiate_quantized_attention(bfloat16, bfloat16_t, 8, 512, 1)
// clang-format on
