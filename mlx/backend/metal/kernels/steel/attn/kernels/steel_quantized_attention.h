// Copyright © 2025 Apple Inc.

#pragma once

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace metal;
using namespace mlx::steel;

constant bool qattn_align_Q [[function_constant(200)]];
constant bool qattn_align_K [[function_constant(201)]];
constant bool qattn_do_causal [[function_constant(301)]];
constant bool qattn_output_logsumexp [[function_constant(304)]];

struct QuantizedKVStrides {
  int64_t KQ_strides[3];
  int64_t KS_strides[3];
  int64_t KB_strides[3];
  int64_t VQ_strides[3];
  int64_t VS_strides[3];
  int64_t VB_strides[3];
};

struct QAttnMaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct QAttnSumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct QAttnMulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct QAttnExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct QAttnDivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short tgp_size,
    short group_size,
    short bits>
struct QAttnQuantizedBlockLoader {
  static_assert(bits == 4, "QAttnQuantizedBlockLoader only supports q4");
  static_assert(group_size == 64, "QAttnQuantizedBlockLoader only supports gs64");

  STEEL_CONST short pack_factor = 2;
  STEEL_CONST short packed_cols = BCOLS / pack_factor;
  STEEL_CONST short groups_per_row = BCOLS / group_size;
  STEEL_CONST short n_reads =
      (BROWS * packed_cols + tgp_size - 1) / tgp_size;

  const short thread_idx;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;
  threadgroup T* dst;

  QAttnQuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        src(src_),
        scales(scales_),
        biases(biases_),
        dst(dst_) {}

  METAL_FUNC void store_pack(short row, short pack_col, uchar packed) const {
    short group = (pack_col * pack_factor) / group_size;
    T scale = scales[row * groups_per_row + group];
    T bias = biases[row * groups_per_row + group];
    threadgroup T* out =
        dst + row * kDstStrRow + pack_col * pack_factor * kDstStrCol;
    out[0] = scale * T(packed & 0x0f) + bias;
    out[kDstStrCol] = scale * T((packed >> 4) & 0x0f) + bias;
  }

  METAL_FUNC void zero_pack(short row, short pack_col) const {
    threadgroup T* out =
        dst + row * kDstStrRow + pack_col * pack_factor * kDstStrCol;
    out[0] = T(0);
    out[kDstStrCol] = T(0);
  }

  METAL_FUNC void load_unsafe() const {
    for (short i = 0; i < n_reads; i++) {
      int linear = thread_idx + i * tgp_size;
      if (linear >= BROWS * packed_cols) {
        return;
      }
      short row = linear / packed_cols;
      short pack_col = linear - row * packed_cols;
      store_pack(row, pack_col, src[row * packed_cols + pack_col]);
    }
  }

  METAL_FUNC void load_safe(short valid_rows) const {
    for (short i = 0; i < n_reads; i++) {
      int linear = thread_idx + i * tgp_size;
      if (linear >= BROWS * packed_cols) {
        return;
      }
      short row = linear / packed_cols;
      short pack_col = linear - row * packed_cols;
      if (row < valid_rows) {
        store_pack(row, pack_col, src[row * packed_cols + pack_col]);
      } else {
        zero_pack(row, pack_col);
      }
    }
  }

  METAL_FUNC void next() {
    src += BROWS * packed_cols;
    scales += BROWS * groups_per_row;
    biases += BROWS * groups_per_row;
  }
};

template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
quantized_attention(
    const device T* Q [[buffer(0)]],
    const device uint32_t* KQ [[buffer(1)]],
    const device T* KS [[buffer(2)]],
    const device T* KB [[buffer(3)]],
    const device uint32_t* VQ [[buffer(4)]],
    const device T* VS [[buffer(5)]],
    const device T* VB [[buffer(6)]],
    device T* O [[buffer(7)]],
    const constant AttnParams* params [[buffer(8)]],
    const constant QuantizedKVStrides* qstrides [[buffer(9)]],
    device float* lse_out [[buffer(10), function_constant(qattn_output_logsumexp)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  ulong3 tidl{tid.x, tid.y, tid.z};
  ulong kv_head_idx = int(tid.y) / params->gqa_factor;

  Q += tidl.z * params->Q_strides[0] + tidl.y * params->Q_strides[1] +
      tidl.x * BQ * params->Q_strides[2];
  KQ += tidl.z * qstrides->KQ_strides[0] + kv_head_idx * qstrides->KQ_strides[1];
  KS += tidl.z * qstrides->KS_strides[0] + kv_head_idx * qstrides->KS_strides[1];
  KB += tidl.z * qstrides->KB_strides[0] + kv_head_idx * qstrides->KB_strides[1];
  VQ += tidl.z * qstrides->VQ_strides[0] + kv_head_idx * qstrides->VQ_strides[1];
  VS += tidl.z * qstrides->VS_strides[0] + kv_head_idx * qstrides->VS_strides[1];
  VB += tidl.z * qstrides->VB_strides[0] + kv_head_idx * qstrides->VB_strides[1];
  O += tidl.z * params->O_strides[0] + tidl.y * params->O_strides[1] +
      tidl.x * BQ * params->O_strides[2];

  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;
  constexpr short tgp_mem_0 = (BK + padK) * BD;
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];
  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
  using KBlockLoader = QAttnQuantizedBlockLoader<
      T,
      BK,
      BD,
      1,
      LDK_tgp,
      WM * WN * 32,
      64,
      4>;
  using VBlockLoader = QAttnQuantizedBlockLoader<
      T,
      BK,
      BD,
      LDV_tgp,
      1,
      WM * WN * 32,
      64,
      4>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      (const device uint8_t*)KQ, KS, KB, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      (const device uint8_t*)VQ, VS, VB, Vs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  constexpr int kNWarps = WM * WN;
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;
  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;
  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!qattn_align_Q && int(tid.x) == params->NQ_aligned) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;
  if (qattn_do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = min(params->NK, (q_max + BK - 1) / BK);
    int q_min = max(0, int(tid.x) * BQ + params->qL_off);
    kb_min_causal = q_min / BK;
  }

  for (int kb = 0; kb < kb_lim; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!qattn_align_K && kb == params->NK_aligned) {
      loader_k.load_safe(short(params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    if (!qattn_align_K && kb == params->NK_aligned) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    if (qattn_do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!qattn_align_K && kb == params->NK_aligned) {
      loader_v.load_safe(short(params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<QAttnMaxOp>(new_max);
    Stile.template row_bin_op<QAttnExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<QAttnSumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }
    Otile.template row_bin_op<QAttnMulOp>(factor);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;
          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);
          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    loader_k.next();
    loader_v.next();
  }

  Otile.template row_bin_op<QAttnDivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);
  O += (tm + sm) * params->O_strides[2] + sn;
  if (!qattn_align_Q && int(tid.x) == params->NQ_aligned) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
      return;
    }
    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }

  if (qattn_output_logsumexp) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      int row = int(tid.x) * BQ + tm + sm + (i * kFragSize);
      if (row < params->qL) {
        int64_t idx = int64_t(tid.z) * params->H * params->qL +
            int64_t(tid.y) * params->qL + row;
        lse_out[idx] =
            float(max_score[i]) * M_LN2_F + metal::precise::log(float(sum_score[i]));
      }
    }
  }
}
