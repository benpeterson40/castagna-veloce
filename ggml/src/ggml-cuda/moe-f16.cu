#include "moe-f16.cuh"
#include "convert.cuh"
#include "mmid.cuh"

// Tokens are grouped by expert (mm_ids_helper), a tile map lists the non-empty (expert, token tile) pairs,
// and each block computes 128 weight rows x BN tokens of one expert. 8 waves, each owning 16 weight rows;
// a wave's weight fragment is reused across all BN/16 token tiles.
//
// WMMA operands: A = activations (M = token), B = weights (N = weight row). On RDNA3 the f32 result
// element i of lane l is (M = 2*i + l/16, N = l%16), so 16 lanes store 16 consecutive floats of a dst row.

static constexpr int MOE_F16_BM      = 128; // weight rows per block
static constexpr int MOE_F16_BK      = 64;  // K elements per shared-memory stage
static constexpr int MOE_F16_THREADS = 256;

typedef _Float16 moe_h2   __attribute__((ext_vector_type(2)));
typedef _Float16 moe_v16h __attribute__((ext_vector_type(16)));
typedef float    moe_v8f  __attribute__((ext_vector_type(8)));

// A shared-memory row holds 64 halves = 8 chunks of 16 bytes. Chunk c of row r is stored at c ^ (r & 7) so
// the 16 lanes reading the same chunk of 16 consecutive rows hit different banks.
static __device__ __forceinline__ int moe_swz(const int r, const int c) {
    return r*8 + (c ^ (r & 7));
}

#if defined(RDNA3)
// 4 codes (one per byte, each < 1024) -> 4 halves (code*scale + bias). The byte permute places each code in
// the mantissa of 1024.0h, subtracting 1024 recovers it exactly, then one packed FMA applies scale and bias.
// OFFSET = 1024 + the code bias: 1024 for unsigned codes, 1152 for q8_0 codes stored as q + 128 (exact either way).
template <int OFFSET = 1024>
static __device__ __forceinline__ void moe_codes_to_half(const uint32_t codes, const moe_h2 scale, const moe_h2 bias,
                                                         uint32_t & lo, uint32_t & hi) {
    constexpr uint32_t magic = 0x64646464u;
    const moe_h2 k1024 = {(_Float16) (float) OFFSET, (_Float16) (float) OFFSET};
    const uint32_t p0 = __builtin_amdgcn_perm(codes, magic, 0x01050004u);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, magic, 0x03070206u);
    const moe_h2 h0 = (__builtin_bit_cast(moe_h2, p0) - k1024)*scale + bias;
    const moe_h2 h1 = (__builtin_bit_cast(moe_h2, p1) - k1024)*scale + bias;
    lo = __builtin_bit_cast(uint32_t, h0);
    hi = __builtin_bit_cast(uint32_t, h1);
}

// 16 codes (4 words) -> 2 chunks of 8 halves
template <int OFFSET = 1024>
static __device__ __forceinline__ void moe_dequant16(const uint32_t w[4], const moe_h2 scale, const moe_h2 bias,
                                                     uint4 & c0, uint4 & c1) {
    moe_codes_to_half<OFFSET>(w[0], scale, bias, c0.x, c0.y);
    moe_codes_to_half<OFFSET>(w[1], scale, bias, c0.z, c0.w);
    moe_codes_to_half<OFFSET>(w[2], scale, bias, c1.x, c1.y);
    moe_codes_to_half<OFFSET>(w[3], scale, bias, c1.z, c1.w);
}
#endif // defined(RDNA3)

// Per-thread weight registers for one stage. Thread t handles row t/2 of the block.
template <ggml_type type> struct moe_f16_wregs;

// q4_K: a 64-K stage is one 32-byte group of a superblock (low nibbles = sub-block 2g, high = 2g+1).
// Thread half q takes bytes 16q..16q+15 -> K positions [16q, 16q+16) and [32+16q, 32+16q+16).
template <> struct moe_f16_wregs<GGML_TYPE_Q4_K> {
    uint4    qs;
    uint32_t dm;
    uint32_t sc0, sc1, sc2; // the 12 scale bytes (named, not an array: indexed arrays spill to scratch)
    int      g;
};

// q5_1: a 64-K stage is two blocks; thread half b takes block b -> K positions [32b, 32b+32).
template <> struct moe_f16_wregs<GGML_TYPE_Q5_1> {
    uint32_t dm;
    uint32_t qh;
    uint32_t qs[4];
};

// q8_0: a 64-K stage is two 34-byte blocks (68 bytes, 4-byte aligned since rows are 16-byte aligned and stages
// start at even blocks). Half 0 loads dwords 0-8 (block 0 at byte 0), half 1 dwords 8-16 (block 1 at byte 34).
template <> struct moe_f16_wregs<GGML_TYPE_Q8_0> {
    uint32_t w[9];
};

template <ggml_type type>
static __device__ __forceinline__ void moe_f16_fetch_w(moe_f16_wregs<type> & r, const char * __restrict__ row_ptr,
                                                       const int kb0, const int half) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        const block_q4_K * b = (const block_q4_K *) row_ptr + kb0/QK_K;
        const int g = (kb0 % QK_K) / 64;
        r.g  = g;
        if (g == 0) { // the superblock header serves its 4 stages; kept in the registers in between
            const uint4 h = *(const uint4 *) b; // dm (4 bytes) + scales (12 bytes)
            r.dm  = h.x;
            r.sc0 = h.y;
            r.sc1 = h.z;
            r.sc2 = h.w;
        }
        r.qs = *(const uint4 *) (b->qs + 32*g + 16*half);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        const uint32_t * p = (const uint32_t *) (row_ptr + (kb0/QK8_0)*sizeof(block_q8_0)) + 8*half;
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            r.w[i] = p[i];
        }
    } else {
        const block_q5_1 * b = (const block_q5_1 *) row_ptr + kb0/QK5_1 + half;
        const uint2 * p = (const uint2 *) b; // 24-byte block, 8-byte aligned
        const uint2 a0 = p[0], a1 = p[1], a2 = p[2];
        r.dm = a0.x; r.qh = a0.y;
        r.qs[0] = a1.x; r.qs[1] = a1.y; r.qs[2] = a2.x; r.qs[3] = a2.y;
    }
}

#if defined(RDNA3)
template <ggml_type type>
static __device__ __forceinline__ void moe_f16_commit_w(const moe_f16_wregs<type> & r, uint4 * __restrict__ s_w,
                                                        const int row, const int half) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        // byte k of the 12 scale bytes, selected without a dynamically indexed array (which spills to scratch)
        const uint32_t sc0 = r.sc0, sc1 = r.sc1, sc2 = r.sc2;
        const auto sc_byte = [sc0, sc1, sc2](const int k) -> int {
            const uint32_t w = k < 4 ? sc0 : (k < 8 ? sc1 : sc2);
            return (w >> (8*(k & 3))) & 0xFF;
        };
        const moe_h2 dm = __builtin_bit_cast(moe_h2, r.dm);
        const float d = (float) dm.x, dmin = (float) dm.y;
        float scale[2], bias[2];
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int j = 2*r.g + s;
            int scj, mj;
            if (j < 4) {
                scj = sc_byte(j) & 63;
                mj  = sc_byte(j + 4) & 63;
            } else {
                scj = (sc_byte(j + 4) & 0xF) | ((sc_byte(j - 4) >> 6) << 4);
                mj  = (sc_byte(j + 4) >>  4) | ((sc_byte(j    ) >> 6) << 4);
            }
            scale[s] = d*scj;
            bias[s]  = -dmin*mj;
        }
        const uint32_t q[4] = {r.qs.x, r.qs.y, r.qs.z, r.qs.w};
        uint32_t lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lo[i] =  q[i]       & 0x0F0F0F0Fu;
            hi[i] = (q[i] >> 4) & 0x0F0F0F0Fu;
        }
        uint4 c0, c1;
        const moe_h2 s0 = {(_Float16) scale[0], (_Float16) scale[0]}, b0 = {(_Float16) bias[0], (_Float16) bias[0]};
        const moe_h2 s1 = {(_Float16) scale[1], (_Float16) scale[1]}, b1 = {(_Float16) bias[1], (_Float16) bias[1]};
        moe_dequant16(lo, s0, b0, c0, c1);
        s_w[moe_swz(row, 2*half + 0)] = c0;
        s_w[moe_swz(row, 2*half + 1)] = c1;
        moe_dequant16(hi, s1, b1, c0, c1);
        s_w[moe_swz(row, 4 + 2*half + 0)] = c0;
        s_w[moe_swz(row, 4 + 2*half + 1)] = c1;
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        // half 0: d = bytes 0-1, codes = bytes 2-33 (shifted by 2 bytes); half 1: d = bytes 2-3, codes = words 1-8
        uint32_t q[8];
        uint32_t dbits;
        if (half == 0) {
            dbits = r.w[0] & 0xFFFF;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                q[i] = __builtin_amdgcn_alignbyte(r.w[i + 1], r.w[i], 2);
            }
        } else {
            dbits = r.w[0] >> 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                q[i] = r.w[i + 1];
            }
        }
        const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) dbits);
        // signed codes: q ^ 0x80 = q + 128 as an unsigned byte; half(1024 + q + 128) - 1152 = q exactly
        const moe_h2 s = {d, d};
        const moe_h2 b = {(_Float16) 0.0f, (_Float16) 0.0f};
        uint32_t lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lo[i] = q[i]     ^ 0x80808080u;
            hi[i] = q[i + 4] ^ 0x80808080u;
        }
        uint4 c0, c1;
        moe_dequant16<1152>(lo, s, b, c0, c1);
        s_w[moe_swz(row, 4*half + 0)] = c0;
        s_w[moe_swz(row, 4*half + 1)] = c1;
        moe_dequant16<1152>(hi, s, b, c0, c1);
        s_w[moe_swz(row, 4*half + 2)] = c0;
        s_w[moe_swz(row, 4*half + 3)] = c1;
    } else {
        const moe_h2 dm = __builtin_bit_cast(moe_h2, r.dm);
        const moe_h2 s  = {dm.x, dm.x}, b = {dm.y, dm.y};
        uint32_t lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // bit 4 of value k comes from qh bit k (low nibbles, values 0..15) or bit k+16 (high nibbles)
            const uint32_t hl = (__umul24((r.qh >> (4*i))      & 0xF, 0x00204081u) & 0x01010101u) << 4;
            const uint32_t hh = (__umul24((r.qh >> (16 + 4*i)) & 0xF, 0x00204081u) & 0x01010101u) << 4;
            lo[i] = ( r.qs[i]       & 0x0F0F0F0Fu) | hl;
            hi[i] = ((r.qs[i] >> 4) & 0x0F0F0F0Fu) | hh;
        }
        uint4 c0, c1;
        moe_dequant16(lo, s, b, c0, c1);
        s_w[moe_swz(row, 4*half + 0)] = c0;
        s_w[moe_swz(row, 4*half + 1)] = c1;
        moe_dequant16(hi, s, b, c0, c1);
        s_w[moe_swz(row, 4*half + 2)] = c0;
        s_w[moe_swz(row, 4*half + 3)] = c1;
    }
}
#endif // defined(RDNA3)

// PAIR: gate/up SwiGLU. A block covers 64 rows of the gate matrix (x, waves 0-3) and the same 64 rows of the up
// matrix (x_up, waves 4-7); the up results meet the gate results in shared memory and silu(gate)*up is stored.
// DENSE: plain MUL_MAT (one weight matrix, token t is src1/dst row t, n_dense tokens); ids/tile map unused.
// OUT16 (PAIR only): store silu(gate)*up as F16 in place (the consumer reads F16, see ggml_cuda_moe_f16_pair)
// NW: waves per block (16 weight rows each); 8 by default, 4 for narrow dense matrices
template <ggml_type type, int BN, bool PAIR, bool DENSE = false, bool OUT16 = false, int NW = 8>
__launch_bounds__(NW*32, 1)
static __global__ void moe_f16_wmma(
        const char * __restrict__ x, const char * __restrict__ x_up, const half * __restrict__ y, const int32_t * __restrict__ ids_src1,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const int2 * __restrict__ tile_map, float * __restrict__ dst,
        const int K, const size_t nb01, const size_t nb02, const int64_t ld_dst, const int n_dense, const int m_dense,
        const int64_t ldy) {
#if defined(RDNA3)
    constexpr int BMK = NW*16;   // weight rows per block
    constexpr int THK = NW*32;   // threads per block
    static_assert(!PAIR || NW == 8, "pair mode assumes 8 waves");
    // DENSE split-K: block z covers K range [z*K/gridDim.z, (z+1)*K/gridDim.z) and writes its partial sums to
    // dst + z*n_dense*ld_dst (reduced afterwards). m_dense: weight rows (the last row block may be partial).
    const int kz_len = K / gridDim.z;
    const int k_beg  = blockIdx.z*kz_len;
    const int k_end  = k_beg + kz_len;
    constexpr int NT = BN/16;               // token tiles
    constexpr int YL = BN*8/THK; // activation chunks loaded per thread per stage
    static_assert(YL >= 1 && BN*8 % THK == 0, "BN must be a multiple of 32");

    int expert, c0, nvalid;
    if constexpr (DENSE) {
        expert = 0;
        c0     = blockIdx.y*BN;
        nvalid = min(BN, n_dense - c0);
    } else {
        const int2 tile = tile_map[blockIdx.y];
        if (tile.x < 0) {
            return;
        }
        expert = tile.x;
        c0     = expert_bounds[expert] + tile.y*BN;
        nvalid = min(BN, expert_bounds[expert + 1] - c0);
    }

    __shared__ uint4 smem[(BMK + BN)*8];
    uint4 * s_w = smem;
    uint4 * s_y = smem + BMK*8;

    const int tid  = threadIdx.x;
    const int wave = tid / 32;
    const int lane = tid % 32;

    // weight rows this thread loads
    const int  w_row  = tid / 2;
    const int  w_half = tid % 2;
    constexpr int rows_per_mat = PAIR ? BMK/2 : BMK; // output rows per block
    const int row0 = blockIdx.x*rows_per_mat;
    const char * w_ptr = PAIR ?
        (w_row < rows_per_mat ? x    + expert*nb02 + (size_t) (row0 + w_row)*nb01
                              : x_up + expert*nb02 + (size_t) (row0 + w_row - rows_per_mat)*nb01) :
        x + expert*nb02 + (size_t) (DENSE ? min(row0 + w_row, m_dense - 1) : row0 + w_row)*nb01;

    // activation chunks this thread loads
    // rows past nvalid read the last valid slot's index (always in bounds) and load zeros
    const half * y_ptr[YL];
    bool y_ok[YL];
    int y_tok[YL], y_chunk[YL];
#pragma unroll
    for (int i = 0; i < YL; ++i) {
        const int idx = tid + i*THK;
        y_tok[i]   = idx / 8;
        y_chunk[i] = idx % 8;
        y_ok[i]    = y_tok[i] < nvalid;
        const int yrow = DENSE ? c0 + min(y_tok[i], nvalid - 1) : ids_src1[c0 + min(y_tok[i], nvalid - 1)];
        y_ptr[i]   = y + (size_t) yrow*ldy + 8*y_chunk[i];
    }

    moe_f16_wregs<type> wr;
    uint4 yr[YL];
    const auto fetch = [&](const int kb0) {
        moe_f16_fetch_w<type>(wr, w_ptr, kb0, w_half);
#pragma unroll
        for (int i = 0; i < YL; ++i) {
            yr[i] = y_ok[i] ? *(const uint4 *) (y_ptr[i] + kb0) : make_uint4(0, 0, 0, 0);
        }
    };

    moe_v8f acc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j] = moe_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }
    const int live_tiles = (nvalid + 15) / 16;
    const int frag_row   = lane % 16;

    fetch(k_beg);
    for (int kb0 = k_beg; kb0 < k_end; kb0 += MOE_F16_BK) {
        moe_f16_commit_w<type>(wr, s_w, w_row, w_half);
#pragma unroll
        for (int i = 0; i < YL; ++i) {
            s_y[moe_swz(y_tok[i], y_chunk[i])] = yr[i];
        }
        __syncthreads();
        if (kb0 + MOE_F16_BK < k_end) {
            fetch(kb0 + MOE_F16_BK);
        }
        __builtin_amdgcn_sched_barrier(0); // keep the next stage's global loads ahead of the WMMA loop
#pragma unroll
        for (int s = 0; s < MOE_F16_BK/16; ++s) {
            const int r = wave*16 + frag_row;
            uint4 b2[2] = {s_w[moe_swz(r, 2*s)], s_w[moe_swz(r, 2*s + 1)]};
            moe_v16h b;
            memcpy(&b, b2, sizeof(b));
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (j >= live_tiles) {
                    continue;
                }
                const int t = j*16 + frag_row;
                uint4 a2[2] = {s_y[moe_swz(t, 2*s)], s_y[moe_swz(t, 2*s + 1)]};
                moe_v16h a;
                memcpy(&a, a2, sizeof(a));
                acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[j]);
            }
        }
        __syncthreads();
    }

    if constexpr (PAIR) {
        // up waves park their results as float [token][64 rows] in the (now idle) staging memory
        static_assert(BN*64*sizeof(float) <= (BMK + BN)*8*sizeof(uint4), "pair epilogue does not fit");
        float * s_up = (float *) smem;
        const int r = (wave % 4)*16 + lane % 16;
        if (wave >= 4) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    s_up[(j*16 + 2*i + lane/16)*64 + r] = acc[j][i];
                }
            }
        }
        __syncthreads();
        if (wave < 4) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int t = j*16 + 2*i + lane/16;
                    if (t < nvalid) {
                        const float g = acc[j][i];
                        const float v = g/(1.0f + expf(-g)) * s_up[t*64 + r];
                        const int64_t o = (int64_t) ids_dst[c0 + t]*ld_dst + row0 + r;
                        if constexpr (OUT16) {
                            ((half *) dst)[o] = __float2half(v);
                        } else {
                            dst[o] = v;
                        }
                    }
                }
            }
        }
    } else {
        const int col = row0 + wave*16 + lane % 16;
        if (DENSE) {
            if (col >= m_dense) {
                return;
            }
            dst += (int64_t) blockIdx.z*n_dense*ld_dst;
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int t = j*16 + 2*i + lane/16;
                if (t < nvalid) {
                    const int64_t o = (int64_t) (DENSE ? c0 + t : ids_dst[c0 + t])*ld_dst + col;
                    if constexpr (OUT16) {
                        ((half *) dst)[o] = __float2half(acc[j][i]);
                    } else {
                        dst[o] = acc[j][i];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, x_up, y, ids_src1, ids_dst, expert_bounds, tile_map, dst, K, nb01, nb02, ld_dst, n_dense, m_dense, ldy);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

static constexpr int MOE_F16_TILE_MAP_THREADS = 256;

static __global__ void moe_f16_build_tile_map(
        const int32_t * __restrict__ expert_bounds, const int n_experts, const int J,
        int2 * __restrict__ tile_map, const int n_tiles_ub) {
    __shared__ int scan[MOE_F16_TILE_MAP_THREADS];

    const int tid   = threadIdx.x;
    const int chunk = (n_experts + MOE_F16_TILE_MAP_THREADS - 1) / MOE_F16_TILE_MAP_THREADS;
    const int e0    = min(tid*chunk, n_experts);
    const int e1    = min(e0 + chunk, n_experts);

    int local = 0;
    for (int e = e0; e < e1; ++e) {
        local += (expert_bounds[e + 1] - expert_bounds[e] + J - 1) / J;
    }
    scan[tid] = local;
    __syncthreads();
    for (int off = 1; off < MOE_F16_TILE_MAP_THREADS; off *= 2) {
        const int v = tid >= off ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    const int total = scan[MOE_F16_TILE_MAP_THREADS - 1];
    int pos = scan[tid] - local;
    for (int e = e0; e < e1; ++e) {
        const int nt = (expert_bounds[e + 1] - expert_bounds[e] + J - 1) / J;
        for (int t = 0; t < nt; ++t) {
            tile_map[pos++] = make_int2(e, t);
        }
    }
    for (int i = total + tid; i < n_tiles_ub; i += MOE_F16_TILE_MAP_THREADS) {
        tile_map[i] = make_int2(-1, -1);
    }
}

static int moe_f16_env(const char * name, const int def) {
    const char * s = getenv(name);
    return s ? atoi(s) : def;
}

bool ggml_cuda_moe_f16_supported(const int cc, const ggml_tensor * src0, const ggml_tensor * src1,
                                 const ggml_tensor * ids, const ggml_tensor * dst) {
    static const int enabled    = moe_f16_env("GGML_MOE_F16", 1);
    static const int min_tokens = moe_f16_env("GGML_MOE_F16_MIN_TOKENS", 64);
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(cc) || ids == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_1) {
        return false;
    }
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src1) && ggml_is_contiguous(dst) && src1->ne[3] == 1 &&
        src0->ne[1] % MOE_F16_BM == 0 && src0->ne[0] % MOE_F16_BK == 0 &&
        src0->nb[1] % 16 == 0 && src0->nb[2] % 16 == 0 &&
        src1->ne[2] >= min_tokens;
}

template <ggml_type type, int BN, bool PAIR, bool OUT16 = false>
static void moe_f16_launch(const char * x, const char * x_up, const half * y, const int32_t * ids_src1, const int32_t * ids_dst,
                           const int32_t * expert_bounds, const int2 * tile_map, float * dst, const int K,
                           const size_t nb01, const size_t nb02, const int64_t ld_dst, const int nrows,
                           const int n_tiles_ub, cudaStream_t stream) {
    const dim3 grid(nrows / (PAIR ? MOE_F16_BM/2 : MOE_F16_BM), n_tiles_ub, 1);
    moe_f16_wmma<type, BN, PAIR, false, OUT16><<<grid, MOE_F16_THREADS, 0, stream>>>(
        x, x_up, y, ids_src1, ids_dst, expert_bounds, tile_map, dst, K, nb01, nb02, ld_dst, 0, 0, K);
}

// src0_up != nullptr: paired gate (src0) / up (src0_up) with SwiGLU into dst; otherwise dst = src0 x src1
static void moe_f16_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src0_up,
                         const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const bool out16 = false) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    cudaStream_t stream = ctx.stream();

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_slots       = ne12*n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    // expert-sorted slot order (forward map: slot -> src1 row, slot -> dst row)
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    {
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ false, stream, &ctx.pool());
        CUDA_CHECK(cudaGetLastError());
    }

    // activations once to F16 (unless the producer already stored them as F16 in place); rows are gathered inside
    // the GEMM kernel
    ggml_cuda_pool_alloc<half> y16_pool(ctx.pool());
    const half * y16 = nullptr;
    if (ggml_cuda_is_f16_inplace(ctx, src1)) {
        GGML_ASSERT(ggml_is_contiguous(src1) && ggml_cuda_f16_inplace_ld(ctx, src1, src1->ne[0]) == src1->ne[0]);
        y16 = (const half *) src1->data;
    } else {
        y16 = y16_pool.alloc(ggml_nelements(src1));
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, (half *) y16, ggml_nelements(src1), stream);
    }

    static const int bn_env = moe_f16_env("GGML_MOE_F16_BN", 0);
    // 128-token tiles won at both 1024 and 2048 tokens (about 20 and 40 per expert): each expert's weights are
    // streamed once, and WMMA work on dead token sub-tiles is skipped
    const int BN = bn_env == 32 || bn_env == 64 ? bn_env : 128;

    const int n_tiles_ub = (n_slots + BN - 1) / BN + ne02;
    ggml_cuda_pool_alloc<int2> tile_map(ctx.pool(), n_tiles_ub);
    moe_f16_build_tile_map<<<1, MOE_F16_TILE_MAP_THREADS, 0, stream>>>(expert_bounds.get(), ne02, BN, tile_map.get(), n_tiles_ub);

    const int64_t ld_dst = dst->nb[1] / sizeof(float);
    const char * x    = (const char *) src0->data;
    const char * x_up = src0_up ? (const char *) src0_up->data : nullptr;
    float * d = (float *) dst->data;

#define MOE_F16_CASE(T, N, P) (out16 ? moe_f16_launch<T, N, P, true> : moe_f16_launch<T, N, P, false>)(x, x_up, y16, ids_src1.get(), ids_dst.get(), \
        expert_bounds.get(), tile_map.get(), d, ne00, nb01, nb02, ld_dst, ne01, n_tiles_ub, stream)
#define MOE_F16_BN_SWITCH(T, P) switch (BN) { \
            case 32: MOE_F16_CASE(T, 32, P); break; \
            case 64: MOE_F16_CASE(T, 64, P); break; \
            default: MOE_F16_CASE(T, 128, P); break; }
    if (src0->type == GGML_TYPE_Q4_K) {
        if (x_up) { MOE_F16_BN_SWITCH(GGML_TYPE_Q4_K, true) } else { MOE_F16_BN_SWITCH(GGML_TYPE_Q4_K, false) }
    } else {
        if (x_up) { MOE_F16_BN_SWITCH(GGML_TYPE_Q5_1, true) } else { MOE_F16_BN_SWITCH(GGML_TYPE_Q5_1, false) }
    }
#undef MOE_F16_BN_SWITCH
#undef MOE_F16_CASE
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       const ggml_tensor * ids, ggml_tensor * dst, const bool out16) {
    if (out16) {
        GGML_ASSERT(ggml_is_contiguous(dst));
        ctx.f16_inplace.push_back({dst, 0});
    }
    moe_f16_impl(ctx, src0, nullptr, src1, ids, dst, out16);
}

bool ggml_cuda_moe_f16_pair_supported(const int cc, const ggml_tensor * gate, const ggml_tensor * up,
                                      const ggml_tensor * glu) {
    static const int pair_enabled = moe_f16_env("GGML_MOE_F16_PAIR", 1);
    return pair_enabled && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
        gate->op == GGML_OP_MUL_MAT_ID && up->op == GGML_OP_MUL_MAT_ID &&
        ggml_cuda_moe_f16_supported(cc, gate->src[0], gate->src[1], gate->src[2], glu) &&
        gate->src[0]->ne[1] % (MOE_F16_BM/2) == 0 &&
        ggml_are_same_shape(gate, glu) && ggml_is_contiguous(glu);
}

void ggml_cuda_moe_f16_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
                            ggml_tensor * glu, const bool out16) {
    if (out16) {
        GGML_ASSERT(ggml_is_contiguous(glu));
        ctx.f16_inplace.push_back({glu, 0});
    }
    moe_f16_impl(ctx, gate->src[0], up->src[0], gate->src[1], gate->src[2], glu, out16);
}

// sum of the split-K partials in a fixed order (deterministic)
static __global__ void dense_f16_reduce(const float * __restrict__ part, float * __restrict__ dst, const int64_t n,
                                        const int nsplit) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (4*i >= n) {
        return;
    }
    float4 s = ((const float4 *) part)[i];
    for (int z = 1; z < nsplit; ++z) {
        const float4 v = ((const float4 *) (part + z*n))[i];
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    ((float4 *) dst)[i] = s;
}

bool ggml_cuda_dense_f16_supported(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int enabled    = moe_f16_env("GGML_DENSE_F16", 1);
    static const int min_tokens = moe_f16_env("GGML_DENSE_F16_MIN_TOKENS", 256);
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(cc) || src0->type != GGML_TYPE_Q8_0 ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src0->ne[1] % 16 != 0 || src0->ne[0] % MOE_F16_BK != 0 || src0->nb[1] % 4 != 0 || src1->ne[1] < min_tokens) { // q8_0 loads are dwords
        return false;
    }
    // wide matrices run as is; narrow ones only with a long K to split (the 320x10240 HC down projection).
    // 640x2560 stays on MMQ (21 vs 24 TFLOPS).
    return src0->ne[1] >= 2048 || src0->ne[0] >= 8192;
}

static void dense_f16_run(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const half * y16, const int N,
                          ggml_tensor * dst, int64_t ldy = 0);

void ggml_cuda_dense_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                         ggml_tensor * dst) {
    ggml_cuda_pool_alloc<half> y16(ctx.pool(), ggml_nelements(src1));
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, y16.get(), ggml_nelements(src1), ctx.stream());
    dense_f16_run(ctx, src0, y16.get(), src1->ne[1], dst);
}

static void dense_f16_run(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const half * y16, const int N,
                          ggml_tensor * dst, int64_t ldy) {
    cudaStream_t stream = ctx.stream();
    const int K = src0->ne[0], M = src0->ne[1];
    if (ldy == 0) {
        ldy = K;
    }

    // narrow matrices (e.g. 320 rows) give too few 128-token blocks for the GPU: use 64-token tiles there
    static const int bn_env = moe_f16_env("GGML_DENSE_F16_BN", 0);
    static const int nw_env = moe_f16_env("GGML_DENSE_F16_NW", 0);
    const int NW = nw_env == 4 ? 4 : 8; // 4 waves measured slower on 320 x 10240 (1.74 vs 1.30 ms)
    const int bx = (M + NW*16 - 1) / (NW*16);
    const int BN = bn_env == 64 || bn_env == 128 ? bn_env :
        (bx*((N + 127)/128) < 128 ? 64 : 128); // 320 x 2048: 48 blocks -> 96 (1.40 -> 1.29 ms)
    const int by = (N + BN - 1) / BN;
    // optional split-K (GGML_DENSE_F16_SPLIT): measured slower on the 320x10240 HC projection (1, 2, 4, 8 splits:
    // 1.43, 1.68, 1.64, 1.76 ms at 2048 tokens), since the row blocks of a token tile already share activations in L2
    const int n_stages = K / MOE_F16_BK;
    static const int split_env = moe_f16_env("GGML_DENSE_F16_SPLIT", 0);
    const int nsplit = split_env > 0 ? split_env : 1;
    GGML_ASSERT(n_stages % nsplit == 0);
    const int64_t ld = dst->nb[1] / sizeof(float);
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    float * out = (float *) dst->data;
    if (nsplit > 1) {
        GGML_ASSERT(ld == M && M % 4 == 0);
        part.alloc((size_t) nsplit*N*M);
        out = part.get();
    }
    const dim3 grid(bx, by, nsplit);
#define DENSE_F16_LAUNCH(BN_, NW_) moe_f16_wmma<GGML_TYPE_Q8_0, BN_, false, true, false, NW_><<<grid, NW_*32, 0, stream>>>( \
            (const char *) src0->data, nullptr, y16, nullptr, nullptr, nullptr, nullptr, out, K, src0->nb[1], 0, ld, N, M, ldy)
    if (NW == 4) {
        if (BN == 64) { DENSE_F16_LAUNCH(64, 4); } else { DENSE_F16_LAUNCH(128, 4); }
    } else {
        if (BN == 64) { DENSE_F16_LAUNCH(64, 8); } else { DENSE_F16_LAUNCH(128, 8); }
    }
#undef DENSE_F16_LAUNCH
    if (nsplit > 1) {
        const int64_t n = (int64_t) N*M;
        dense_f16_reduce<<<(n/4 + 255)/256, 256, 0, stream>>>(part.get(), (float *) dst->data, n, nsplit);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connection up projection fused with DSV4_HC_PRE (gated): gate = W_up (q8_0, [hc*n_embd, K]) x lo, then
// mixed[e, t] = scale * sum_s xn[e, s, t] * sigmoid(gate[s*n_embd + e, t]). The gate tensor is never stored.
// A block covers 32 embedding indices x 4 streams (128 weight rows) x 128 tokens. Wave w owns 16 embedding indices
// (w % 2) of all 4 streams for 32 tokens (w / 2), so the 4 stream values of one (e, t) end up in the same lane.
static constexpr int HC_MIX_EMB = 32; // embedding indices per block
static constexpr int HC_MIX_BN  = 128;

template <bool XN16> // xn stored as F16 in place: [t][s*n_embd + e], contiguous
__launch_bounds__(MOE_F16_THREADS, 1)
static __global__ void hc_up_mix_wmma(
        const char * __restrict__ w, const half * __restrict__ y, const float * __restrict__ xn, float * __restrict__ dst,
        const int K, const size_t nb01, const int n_embd, const int n_tok,
        const int64_t sx1, const int64_t sx2, const int64_t sd1, const float scale, const int64_t xn_ld16) {
#if defined(RDNA3)
    constexpr int HC = 4;
    constexpr int YL = HC_MIX_BN*8/MOE_F16_THREADS;
    __shared__ uint4 smem[(MOE_F16_BM + HC_MIX_BN)*8];
    uint4 * s_w = smem;
    uint4 * s_y = smem + MOE_F16_BM*8;

    const int tid  = threadIdx.x;
    const int wave = tid / 32;
    const int lane = tid % 32;
    const int e0   = blockIdx.x*HC_MIX_EMB;
    const int c0   = blockIdx.y*HC_MIX_BN;
    const int nvalid = min(HC_MIX_BN, n_tok - c0);

    // block row r = s*32 + el <-> weight row s*n_embd + e0 + el
    const int w_row  = tid / 2;
    const int w_half = tid % 2;
    const char * w_ptr = w + (size_t) ((w_row / HC_MIX_EMB)*n_embd + e0 + w_row % HC_MIX_EMB)*nb01;

    const half * y_ptr[YL];
    bool y_ok[YL];
    int y_tok[YL], y_chunk[YL];
#pragma unroll
    for (int i = 0; i < YL; ++i) {
        const int idx = tid + i*MOE_F16_THREADS;
        y_tok[i]   = idx / 8;
        y_chunk[i] = idx % 8;
        y_ok[i]    = y_tok[i] < nvalid;
        y_ptr[i]   = y + (size_t) (c0 + min(y_tok[i], nvalid - 1))*K + 8*y_chunk[i];
    }

    moe_f16_wregs<GGML_TYPE_Q8_0> wr;
    uint4 yr[YL];
    const auto fetch = [&](const int kb0) {
        moe_f16_fetch_w<GGML_TYPE_Q8_0>(wr, w_ptr, kb0, w_half);
#pragma unroll
        for (int i = 0; i < YL; ++i) {
            yr[i] = y_ok[i] ? *(const uint4 *) (y_ptr[i] + kb0) : make_uint4(0, 0, 0, 0);
        }
    };

    const int eg = wave % 2; // embedding half
    const int tg = wave / 2; // token quarter (32 tokens = 2 tiles)
    moe_v8f acc[HC][2];
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        acc[s][0] = moe_v8f{0, 0, 0, 0, 0, 0, 0, 0};
        acc[s][1] = moe_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }
    const int frag_row = lane % 16;

    fetch(0);
    for (int kb0 = 0; kb0 < K; kb0 += MOE_F16_BK) {
        moe_f16_commit_w<GGML_TYPE_Q8_0>(wr, s_w, w_row, w_half);
#pragma unroll
        for (int i = 0; i < YL; ++i) {
            s_y[moe_swz(y_tok[i], y_chunk[i])] = yr[i];
        }
        __syncthreads();
        if (kb0 + MOE_F16_BK < K) {
            fetch(kb0 + MOE_F16_BK);
        }
#pragma unroll
        for (int ks = 0; ks < MOE_F16_BK/16; ++ks) {
            moe_v16h a[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int t = tg*32 + j*16 + frag_row;
                uint4 a2[2] = {s_y[moe_swz(t, 2*ks)], s_y[moe_swz(t, 2*ks + 1)]};
                memcpy(&a[j], a2, sizeof(a[j]));
            }
#pragma unroll
            for (int s = 0; s < HC; ++s) {
                const int r = s*HC_MIX_EMB + eg*16 + frag_row;
                uint4 b2[2] = {s_w[moe_swz(r, 2*ks)], s_w[moe_swz(r, 2*ks + 1)]};
                moe_v16h b;
                memcpy(&b, b2, sizeof(b));
                // unconditional: a conditional WMMA here doubled the live accumulators (256 VGPRs + spills)
                acc[s][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[0], b, acc[s][0]);
                acc[s][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[1], b, acc[s][1]);
            }
            __builtin_amdgcn_sched_barrier(0); // one k-step of fragments live at a time (hoisting all of them spilled)
        }
        __syncthreads();
    }

    const int e = e0 + eg*16 + lane % 16;
    // streams outermost with a scheduling barrier between them, so only 16 xn loads are live at a time
    // (all 64 at once pushed the kernel to 256 VGPRs with spills)
    float sum[2][8];
    int64_t t_off[2][8];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int tl = tg*32 + j*16 + 2*i + lane/16;
            sum[j][i]   = 0.0f;
            t_off[j][i] = tl < nvalid ? (int64_t) (c0 + tl) : -1;
        }
    }
#pragma unroll
    for (int s = 0; s < HC; ++s) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (t_off[j][i] >= 0) {
                    const float xv = XN16 ? __half2float(((const half *) xn)[e + s*n_embd + t_off[j][i]*xn_ld16]) :
                                            xn[e + s*sx1 + t_off[j][i]*sx2];
                    sum[j][i] += xv * (1.0f / (1.0f + expf(-acc[s][j][i])));
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (t_off[j][i] >= 0) {
                dst[e + t_off[j][i]*sd1] = scale * sum[j][i];
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, y, xn, dst, K, nb01, n_embd, n_tok, sx1, sx2, sd1, scale, xn_ld16);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

bool ggml_cuda_hc_up_mix_supported(const int cc, const ggml_tensor * mm, const ggml_tensor * pre) {
    static const int enabled = moe_f16_env("GGML_HC_UP_MIX", 1);
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(cc)) {
        return false;
    }
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * lo = mm->src[1];
    const ggml_tensor * xn = pre->src[0];
    return w->type == GGML_TYPE_Q8_0 && lo->type == GGML_TYPE_F32 && xn->type == GGML_TYPE_F32 &&
        pre->type == GGML_TYPE_F32 && ggml_get_op_params_i32(pre, 1) != 0 && // gated
        xn->ne[1] == 4 && w->ne[1] == 4*xn->ne[0] && xn->ne[0] % HC_MIX_EMB == 0 &&
        w->ne[0] % MOE_F16_BK == 0 && w->nb[1] % 4 == 0 && ggml_is_contiguous(w) && ggml_is_contiguous(lo) && // q8_0 loads are dwords
        lo->ne[2] == 1 && lo->ne[1] == xn->ne[2] && xn->nb[0] == sizeof(float) && pre->nb[0] == sizeof(float) &&
        lo->ne[1] >= 256;
}

void ggml_cuda_hc_up_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * pre) {
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * lo = mm->src[1];
    const ggml_tensor * xn = pre->src[0];
    const int K = w->ne[0], n_embd = xn->ne[0], N = lo->ne[1];

    ggml_cuda_pool_alloc<half> y16(ctx.pool(), ggml_nelements(lo));
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(lo->data, y16.get(), ggml_nelements(lo), stream);

    const dim3 grid(n_embd / HC_MIX_EMB, (N + HC_MIX_BN - 1) / HC_MIX_BN, 1);
    const int64_t xn_ld16 = ggml_cuda_f16_inplace_ld(ctx, xn, 4*(int64_t) n_embd);
    const bool xn16 = xn_ld16 != 0;
    (xn16 ? hc_up_mix_wmma<true> : hc_up_mix_wmma<false>)<<<grid, MOE_F16_THREADS, 0, stream>>>(
        (const char *) w->data, y16.get(), (const float *) xn->data, (float *) pre->data, K, w->nb[1], n_embd, N,
        xn->nb[1] / sizeof(float), xn->nb[2] / sizeof(float), pre->nb[1] / sizeof(float),
        ggml_get_op_params_f32(pre, 0), xn_ld16);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connection combine + grouped RMSNorm (+ HC down projection):
//   x[e,s,t]  = residual[e,s,t] + block_out[e,t]*post[s,t]          (DSV4_HC_POST, optional)
//   xn[e,s,t] = x[e,s,t] * rsqrt(mean_e x[.,s,t]^2 + eps) * w[e,s]    (RMS_NORM + MUL)
// One block per token keeps its 4 x n_embd row in registers; x and xn are stored (both are graph outputs) and an
// F16 copy of xn feeds the down projection directly, replacing its separate F32 -> F16 conversion pass.
static constexpr int HC_NORM_THREADS = 256;
static constexpr int HC_NORM_MAXV4   = 16; // float4 per thread: up to 256*16*4 = 16384 values per token

template <bool POST, bool STORE_F32, bool RAWP = false, bool STORE_F16 = true>
__launch_bounds__(HC_NORM_THREADS, 1)
static __global__ void hc_combine_norm(
        const float * __restrict__ residual, const float * __restrict__ bo, const float * __restrict__ post,
        const float * __restrict__ w, float * __restrict__ x_out, float * __restrict__ xn_out, half * __restrict__ xn16,
        const int n_embd, const int64_t s_res, const int64_t s_bo, const int64_t sp0, const int64_t sp1,
        const int64_t s_x, const int64_t s_xn, const float eps, const float ps1, const float ps2, const int64_t ld16) {
    constexpr int HC = 4;
    const int64_t t   = blockIdx.x;
    const int     tid = threadIdx.x;
    const int     nv4 = HC*n_embd/4;      // float4 per token
    const int     sv4 = n_embd/4;         // float4 per stream

    float4 v[HC_NORM_MAXV4];
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    float pw[HC];
    if constexpr (POST) {
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            pw[s] = post[s*sp0 + t*sp1];
            if constexpr (RAWP) {
                pw[s] = ps2 / (1.0f + expf(-ps1*pw[s])); // folded SCALE -> SIGMOID -> SCALE
            }
        }
    }
#pragma unroll
    for (int k = 0; k < HC_NORM_MAXV4; ++k) {
        const int j = tid + k*HC_NORM_THREADS;
        if (j < nv4) {
            float4 a = ((const float4 *) (residual + t*s_res))[j];
            const int st = j / sv4;
            if constexpr (POST) {
                const float4 b = ((const float4 *) (bo + t*s_bo))[j - st*sv4];
                const float  p = st == 0 ? pw[0] : st == 1 ? pw[1] : st == 2 ? pw[2] : pw[3];
                a.x += b.x*p; a.y += b.y*p; a.z += b.z*p; a.w += b.w*p;
                ((float4 *) (x_out + t*s_x))[j] = a;
            }
            v[k] = a;
            const float q = a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w;
            ss[0] += st == 0 ? q : 0.0f; ss[1] += st == 1 ? q : 0.0f;
            ss[2] += st == 2 ? q : 0.0f; ss[3] += st == 3 ? q : 0.0f;
        }
    }
    __shared__ float red[HC][HC_NORM_THREADS/WARP_SIZE];
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        ss[s] = warp_reduce_sum(ss[s]);
    }
    if (tid % WARP_SIZE == 0) {
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            red[s][tid / WARP_SIZE] = ss[s];
        }
    }
    __syncthreads();
    float rs[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        float tot = 0.0f;
        for (int i = 0; i < HC_NORM_THREADS/WARP_SIZE; ++i) {
            tot += red[s][i];
        }
        rs[s] = rsqrtf(tot/n_embd + eps);
    }
#pragma unroll
    for (int k = 0; k < HC_NORM_MAXV4; ++k) {
        const int j = tid + k*HC_NORM_THREADS;
        if (j < nv4) {
            const int st = j / sv4;
            const float r = st == 0 ? rs[0] : st == 1 ? rs[1] : st == 2 ? rs[2] : rs[3];
            const float4 g = ((const float4 *) w)[j]; // w is [n_embd, hc] contiguous: same flat index
            float4 o;
            o.x = v[k].x*r*g.x; o.y = v[k].y*r*g.y; o.z = v[k].z*r*g.z; o.w = v[k].w*r*g.w;
            if constexpr (STORE_F32) {
                ((float4 *) (xn_out + t*s_xn))[j] = o;
            }
            if constexpr (STORE_F16) {
                half2 * h = (half2 *) (xn16 + t*ld16) + 2*j;
                h[0] = __floats2half2_rn(o.x, o.y);
                h[1] = __floats2half2_rn(o.z, o.w);
            }
        }
    }
}

// The fused combine/norm kernels write x (the combine) and xn while other blocks still read the residual, the block
// output and the weights of other tokens. The allocator may place x or xn in memory freed by those inputs (e.g. xn
// over the dead block output), so any overlap is rejected, except x exactly replacing the residual in place
// (same layout, each element read before it is written by the same thread).
static bool hc_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static bool hc_combine_norm_aliasing_ok(const ggml_tensor * post_node, const ggml_tensor * mul, const ggml_tensor * w,
                                        const ggml_tensor * raw_post) {
    const ggml_tensor * bo  = post_node->src[0];
    const ggml_tensor * res = post_node->src[1];
    const ggml_tensor * pw  = raw_post ? raw_post : post_node->src[2];
    // one token = one block: bo, the residual and the weights are all read before the block-wide barrier and xn is
    // only written after it, so xn may overlap them (the norm weight w is read after the barrier)
    const bool single_block = post_node->ne[2] == 1;
    const ggml_tensor * ins[4] = { bo, res, pw, w };
    // xn exactly replacing the dead residual (same layout, F32) is safe too: block t reads and writes only token t's row
    // (not when xn is stored as F16 in place, see ggml_cuda_hc_xn_f16_inplace_safe)
    const bool xn_on_res = mul->data == res->data && ggml_are_same_shape(mul, res) && ggml_are_same_stride(mul, res);
    for (const ggml_tensor * in : ins) {
        if (hc_ranges_overlap(mul, in) && !(single_block && in != w) && !(in == res && xn_on_res)) {
            return false;
        }
        if (in != res && hc_ranges_overlap(post_node, in)) {
            return false;
        }
    }
    if (hc_ranges_overlap(post_node, res) && (post_node->data != res->data || !ggml_are_same_stride(post_node, res))) {
        return false;
    }
    return !hc_ranges_overlap(post_node, mul);
}


// nodes: [DSV4_HC_POST,] RMS_NORM, MUL, RESHAPE, RESHAPE, MUL_MAT (q8_0 down projection)
bool ggml_cuda_hc_norm_down_supported(const int cc, const ggml_tensor * post_node, const ggml_tensor * rms,
                                      const ggml_tensor * mul, const ggml_tensor * mm, const ggml_tensor * raw_post) {
    static const int enabled = moe_f16_env("GGML_HC_NORM_DOWN", 1);
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(cc)) {
        return false;
    }
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const bool shapes = x->type == GGML_TYPE_F32 && x->ne[1] == 4 && x->ne[3] == 1 && ggml_is_contiguous(x) &&
        x->ne[0] % 4 == 0 && 4*x->ne[0] <= HC_NORM_THREADS*HC_NORM_MAXV4*4 &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == x->ne[0] && w->ne[1] == 4 && w->ne[2] == 1 &&
        ggml_is_contiguous(mul) && ggml_are_same_shape(mul, x) &&
        mm->src[1]->data == mul->data && ggml_cuda_dense_f16_supported(cc, mm->src[0], mm->src[1], mm);
    if (!shapes) {
        return false;
    }
    if (post_node) {
        const ggml_tensor * bo = post_node->src[0];
        const ggml_tensor * rs = post_node->src[1];
        const ggml_tensor * pw = post_node->src[2];
        return post_node->src[3] == nullptr && x == post_node && bo->type == GGML_TYPE_F32 && rs->type == GGML_TYPE_F32 &&
            pw->type == GGML_TYPE_F32 && ggml_is_contiguous(rs) && ggml_are_same_shape(rs, x) &&
            bo->ne[0] == x->ne[0] && bo->nb[0] == sizeof(float) && bo->nb[1] % 16 == 0 && pw->ne[0] == 4 &&
            hc_combine_norm_aliasing_ok(post_node, mul, w, raw_post);
    }
    return !hc_ranges_overlap(mul, x) && !hc_ranges_overlap(mul, w);
}

void ggml_cuda_hc_norm_down(ggml_backend_cuda_context & ctx, ggml_tensor * post_node, const ggml_tensor * rms,
                            ggml_tensor * mul, ggml_tensor * mm, const bool xn_f16_inplace,
                            const ggml_tensor * raw_post, const float ps1, const float ps2) {
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int n_embd = x->ne[0];
    const int n_tok  = x->ne[2];
    const float eps  = ggml_get_op_params_f32(rms, 0);

    // xn_f16_inplace: every consumer of xn reads F16, so it is stored as F16 in its own buffer and not as F32
    ggml_cuda_pool_alloc<half> xn16_pool(ctx.pool());
    half * xn16 = nullptr;
    const int64_t ld16 = (xn_f16_inplace ? 8 : 4)*(int64_t) n_embd; // F16 row stride in halves
    if (xn_f16_inplace) {
        GGML_ASSERT(ggml_is_contiguous(mul));
        // F16 row t at the start of its own F32 row (stride = 2x the row length in halves): each block writes only
        // inside its token's footprint, which stays race-free when xn replaces the residual in place
        xn16 = (half *) mul->data;
        ctx.f16_inplace.push_back({mul, 2*4*(int64_t) n_embd});
    } else {
        xn16 = xn16_pool.alloc((size_t) 4*n_embd*n_tok);
    }
    const float * res = post_node ? (const float *) post_node->src[1]->data : (const float *) x->data;
    if (post_node) {
        const ggml_tensor * bo = post_node->src[0];
        const ggml_tensor * pw = raw_post ? raw_post : post_node->src[2];
        (raw_post ? (xn_f16_inplace ? hc_combine_norm<true, false, true> : hc_combine_norm<true, true, true>) :
                    (xn_f16_inplace ? hc_combine_norm<true, false> : hc_combine_norm<true, true>))<<<n_tok, HC_NORM_THREADS, 0, stream>>>(
            res, (const float *) bo->data,
            (const float *) pw->data, (const float *) w->data, (float *) post_node->data, (float *) mul->data, xn16,
            n_embd, post_node->src[1]->nb[2]/sizeof(float), bo->nb[1]/sizeof(float), pw->nb[0]/sizeof(float),
            pw->nb[1]/sizeof(float), post_node->nb[2]/sizeof(float), mul->nb[2]/sizeof(float), eps, ps1, ps2, ld16);
    } else {
        (xn_f16_inplace ? hc_combine_norm<false, false> : hc_combine_norm<false, true>)<<<n_tok, HC_NORM_THREADS, 0, stream>>>(
            res, nullptr, nullptr, (const float *) w->data,
            nullptr, (float *) mul->data, xn16, n_embd, x->nb[2]/sizeof(float), 0, 0, 0, 0,
            mul->nb[2]/sizeof(float), eps, 1.0f, 1.0f, ld16);
    }
    CUDA_CHECK(cudaGetLastError());
    dense_f16_run(ctx, mm->src[0], xn16, n_tok, mm, ld16);
}

// ---------------------------------------------------------------------------------------------------------------
// Gated output: y = a * sigmoid(z) -> RESHAPE -> dense q8_0 projection. The gated product is written once as F16
// straight into the projection's activation buffer instead of F32 followed by a separate F16 conversion.
static __global__ void gate_sigmoid_mul_f16(const float * __restrict__ a, const float * __restrict__ z,
                                            half * __restrict__ y, const int64_t n4) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n4) {
        return;
    }
    const float4 av = ((const float4 *) a)[i];
    const float4 zv = ((const float4 *) z)[i];
    half2 * o = (half2 *) y + 2*i;
    o[0] = __floats2half2_rn(av.x/(1.0f + expf(-zv.x)), av.y/(1.0f + expf(-zv.y)));
    o[1] = __floats2half2_rn(av.z/(1.0f + expf(-zv.z)), av.w/(1.0f + expf(-zv.w)));
}

bool ggml_cuda_gate_proj_supported(const int cc, const ggml_tensor * sig, const ggml_tensor * mul, const ggml_tensor * mm) {
    static const int enabled = moe_f16_env("GGML_GATE_PROJ", 1);
    if (!enabled) {
        return false;
    }
    const ggml_tensor * z = sig->src[0];
    const ggml_tensor * a = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    return ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && z->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(z) && ggml_is_contiguous(a) && ggml_are_same_shape(a, z) && ggml_are_same_shape(mul, z) &&
        ggml_nelements(z) % 4 == 0 && mm->src[1]->ne[0]*mm->src[1]->ne[1] == ggml_nelements(z) &&
        ggml_cuda_dense_f16_supported(cc, mm->src[0], mm->src[1], mm);
}

void ggml_cuda_gate_proj(ggml_backend_cuda_context & ctx, const ggml_tensor * sig, const ggml_tensor * mul, ggml_tensor * mm) {
    const ggml_tensor * z = sig->src[0];
    const ggml_tensor * a = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    const int64_t n = ggml_nelements(z);
    ggml_cuda_pool_alloc<half> y16(ctx.pool(), n);
    gate_sigmoid_mul_f16<<<(n/4 + 255)/256, 256, 0, ctx.stream()>>>((const float *) a->data, (const float *) z->data,
                                                                    y16.get(), n/4);
    CUDA_CHECK(cudaGetLastError());
    dense_f16_run(ctx, mm->src[0], y16.get(), mm->src[1]->ne[1], mm);
}

// DSV4_HC_POST (weights 2*sigmoid(x/4) folded in, optional) -> RMS_NORM -> MUL as one pass on any GPU and batch size
// (no down projection); mostly saves launches in decode

bool ggml_cuda_hc_combine_norm_supported(const ggml_tensor * post_node, const ggml_tensor * rms, const ggml_tensor * mul,
                                         const ggml_tensor * raw_post) {
    static const bool disabled = [] { const char * e = getenv("GGML_HC_NO_COMBINE_NORM"); return e && atoi(e) != 0; }();
    if (disabled) {
        return false;
    }
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const ggml_tensor * bo = post_node->src[0];
    const ggml_tensor * rs = post_node->src[1];
    // one block per token: at decode sizes the separate, wider combine and norm kernels are faster
    // (gfx906 tg64: 29.6 fused vs 29.9 unfused)
    return x->ne[2] >= 16 && x == post_node && post_node->src[3] == nullptr && x->type == GGML_TYPE_F32 && x->ne[1] == 4 && x->ne[3] == 1 &&
        ggml_is_contiguous(x) && x->ne[0] % 4 == 0 && 4*x->ne[0] <= HC_NORM_THREADS*HC_NORM_MAXV4*4 &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == x->ne[0] && w->ne[1] == 4 && w->ne[2] == 1 &&
        ggml_is_contiguous(mul) && ggml_are_same_shape(mul, x) && bo->type == GGML_TYPE_F32 && rs->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(rs) && ggml_are_same_shape(rs, x) && bo->ne[0] == x->ne[0] && bo->nb[0] == sizeof(float) &&
        bo->nb[1] % 16 == 0 && ((uintptr_t) bo->data) % 16 == 0 && ((uintptr_t) rs->data) % 16 == 0 &&
        ((uintptr_t) mul->data) % 16 == 0 && ((uintptr_t) post_node->data) % 16 == 0 && ((uintptr_t) w->data) % 16 == 0 &&
        hc_combine_norm_aliasing_ok(post_node, mul, w, raw_post);
}

void ggml_cuda_hc_combine_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post_node, const ggml_tensor * rms,
                               ggml_tensor * mul, const ggml_tensor * raw_post, const float ps1, const float ps2) {
    const ggml_tensor * x  = rms->src[0];
    const ggml_tensor * w  = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const ggml_tensor * bo = post_node->src[0];
    const ggml_tensor * pw = raw_post ? raw_post : post_node->src[2];
    const int n_embd = x->ne[0];
    const int n_tok  = x->ne[2];
    (raw_post ? hc_combine_norm<true, true, true, false> : hc_combine_norm<true, true, false, false>)
        <<<n_tok, HC_NORM_THREADS, 0, ctx.stream()>>>(
        (const float *) post_node->src[1]->data, (const float *) bo->data, (const float *) pw->data,
        (const float *) w->data, (float *) post_node->data, (float *) mul->data, nullptr,
        n_embd, post_node->src[1]->nb[2]/sizeof(float), bo->nb[1]/sizeof(float), pw->nb[0]/sizeof(float),
        pw->nb[1]/sizeof(float), post_node->nb[2]/sizeof(float), mul->nb[2]/sizeof(float),
        ggml_get_op_params_f32(rms, 0), ps1, ps2, 0);
    CUDA_CHECK(cudaGetLastError());
}

// Decode / verify (< 16 tokens): one block per token as in hc_combine_norm, but 1024 threads (256 per stream, each
// with ~2.5 float4) instead of 256 threads with 10 dependent float4 each (at 1 token that single block lost to the
// separate combine and norm kernels). Same read-before-barrier order, so the same aliasing rules. Saves the norm launch.
template <bool RAWP>
__launch_bounds__(1024)
static __global__ void hc_combine_norm_dec(
        const float * __restrict__ residual, const float * __restrict__ bo, const float * __restrict__ post,
        const float * __restrict__ w, float * __restrict__ x_out, float * __restrict__ xn_out,
        const int n_embd, const int64_t s_res, const int64_t s_bo, const int64_t sp0, const int64_t sp1,
        const int64_t s_x, const int64_t s_xn, const float eps, const float ps1, const float ps2,
        block_q8_1 * __restrict__ xq, const int64_t s_xq) {
    constexpr int MAXV = 4; // float4 per thread: n_embd <= 4096
    constexpr int GW   = 256/WARP_SIZE; // warps per stream
    const int64_t t = blockIdx.x;
    const int s   = threadIdx.x / 256;
    const int tid = threadIdx.x % 256;
    const int sv4 = n_embd/4;
    float p = post[s*sp0 + t*sp1];
    if constexpr (RAWP) {
        p = ps2 / (1.0f + expf(-ps1*p));
    }
    const float4 * rp = (const float4 *) (residual + t*s_res + (int64_t) s*n_embd);
    const float4 * bp = (const float4 *) (bo + t*s_bo);
    float4 v[MAXV];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int j = tid + k*256;
        if (j < sv4) {
            float4 a = rp[j];
            const float4 b = bp[j];
            a.x += b.x*p; a.y += b.y*p; a.z += b.z*p; a.w += b.w*p;
            v[k] = a;
            ss += a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w;
        }
    }
    __shared__ float red[4*GW];
    ss = warp_reduce_sum(ss);
    if (tid % WARP_SIZE == 0) {
        red[s*GW + tid / WARP_SIZE] = ss;
    }
    __syncthreads(); // every input except w has been read by the whole block
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < GW; ++i) {
        tot += red[s*GW + i];
    }
    const float r = rsqrtf(tot/n_embd + eps);
    const float4 * wp = (const float4 *) (w + (int64_t) s*n_embd);
    float4 * xp = (float4 *) (x_out + t*s_x + (int64_t) s*n_embd);
    float4 * op = (float4 *) (xn_out + t*s_xn + (int64_t) s*n_embd);
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int j = tid + k*256;
        if (j < sv4) {
            xp[j] = v[k];
            const float4 g = wp[j];
            const float4 o = make_float4(v[k].x*r*g.x, v[k].y*r*g.y, v[k].z*r*g.z, v[k].w*r*g.w);
            op[j] = o;
            if (xq) {
                // q8_1 copy for the down projection (same scale and quants as quantize_q8_1): 8 lanes = one 32-value block
                float amax = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
                float sum  = (o.x + o.y) + (o.z + o.w);
#pragma unroll
                for (int off = 1; off < 8; off *= 2) {
                    amax = fmaxf(amax, __shfl_xor(amax, off, 8));
                    sum += __shfl_xor(sum, off, 8);
                }
                const float d = amax / 127.0f;
                char4 q;
                q.x = amax == 0.0f ? 0 : roundf(o.x / d);
                q.y = amax == 0.0f ? 0 : roundf(o.y / d);
                q.z = amax == 0.0f ? 0 : roundf(o.z / d);
                q.w = amax == 0.0f ? 0 : roundf(o.w / d);
                const int e0 = s*n_embd + 4*j; // value index in the token's row of 4*n_embd
                block_q8_1 * b = xq + t*s_xq + e0/QK8_1;
                *(char4 *) (b->qs + e0 % QK8_1) = q;
                if (e0 % QK8_1 == 0) {
                    b->ds = make_half2(d, sum);
                }
            }
        }
    }
}

bool ggml_cuda_hc_combine_norm_dec_supported(const ggml_tensor * post_node, const ggml_tensor * rms, const ggml_tensor * mul,
                                             const ggml_tensor * raw_post) {
    static const bool disabled = [] { const char * e = getenv("GGML_HC_NO_COMBINE_NORM_DEC"); return e && atoi(e) != 0; }();
    if (disabled) {
        return false;
    }
    const ggml_tensor * x  = rms->src[0];
    const ggml_tensor * w  = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const ggml_tensor * bo = post_node->src[0];
    const ggml_tensor * rs = post_node->src[1];
    const ggml_tensor * pw = raw_post ? raw_post : post_node->src[2];
    return x->ne[2] < 16 && x == post_node && post_node->src[3] == nullptr && x->type == GGML_TYPE_F32 && x->ne[1] == 4 &&
        x->ne[3] == 1 && ggml_is_contiguous(x) && x->ne[0] % 4 == 0 && x->ne[0] <= 4096 &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == x->ne[0] && w->ne[1] == 4 && w->ne[2] == 1 &&
        ggml_is_contiguous(mul) && ggml_are_same_shape(mul, x) && bo->type == GGML_TYPE_F32 && rs->type == GGML_TYPE_F32 &&
        pw->type == GGML_TYPE_F32 && ggml_is_contiguous(rs) && ggml_are_same_shape(rs, x) && bo->ne[0] == x->ne[0] &&
        bo->nb[0] == sizeof(float) && bo->nb[1] % 16 == 0 && ((uintptr_t) bo->data) % 16 == 0 &&
        ((uintptr_t) rs->data) % 16 == 0 && ((uintptr_t) mul->data) % 16 == 0 && ((uintptr_t) post_node->data) % 16 == 0 &&
        ((uintptr_t) w->data) % 16 == 0 && hc_combine_norm_aliasing_ok(post_node, mul, w, raw_post);
}

void ggml_cuda_hc_combine_norm_dec(ggml_backend_cuda_context & ctx, ggml_tensor * post_node, const ggml_tensor * rms,
                                   ggml_tensor * mul, const ggml_tensor * raw_post, const float ps1, const float ps2,
                                   void * xq, const int64_t s_xq) {
    const ggml_tensor * x  = rms->src[0];
    const ggml_tensor * w  = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const ggml_tensor * bo = post_node->src[0];
    const ggml_tensor * pw = raw_post ? raw_post : post_node->src[2];
    (raw_post ? hc_combine_norm_dec<true> : hc_combine_norm_dec<false>)<<<(unsigned) x->ne[2], 1024, 0, ctx.stream()>>>(
        (const float *) post_node->src[1]->data, (const float *) bo->data, (const float *) pw->data,
        (const float *) w->data, (float *) post_node->data, (float *) mul->data,
        (int) x->ne[0], post_node->src[1]->nb[2]/sizeof(float), bo->nb[1]/sizeof(float), pw->nb[0]/sizeof(float),
        pw->nb[1]/sizeof(float), post_node->nb[2]/sizeof(float), mul->nb[2]/sizeof(float),
        ggml_get_op_params_f32(rms, 0), ps1, ps2, (block_q8_1 *) xq, s_xq);
    CUDA_CHECK(cudaGetLastError());
}

// xn stored as F16 in place must not overlap the combine's inputs (except exactly replacing the residual)
bool ggml_cuda_hc_xn_f16_inplace_safe(const ggml_tensor * post_node, const ggml_tensor * mul, const ggml_tensor * raw_post) {
    if (!post_node) {
        return true; // no-post variant: xn must not overlap x at all (ggml_cuda_hc_norm_down_supported)
    }
    const ggml_tensor * pw  = raw_post ? raw_post : post_node->src[2];
    const ggml_tensor * res = post_node->src[1];
    // the F16 rows are strided like the F32 rows (row t at the start of its own F32 row), so xn exactly replacing the
    // residual is fine: block t only writes inside token t's footprint after reading it
    const bool on_res = mul->data == res->data && ggml_are_same_shape(mul, res) && ggml_are_same_stride(mul, res);
    return !hc_ranges_overlap(mul, post_node->src[0]) && (!hc_ranges_overlap(mul, res) || on_res) && !hc_ranges_overlap(mul, pw);
}
