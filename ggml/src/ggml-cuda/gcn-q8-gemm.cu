#include "gcn-q8-gemm.cuh"
#include "mmid.cuh"
#include "vecdotq.cuh"

// Dense q8_0 x q8_1 GEMM for GCN (gfx906), no matrix cores: a 256-thread block computes 128 weight rows x 64 tokens,
// each thread an 8 x 4 register tile. Per 32-value block the thread keeps integer dp4a sums and applies the two block
// scales once, so the inner loop is 12 LDS reads per 32 dp4a. Rows of a thread are interleaved (row group + 16*i) and
// LDS rows padded to 9 dwords, so the 16 row groups of a wave hit 16 different banks. The next K step (4 blocks) is
// fetched into registers while the current one is computed from LDS.

static constexpr int GQ_BM  = 128; // weight rows per block
static constexpr int GQ_BN  = 64;  // tokens per block
static constexpr int GQ_NT  = 256;
static constexpr int GQ_KB  = 4;   // q8_0 blocks (32 values) per K step
#define GQ_PSTR(x) #x
#define GQ_PRAGMA(x) _Pragma(GQ_PSTR(x))
static constexpr int GQ_PAD = 9;   // dwords per LDS row (8 + 1)

static __global__ void gcn_q8_quantize(const float * __restrict__ x, block_q8_1 * __restrict__ y, const int64_t K,
        const int64_t s_col) {
    const int64_t i0  = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t col = blockIdx.y;
    if (i0 >= K) {
        return;
    }
    const float xi = x[col*s_col + i0];
    float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    float sum  = warp_reduce_sum<QK8_1>(xi);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    block_q8_1 * yb = y + (col*K + i0) / QK8_1;
    yb->qs[i0 % QK8_1] = q;
    if (i0 % QK8_1 == 0) {
        yb->ds = make_half2(d, sum);
    }
}

// 4 consecutive values per thread (one 16-byte load, one 4-byte store of quants), 8 lanes per 32-value block.
// The one-value-per-thread version above ran at ~200 GB/s (320 x 10240: 66 us).
static __global__ void gcn_q8_quantize4(const float * __restrict__ x, block_q8_1 * __restrict__ y, const int64_t K,
        const int64_t s_col) {
    const int64_t i0  = 4*((int64_t) blockDim.x*blockIdx.x + threadIdx.x);
    const int64_t col = blockIdx.y;
    if (i0 >= K) {
        return; // K % 32 == 0: whole 8-lane groups exit together
    }
    const float4 xv = *(const float4 *) (x + col*s_col + i0);
    float amax = fmaxf(fmaxf(fabsf(xv.x), fabsf(xv.y)), fmaxf(fabsf(xv.z), fabsf(xv.w)));
    float sum  = (xv.x + xv.y) + (xv.z + xv.w);
#pragma unroll
    for (int off = 1; off < QK8_1/4; off *= 2) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 64));
        sum += __shfl_xor(sum, off, 64);
    }
    const float d = amax / 127.0f;
    char4 q;
    q.x = amax == 0.0f ? 0 : roundf(xv.x / d);
    q.y = amax == 0.0f ? 0 : roundf(xv.y / d);
    q.z = amax == 0.0f ? 0 : roundf(xv.z / d);
    q.w = amax == 0.0f ? 0 : roundf(xv.w / d);
    block_q8_1 * yb = y + (col*K + i0) / QK8_1;
    *(char4 *) (yb->qs + i0 % QK8_1) = q;
    if (i0 % QK8_1 == 0) {
        yb->ds = make_half2(d, sum);
    }
}

#ifndef GQ_MINB
#define GQ_MINB 2
#endif
// MIX: HC up projection fused with the gated DSV4_HC_PRE (4 streams of E rows): the tile's 128 rows are 32 embedding
// columns x 4 streams (tile row lr -> weight row (lr/32)*E + e0 + lr%32), so a thread's 8 rows rg + 16 i are 2 columns
// x 4 streams and the epilogue writes pre[e,t] = scale * sum_c xn[e,c,t] * sigmoid(g[c*E+e,t]) without storing g
struct gcn_q8_mix_args {
    const float * xn; float * pre; int E; int64_t sx1, sx2, sp1; float scale;
};

#ifndef GQ_MIX_MINB
#define GQ_MIX_MINB 2
#endif
template <bool MIX = false>
__launch_bounds__(GQ_NT, MIX ? GQ_MIX_MINB : GQ_MINB)
// split-K: blockIdx.z covers blocks [z*kb_per_split, ...) of K and writes dst + z*split_stride (partials, reduced after)
static __global__ void gcn_q8_gemm(const char * __restrict__ w, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int M, const int N, const int K, const int64_t stride_row_w, const int64_t s1_dst,
        const int kb_per_split, const int64_t split_stride, const gcn_q8_mix_args mix = {}) {
#ifndef GQ_B64
#define GQ_B64 0
#endif
    // GQ_B64: 8-byte LDS reads (A rows padded to 10 dwords, 16 row groups -> 16 distinct even banks; B unpadded,
    // only 4 distinct columns per wave read)
    constexpr int APAD = GQ_B64 ? 10 : GQ_PAD;
    constexpr int BPAD = GQ_B64 ? 8  : GQ_PAD;
    __shared__ __align__(16) int   aq[GQ_KB][GQ_BM][APAD];
    __shared__ float ad[GQ_KB][GQ_BM];
    __shared__ __align__(16) int   bq[GQ_KB][GQ_BN][BPAD];
    __shared__ float bd[GQ_KB][GQ_BN];

    const int tid  = threadIdx.x;
#ifndef GQ_TOKENS_FAST
#define GQ_TOKENS_FAST 1
#endif
    // GQ_TOKENS_FAST: blockIdx.x walks the token tiles, so the blocks sharing a weight tile run together and all but
    // the first read it from L2 (row tiles fastest re-read every weight tile from DRAM once per token tile)
    const int row0 = (GQ_TOKENS_FAST ? blockIdx.y : blockIdx.x)*GQ_BM;
    const int col0 = (GQ_TOKENS_FAST ? blockIdx.x : blockIdx.y)*GQ_BN;
    const int nkb_all = K / QK8_0;          // blocks along K (rows of y)
    const int kb_beg  = blockIdx.z*kb_per_split;
    const int nkb     = min(kb_per_split, nkb_all - kb_beg); // blocks of this split
    const int nstep = (nkb + GQ_KB - 1) / GQ_KB; // the last step may be partial: missing blocks load as zeros
    dst += blockIdx.z*split_stride;

    // loader roles: A: row tid/2, blocks 2*(tid%2) and +1 of the step (68 contiguous bytes, 4-byte aligned)
    //               B: column tid/4, block tid%4 of the step (one 36-byte block_q8_1)
    const int  la_row = tid / 2;
    const int  la_h   = tid % 2;
    const int  wrow   = MIX ? (la_row/32)*mix.E + (GQ_TOKENS_FAST ? blockIdx.y : blockIdx.x)*32 + la_row%32 : min(row0 + la_row, M - 1);
    const char * wa   = w + (int64_t) wrow*stride_row_w + (kb_beg + la_h*2)*sizeof(block_q8_0);
    const int  lb_col = tid / 4;
    const int  lb_kb  = tid % 4;
    const block_q8_1 * yb = y + (int64_t) min(col0 + lb_col, N - 1)*nkb_all + kb_beg + lb_kb;

    // compute roles
    const int rg = tid % 16;  // rows rg + 16*i
    const int cg = tid / 16;  // cols cg + 16*j

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    int ra[17];
    int rb[9];
    auto fetch_full = [&](const int step) {
        const int * pa = (const int *) (wa + (int64_t) step*GQ_KB*sizeof(block_q8_0));
#pragma unroll
        for (int i = 0; i < 17; ++i) {
            ra[i] = pa[i];
        }
        const int * pb = (const int *) (yb + step*GQ_KB);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            rb[i] = pb[i];
        }
    };
    auto fetch_partial = [&](const int step) {
        const int kba = step*GQ_KB + 2*la_h; // first of the loader's two A blocks
        const int * pa = (const int *) (wa + (int64_t) step*GQ_KB*sizeof(block_q8_0));
        if (kba + 1 < nkb) {
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                ra[i] = pa[i];
            }
        } else {
            // partial K step: first block only (dwords 0..8, bytes 0..35 cover it), or none
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                ra[i] = (kba < nkb && i < 8) ? pa[i] : 0;
            }
            if (kba < nkb) {
                ra[8] = *(const unsigned short *) (pa + 8); // bytes 32-33 end the block (no read past the row)
            }
        }
        const int * pb = (const int *) (yb + step*GQ_KB);
        const bool bvalid = step*GQ_KB + lb_kb < nkb;
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            rb[i] = bvalid ? pb[i] : 0;
        }
    };
    auto store = [&]() {
        // block 2h: d = bytes 0-1, quants at byte 2 (shifted by 2 bytes); block 2h+1: d = bytes 34-35, quants at 36
        const int kb0 = 2*la_h;
        ad[kb0    ][la_row] = __half2float(__ushort_as_half((unsigned short) (ra[0] & 0xFFFF)));
        ad[kb0 + 1][la_row] = __half2float(__ushort_as_half((unsigned short) ((unsigned) ra[8] >> 16)));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            aq[kb0    ][la_row][i] = __builtin_amdgcn_alignbyte(ra[i + 1], ra[i], 2);
            aq[kb0 + 1][la_row][i] = ra[9 + i];
        }
        // block_q8_1: ds (half2) first, then the 32 quants
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            bq[lb_kb][lb_col][i] = rb[1 + i];
        }
        bd[lb_kb][lb_col] = __half2float(__ushort_as_half((unsigned short) (rb[0] & 0xFFFF)));
    };

    const int nfull = nkb / GQ_KB; // steps without missing blocks
    auto fetch = [&](const int step) {
        if (step < nfull) {
            fetch_full(step);
        } else {
            fetch_partial(step);
        }
    };
    fetch(0);
    for (int step = 0; step < nstep; ++step) {
        store();
        __syncthreads();
        if (step + 1 < nstep) {
            fetch(step + 1);
        }
#ifndef GQ_KB_UNROLL
#define GQ_KB_UNROLL 1
#endif
#pragma unroll GQ_KB_UNROLL
        for (int kb = 0; kb < GQ_KB; ++kb) {
#ifndef GQ_HALVES
#define GQ_HALVES 1
#endif
          if constexpr (GQ_HALVES) {
            // rows in two halves of 4: 16 live integer sums instead of 32 (fewer VGPRs, more waves), B read twice
            float db[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                db[j] = bd[kb][cg + 16*j];
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int s[4][4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        s[i][j] = 0;
                    }
                }
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    int a[4], b[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        a[i] = aq[kb][rg + 16*(4*h + i)][kk];
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        b[j] = bq[kb][cg + 16*j][kk];
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            s[i][j] = ggml_cuda_dp4a(a[i], b[j], s[i][j]);
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float da = ad[kb][rg + 16*(4*h + i)];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[4*h + i][j] = fmaf((float) s[i][j], da*db[j], acc[4*h + i][j]);
                    }
                }
            }
            continue;
          }
            int s[8][4];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[i][j] = 0;
                }
            }
            if constexpr (GQ_B64) {
#pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) {
                    int2 a[8], b[4];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        a[i] = *(const int2 *) &aq[kb][rg + 16*i][2*k2];
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        b[j] = *(const int2 *) &bq[kb][cg + 16*j][2*k2];
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            s[i][j] = ggml_cuda_dp4a(a[i].x, b[j].x, s[i][j]);
                            s[i][j] = ggml_cuda_dp4a(a[i].y, b[j].y, s[i][j]);
                        }
                    }
                }
            } else {
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                int a[8], b[4];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    a[i] = aq[kb][rg + 16*i][kk];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    b[j] = bq[kb][cg + 16*j][kk];
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        s[i][j] = ggml_cuda_dp4a(a[i], b[j], s[i][j]);
                    }
                }
            }
            }
            float da[8], db[4];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                da[i] = ad[kb][rg + 16*i];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                db[j] = bd[kb][cg + 16*j];
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf((float) s[i][j], da[i]*db[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    if constexpr (MIX) {
        const int e0 = (GQ_TOKENS_FAST ? blockIdx.y : blockIdx.x)*32;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + cg + 16*j;
            if (col >= N) {
                continue;
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int e = e0 + rg + 16*h;
                float sum = 0.0f;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float xv = mix.xn[e + c*mix.sx1 + col*mix.sx2];
                    sum += xv * (1.0f / (1.0f + expf(-acc[2*c + h][j])));
                }
                mix.pre[e + col*mix.sp1] = mix.scale * sum;
            }
        }
        return;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = col0 + cg + 16*j;
        if (col >= N) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + rg + 16*i;
            if (row < M) {
                dst[(int64_t) col*s1_dst + row] = acc[i][j];
            }
        }
    }
}

static __global__ void gcn_q8_splitk_reduce(const float * __restrict__ part, float * __restrict__ dst, const int nsplit,
        const int64_t M, const int64_t N, const int64_t s1_dst) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= M*N) {
        return;
    }
    float sum = 0.0f;
    for (int z = 0; z < nsplit; ++z) {
        sum += part[z*M*N + i];
    }
    dst[(i / M)*s1_dst + i % M] = sum;
}


// ---- K-quant and iq4_xs weights (q4_K, q5_K, q6_K, iq4_xs) with the same tiling and compute loop ----
// A K step is half a 256-value super-block (4 sub-blocks of 32, the q8_0 kernel's 4 blocks). The loader unpacks the
// sub-blocks to int8 values in LDS plus per-row factors: d*sc and dmin*m (q4_K, q5_K: the min term is subtracted with
// the activation block sum that block_q8_1 carries), d*(ls-32) (iq4_xs), and for q6_K the two int8 scales of the
// 16-value halves with the half d packed in one word. Activation scale and sum stay the packed half2 of block_q8_1,
// so the LDS use is exactly 32 KiB (two blocks per CU, as the q8_0 kernel). Qwen3.8-27B UD-Q4_K_XL prefill is ~94%
// matmuls in these types; MMQ reached ~12 TFLOPS on gfx906 for them.
template <ggml_type T> struct gkq;
template <> struct gkq<GGML_TYPE_Q4_K>   { static constexpr int bytes = sizeof(block_q4_K);   static constexpr int nfetch = 12; };
template <> struct gkq<GGML_TYPE_Q5_K>   { static constexpr int bytes = sizeof(block_q5_K);   static constexpr int nfetch = 14; };
template <> struct gkq<GGML_TYPE_IQ4_XS> { static constexpr int bytes = sizeof(block_iq4_xs); static constexpr int nfetch = 10; };
template <> struct gkq<GGML_TYPE_IQ2_XS>  { static constexpr int bytes = sizeof(block_iq2_xs);  static constexpr int nfetch = 4; };
template <> struct gkq<GGML_TYPE_IQ2_XXS> { static constexpr int bytes = sizeof(block_iq2_xxs); static constexpr int nfetch = 4; };
template <> struct gkq<GGML_TYPE_IQ3_XXS> { static constexpr int bytes = sizeof(block_iq3_xxs); static constexpr int nfetch = 4; };
template <> struct gkq<GGML_TYPE_Q6_K>   { static constexpr int bytes = sizeof(block_q6_K);   static constexpr int nfetch = 15; };
// q3_K: 110-byte super-blocks (2-byte aligned); scales per 16 values, so it runs the q6_K compute path
template <> struct gkq<GGML_TYPE_Q3_K>   { static constexpr int bytes = sizeof(block_q3_K);   static constexpr int nfetch = 12; };
// q2_K: 84-byte super-blocks (4-byte aligned), v4 only
template <> struct gkq<GGML_TYPE_Q2_K>   { static constexpr int bytes = sizeof(block_q2_K);   static constexpr int nfetch = 5; };
// iq4_nl: 8 blocks of 32 (18 bytes each) per 256 values; a loader pair (2 blocks, 36 bytes) starts 4-byte aligned
template <> struct gkq<GGML_TYPE_IQ4_NL> { static constexpr int bytes = 8*sizeof(block_iq4_nl); static constexpr int nfetch = 9; };
// q8_0 (v4 only): 8 blocks of 32 (34 bytes each: f16 scale, 32 int8) per 256 values; GLM-5-Next keeps its KDA and MLA
// projections in q8_0
template <> struct gkq<GGML_TYPE_Q8_0>   { static constexpr int bytes = 8*sizeof(block_q8_0);   static constexpr int nfetch = 10; };

// 4-byte load at a 2-byte aligned address (q6_K super-blocks are 210 bytes; the alignment is uniform per super-block)
static __device__ __forceinline__ int gkq_ld(const char * p) {
    const uintptr_t a = (uintptr_t) p;
    if ((a & 3) == 0) {
        return *(const int *) p;
    }
    const int * q = (const int *) (a & ~(uintptr_t) 3);
    return (int) __builtin_amdgcn_alignbyte(q[1], q[0], (unsigned) (a & 3));
}

// byte i of the 12-byte K-quant scales held in w[0..2]. i may be a runtime value: select the register instead of
// indexing the array (a runtime index into a register array puts the whole array in scratch memory)
static __device__ __forceinline__ int gkq_byte(const int * w, const int i) {
    const int d = i < 4 ? w[0] : (i < 8 ? w[1] : w[2]);
    return (d >> (8*(i & 3))) & 0xFF;
}

// get_scale_min_k4 on the 12 scale bytes held in 3 registers
static __device__ __forceinline__ void gkq_scale_min_k4(const int j, const int * sc, int & d, int & m) {
    if (j < 4) {
        d = gkq_byte(sc, j) & 63;
        m = gkq_byte(sc, j + 4) & 63;
    } else {
        d = (gkq_byte(sc, j + 4) & 0xF) | ((gkq_byte(sc, j - 4) >> 6) << 4);
        m = (gkq_byte(sc, j + 4) >>  4) | ((gkq_byte(sc, j    ) >> 6) << 4);
    }
}

template <ggml_type T>
__launch_bounds__(GQ_NT, GQ_MINB)
static __global__ void gcn_kq_gemm(const char * __restrict__ w, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int M, const int N, const int K, const int64_t stride_row_w, const int64_t s1_dst,
        const int sb_per_split, const int64_t split_stride) {
    constexpr bool HAS_MIN = T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K;
    constexpr bool IS_Q6K  = T == GGML_TYPE_Q6_K || T == GGML_TYPE_Q3_K; // two factors per sub-block (16-value halves)
    __shared__ __align__(16) int   aq[GQ_KB][GQ_BM][GQ_PAD];
    __shared__ float ad[GQ_KB][GQ_BM];   // d*sc (q4_K, q5_K), d*(ls-32) (iq4_xs), d*sc of values 0-15 (q6_K)
    __shared__ float am[GQ_KB][GQ_BM];   // dmin*m (q4_K, q5_K), d*sc of values 16-31 (q6_K)
    __shared__ __align__(16) int   bq[GQ_KB][GQ_BN][GQ_PAD];
    __shared__ half2 bds[GQ_KB][GQ_BN];  // block_q8_1 ds: (d, sum of the 32 activations)

    const int tid  = threadIdx.x;
    const int row0 = (GQ_TOKENS_FAST ? blockIdx.y : blockIdx.x)*GQ_BM;
    const int col0 = (GQ_TOKENS_FAST ? blockIdx.x : blockIdx.y)*GQ_BN;
    const int nsb_all = K / QK_K;
    const int sb_beg  = blockIdx.z*sb_per_split;
    const int nsb     = min(sb_per_split, nsb_all - sb_beg);
    const int nstep   = 2*nsb; // two K steps per super-block
    const int nkb_all = K / QK8_1;
    dst += blockIdx.z*split_stride;

    // loader roles: A: row tid/2; q4_K/q5_K/iq4_xs: sub-block pair tid%2 of the step, q6_K: values 16*(tid%2)..+15 of
    //               each of the step's 4 sub-blocks. B: column tid/4, block tid%4 of the step (one block_q8_1)
    const int  la_row = tid / 2;
    const int  la_h   = tid % 2;
    const int  wrow   = min(row0 + la_row, M - 1);
    const char * wa   = w + (int64_t) wrow*stride_row_w + (int64_t) sb_beg*gkq<T>::bytes;
    const int  lb_col = tid / 4;
    const int  lb_kb  = tid % 4;
    const block_q8_1 * yb = y + (int64_t) min(col0 + lb_col, N - 1)*nkb_all + sb_beg*(QK_K/QK8_1) + lb_kb;

    const int rg = tid % 16;
    const int cg = tid / 16;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    int ra[gkq<T>::nfetch];
    int rb[9];
    auto fetch = [&](const int step) {
        const char * b  = wa + (int64_t) (step/2)*gkq<T>::bytes;
        const int    hs = step % 2; // which half of the super-block
        if constexpr (T == GGML_TYPE_Q4_K) {
            const int p = 2*hs + la_h;  // sub-block pair 2p, 2p+1
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ra[i] = ((const int *) b)[i];                     // dm, scales[12]
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                ra[4 + k] = ((const int *) (b + 16 + 32*p))[k];  // qs of the pair
            }
        } else if constexpr (T == GGML_TYPE_Q5_K) {
            const int p = 2*hs + la_h;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ra[i] = ((const int *) b)[i];                     // dm, scales[12]
            }
            // qh[32]: only bits 2p, 2p+1 of each byte matter to this thread; pack them into 2 registers right away
            // (byte t of ra[4 + k/4] holds the 2 bits of byte t of qh dword k at bits 2*(k%4)) - fewer live VGPRs
            ra[4] = 0;
            ra[5] = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int qh = ((const int *) (b + 16))[k];
                ra[4 + k/4] |= ((qh >> (2*p)) & 0x03030303) << (2*(k % 4));
                ra[6 + k] = ((const int *) (b + 48 + 32*p))[k];   // qs of the pair
            }
        } else if constexpr (T == GGML_TYPE_IQ4_NL) {
            const int p = 2*hs + la_h; // blocks 2p, 2p+1: d0, qs0 (at +2), d1 (at +18), qs1 (at +20)
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                ra[k] = ((const int *) (b + 36*p))[k];
            }
        } else if constexpr (T == GGML_TYPE_IQ4_XS) {
            const int p = 2*hs + la_h;
            ra[0] = ((const int *) b)[0]; // d, scales_h
            ra[1] = ((const int *) b)[1]; // scales_l[4]
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                ra[2 + k] = ((const int *) (b + 8 + 32*p))[k];   // qs of sub-blocks 2p (16 bytes), 2p+1 (16 bytes)
            }
        } else if constexpr (T == GGML_TYPE_Q3_K) { // values 128*hs + 32*g + 16*la_h + (0..15) of g = 0..3
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                ra[k]     = gkq_ld(b + 32 + 32*hs + 16*la_h + 4*k); // qs: 2 bits per sub-block g
                ra[4 + k] = gkq_ld(b + 16*la_h + 4*k);             // hmask: bit 4*hs + g
            }
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                ra[8 + k] = gkq_ld(b + 96 + 4*k);                  // scales[12]
            }
            ra[11] = *(const unsigned short *) (b + 108);           // d
        } else { // q6_K: values 128*hs + 32*g + 16*la_h + (0..15) of g = 0..3
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                ra[k]     = gkq_ld(b + 64*hs +      16*la_h + 4*k); // ql, sub-blocks 0, 2 (low / high nibbles)
                ra[4 + k] = gkq_ld(b + 64*hs + 32 + 16*la_h + 4*k); // ql, sub-blocks 1, 3
                ra[8 + k] = gkq_ld(b + 128 + 32*hs + 16*la_h + 4*k); // qh (2 bits per sub-block)
            }
            ra[12] = gkq_ld(b + 192 + 8*hs);     // scales of the half: 16-value groups 0..3
            ra[13] = gkq_ld(b + 192 + 8*hs + 4); // groups 4..7
            ra[14] = *(const unsigned short *) (b + 208); // d
        }
        const int * pb = (const int *) (yb + step*GQ_KB);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            rb[i] = pb[i];
        }
#ifdef GKQ_NO_GMEM // measurement only: loads replaced by register values (wrong results)
#pragma unroll
        for (int i = 0; i < gkq<T>::nfetch; ++i) {
            ra[i] = 0x01020304 ^ (step*7 + i + tid);
        }
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            rb[i] = 0x3c003c00 ^ (step + i);
        }
#endif
    };
    auto store = [&](const int step) {
        const int hs = step % 2;
        if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K) {
            const int   p    = 2*hs + la_h;
            const half2 dm   = *(const half2 *) &ra[0];
            const float d    = __low2float(dm);
            const float dmin = __high2float(dm);
            const int kb0 = 2*la_h; // the pair's sub-blocks within the step
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                int sc, m;
                gkq_scale_min_k4(2*p + t, ra + 1, sc, m);
                ad[kb0 + t][la_row] = d*sc;
                am[kb0 + t][la_row] = dmin*m;
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int qs = T == GGML_TYPE_Q5_K ? ra[6 + k] : ra[4 + k];
                int lo = qs & 0x0F0F0F0F;
                int hi = (qs >> 4) & 0x0F0F0F0F;
                if constexpr (T == GGML_TYPE_Q5_K) {
                    const int hb = (ra[4 + k/4] >> (2*(k % 4))) & 0x03030303; // bit 0: sub-block 2p, bit 1: 2p+1
                    lo |= (hb & 0x01010101) << 4;
                    hi |= (hb & 0x02020202) << 3;
                }
                aq[kb0    ][la_row][k] = lo;
                aq[kb0 + 1][la_row][k] = hi;
            }
        } else if constexpr (T == GGML_TYPE_IQ4_NL) {
            const int kb0 = 2*la_h;
            ad[kb0    ][la_row] = __half2float(__ushort_as_half((unsigned short) (ra[0] & 0xFFFF)));
            ad[kb0 + 1][la_row] = __half2float(__ushort_as_half((unsigned short) ((unsigned) ra[4] >> 16)));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int q0 = (int) __builtin_amdgcn_alignbyte(ra[k + 1], ra[k], 2); // block 2p: bytes 2..17
                const int2 v0 = get_int_from_table_16(q0, kvalues_iq4nl);
                aq[kb0][la_row][k]     = v0.x;
                aq[kb0][la_row][4 + k] = v0.y;
                const int2 v1 = get_int_from_table_16(ra[5 + k], kvalues_iq4nl);
                aq[kb0 + 1][la_row][k]     = v1.x;
                aq[kb0 + 1][la_row][4 + k] = v1.y;
            }
        } else if constexpr (T == GGML_TYPE_IQ4_XS) {
            const int   p  = 2*hs + la_h;
            const float d  = __half2float(__ushort_as_half((unsigned short) (ra[0] & 0xFFFF)));
            const int   sh = (int) ((unsigned) ra[0] >> 16);
            const int kb0 = 2*la_h;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int j  = 2*p + t;
                const int ls = ((gkq_byte(ra + 1, j/2) >> (4*(j % 2))) & 0xF) | (((sh >> (2*j)) & 3) << 4);
                ad[kb0 + t][la_row] = d*(ls - 32);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int2 v = get_int_from_table_16(ra[2 + 4*t + k], kvalues_iq4nl);
                    aq[kb0 + t][la_row][k]     = v.x; // values 4k..4k+3 (low nibbles)
                    aq[kb0 + t][la_row][4 + k] = v.y; // values 16+4k.. (high nibbles)
                }
            }
        } else if constexpr (T == GGML_TYPE_Q3_K) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int qa = ra[k], hm = ra[4 + k];
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const int v = ((qa >> (2*g)) & 0x03030303) | (((hm >> (4*hs + g)) & 0x01010101) << 2); // 0..7
                    // bytes 0..7 -> signed q - 4 without borrows between bytes
                    aq[g][la_row][4*la_h + k] = (int) (((unsigned) (v | 0x80808080) - 0x04040404u) ^ 0x80808080u);
                }
            }
            // 16 6-bit scales (ggml's q3_K layout), this thread's 16-value groups 8*hs + 2*g + la_h
            {
                const unsigned a0 = ra[8], a1 = ra[9], tmp = ra[10];
                const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
                const unsigned s0 = (a0 & km2) | (((tmp >> 0) & km1) << 4);
                const unsigned s1 = (a1 & km2) | (((tmp >> 2) & km1) << 4);
                const unsigned s2 = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);
                const unsigned s3 = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);
                const unsigned sw0 = hs ? s2 : s0, sw1 = hs ? s3 : s1; // scales 8*hs .. 8*hs + 7
                const float d = __half2float(__ushort_as_half((unsigned short) ra[11]));
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const int j = 2*g + la_h; // 0..7 within the half
                    const unsigned w = j < 4 ? sw0 : sw1;
                    const int sc = (int) ((w >> (8*(j & 3))) & 0xFF) - 32;
                    const float f = d*(float) sc;
                    if (la_h == 0) {
                        ad[g][la_row] = f;
                    } else {
                        am[g][la_row] = f;
                    }
                }
            }
        } else { // q6_K
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int qa = ra[k], qb = ra[4 + k], qh = ra[8 + k];
                int v[4];
                v[0] = ( qa       & 0x0F0F0F0F) | (((qh     ) & 0x03030303) << 4);
                v[1] = ( qb       & 0x0F0F0F0F) | (((qh >> 2) & 0x03030303) << 4);
                v[2] = ((qa >> 4) & 0x0F0F0F0F) | (((qh >> 4) & 0x03030303) << 4);
                v[3] = ((qb >> 4) & 0x0F0F0F0F) | (((qh >> 6) & 0x03030303) << 4);
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    // bytes 0..63 -> signed q - 32 without borrows between bytes
                    aq[g][la_row][4*la_h + k] = (int) (((unsigned) (v[g] | 0x80808080) - 0x20202020u) ^ 0x80808080u);
                }
            }
            // the two 16-value halves' factors d*sc; each loader half writes one of them
            {
                const float d = __half2float(__ushort_as_half((unsigned short) ra[14]));
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const float f = d*(float) (int8_t) gkq_byte(ra + 12, 2*g + la_h);
                    if (la_h == 0) {
                        ad[g][la_row] = f;
                    } else {
                        am[g][la_row] = f;
                    }
                }
            }
        }
        // block_q8_1: ds (half2) first, then the 32 quants
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            bq[lb_kb][lb_col][i] = rb[1 + i];
        }
        bds[lb_kb][lb_col] = *(const half2 *) &rb[0];
    };

    fetch(0);
    for (int step = 0; step < nstep; ++step) {
        store(step);
        __syncthreads();
        if (step + 1 < nstep) {
            fetch(step + 1);
        }
#ifndef GKQ_KB_UNROLL
#define GKQ_KB_UNROLL 1
#endif
        GQ_PRAGMA(unroll GKQ_KB_UNROLL)
        for (int kb = 0; kb < GQ_KB; ++kb) {
            float db[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                db[j] = __low2float(bds[kb][cg + 16*j]);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int s[4][4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        s[i][j] = 0;
                    }
                }
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    int a[4], b[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        a[i] = aq[kb][rg + 16*(4*h + i)][kk];
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        b[j] = bq[kb][cg + 16*j][kk];
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            s[i][j] = ggml_cuda_dp4a(a[i], b[j], s[i][j]);
                        }
                    }
                    if constexpr (IS_Q6K) {
                        if (kk == 3) {
                            // values 0..15 of the sub-block done: apply their factor, restart the sums
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const float f = ad[kb][rg + 16*(4*h + i)];
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    acc[4*h + i][j] = fmaf((float) s[i][j], f*db[j], acc[4*h + i][j]);
                                    s[i][j] = 0;
                                }
                            }
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = rg + 16*(4*h + i);
                    if constexpr (IS_Q6K) {
                        const float f = am[kb][r];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc[4*h + i][j] = fmaf((float) s[i][j], f*db[j], acc[4*h + i][j]);
                        }
                    } else {
#ifdef GKQ_SKEL // measurement only: no scale math (wrong results)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc[4*h + i][j] += (float) s[i][j];
                        }
                        if constexpr (false) {
#else
                        const float da = ad[kb][r];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc[4*h + i][j] = fmaf((float) s[i][j], da*db[j], acc[4*h + i][j]);
                        }
                        if constexpr (HAS_MIN) {
#endif
                            const float ma = am[kb][r];
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc[4*h + i][j] = fmaf(-ma, __high2float(bds[kb][cg + 16*j]), acc[4*h + i][j]);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = col0 + cg + 16*j;
        if (col >= N) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + rg + 16*i;
            if (row < M) {
                dst[(int64_t) col*s1_dst + row] = acc[i][j];
            }
        }
    }
}

// ---- v3 (opt-in, GGML_CUDA_GCN_KQV=3): integer sub-block scales (q4_K, q5_K, q6_K, iq4_xs, q3_K) ----
// v1 applies two float factors per 32-value sub-block and output (cvt + mul + fma, plus the q4_K/q5_K min fma): ~30% of
// its time (a variant without them ran 41% faster). v3 quantizes the activations per 128 values (one K step, 4 sub-blocks)
// instead of per 32, so a step's sub-block dot products are summed in int32 with the weights' integer sub-block scales
// (one v_mad_i32_i24 per sub-block and output) and floats are touched once per step and output. The q4_K/q5_K min term
// -dmin*sum_kb m_kb*bsum_kb uses the activation sub-block sums as int16 pairs: two v_dot2_i32_i16 per step and output.
// The int32 step sums next to the dp4a sums and float accumulators need registers: a 128 x 64 tile with 8 x 4 outputs
// per thread (two blocks per CU, 128 VGPRs) spilled, so v3 uses 64 x 64 tiles with a 4 x 4 register tile per thread:
// ~80 VGPRs and ~20 KiB of LDS, three blocks per CU (84 VGPRs), three waves per SIMD to cover LDS and barrier latency.
// LDS: weight rows padded to 10 dwords (8-byte aligned, the 16 rows a wave reads land on 16 distinct bank pairs), token
// rows unpadded (a wave reads 4 tokens, broadcast): every compute read is a ds_read_b64 at a constant offset from one
// base per thread. Loaders: A: row tid/4, quarter u = tid%4 of the step's quant bytes plus the scale (and min) of step
// sub-block u; B: token tid/4, sub-block tid%4 (32 int8), and the step's meta for tid%4 == 0. Global loads are issued at
// the step start (no register prefetch): the other blocks of the CU compute meanwhile.
// The coarser activation scale costs accuracy: KLD vs MMQ (per-32 q8_1) 0.014 vs 0.009 for v1 on Qwen3.8-27B (8 x 2048
// wikitext; per-64 scales: 0.0125; power-of-2 sub-scales per 32 did not fit the registers), hence v4 by default.
// Activation format: qs int8 [N][K]; meta int4 [N][K/128] = (d as float bits, sums of sub-blocks 0|1 and 2|3 as int16
// pairs, 0).
static __global__ void gcn_q8s_quantize(const float * __restrict__ x, int8_t * __restrict__ qs, int4 * __restrict__ meta,
        const int64_t K, const int64_t s_col) {
    // 4 waves per block, a wave per 128-value block, 2 values per lane (lanes 16*kb .. 16*kb + 15: sub-block kb)
    const int64_t col = blockIdx.y;
    const int wv = threadIdx.x >> 6, l = threadIdx.x & 63;
    const int64_t b = (int64_t) blockIdx.x*4 + wv;
    if (b*128 >= K) {
        return;
    }
    const float2 v = *(const float2 *) (x + col*s_col + b*128 + 2*l);
    float amax = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
    for (int off = 1; off < 64; off *= 2) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 64));
    }
    const float d  = amax / 127.0f;
    const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
    const int q0 = (int) roundf(v.x*id), q1 = (int) roundf(v.y*id);
    *(short *) (qs + col*K + b*128 + 2*l) = (short) ((q0 & 0xFF) | ((q1 & 0xFF) << 8));
    int s = q0 + q1;
#pragma unroll
    for (int off = 1; off < 16; off *= 2) {
        s += __shfl_xor(s, off, 64);
    }
    const int s0 = __shfl(s, 0, 64), s1 = __shfl(s, 16, 64), s2 = __shfl(s, 32, 64), s3 = __shfl(s, 48, 64);
    if (l == 0) {
        meta[col*(K/128) + b] = make_int4(__float_as_int(d), (s0 & 0xFFFF) | (s1 << 16), (s2 & 0xFFFF) | (s3 << 16), 0);
    }
}

static constexpr int G3_BM   = 64;
static constexpr int G3_BN   = 64;
static constexpr int G3_PADA = 10;
static constexpr int G3_PADB = 8;
#ifndef G3_MINB
#define G3_MINB 3
#endif
#ifndef G3_KB_UNROLL
#define G3_KB_UNROLL 4
#endif

// 8 / 4 bytes at a 2-byte aligned address (q6_K, q3_K super-blocks), without branches (a branch in the fetch let the
// compiler sink the dp4a below it) and without reading past the last aligned dword
static __device__ __forceinline__ int2 gkq_ld2(const char * p) {
    const uintptr_t a  = (uintptr_t) p;
    const int *     q  = (const int *) (a & ~(uintptr_t) 3);
    const unsigned  sh = (unsigned) (a & 3);
    const int x0 = q[0], x1 = q[1], x2 = q[sh ? 2 : 1];
    return make_int2((int) __builtin_amdgcn_alignbyte(x1, x0, sh), (int) __builtin_amdgcn_alignbyte(x2, x1, sh));
}
static __device__ __forceinline__ int gkq_ld1(const char * p) {
    const uintptr_t a  = (uintptr_t) p;
    const int *     q  = (const int *) (a & ~(uintptr_t) 3);
    const unsigned  sh = (unsigned) (a & 3);
    return (int) __builtin_amdgcn_alignbyte(q[sh ? 1 : 0], q[0], sh);
}
// a q8_0 block (34 bytes at a 2-byte aligned address): its 32 quants as 8 dwords and its f16 scale bits, from 10
// aligned dword loads (the quants start 0 or 2 bytes into a dword; neither read leaves the block's aligned span)
static __device__ __forceinline__ void gkq_ld_q8_0(const char * blk, int * q, unsigned & d) {
    const uintptr_t a  = (uintptr_t) (blk + 2);
    const int *     p  = (const int *) (a & ~(uintptr_t) 3);
    const unsigned  sh = (unsigned) (a & 3);
    int x[9];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = p[i];
    }
    x[8] = p[sh ? 8 : 7];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        q[i] = (int) __builtin_amdgcn_alignbyte(x[i + 1], x[i], sh);
    }
    const int dw = p[sh ? 0 : -1];
    d = sh ? ((unsigned) dw & 0xFFFFu) : ((unsigned) dw >> 16);
}

// __syncthreads() on gfx906 (no back-off barrier) waits for all outstanding memory operations, global loads included;
// this barrier only completes the LDS operations, so a global prefetch stays in flight across it (as Tensile's gfx906
// kernels do)
#ifndef G3_ASM_BARRIER
#define G3_ASM_BARRIER 1
#endif
static __device__ __forceinline__ void g3_sync() {
#if G3_ASM_BARRIER
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}

// bytes 0..127 -> signed byte - off without borrows between bytes
static __device__ __forceinline__ int gkq_sub_bytes(const int v, const unsigned off4) {
    return (int) (((unsigned) (v | 0x80808080) - off4) ^ 0x80808080u);
}

typedef short gkq_short2 __attribute__((ext_vector_type(2)));
static __device__ __forceinline__ int gkq_sdot2(const int a, const int b, const int c) {
    return __builtin_amdgcn_sdot2(__builtin_bit_cast(gkq_short2, a), __builtin_bit_cast(gkq_short2, b), c, false);
}

// G3_MINB=3 (default): three blocks per CU (84 VGPRs), one buffer of LDS operands and the global loads issued at the
// step start; G3_MINB=2: 128 VGPRs, so the next round's LDS operands (G3_PIPE) and the next step's global data (G3_GPF)
// are loaded during the compute
#ifndef G3_PIPE
#define G3_PIPE (G3_MINB < 3)
#endif
#ifndef G3_GPF
#define G3_GPF (G3_MINB < 3 ? 1 : 0)
#endif

template <ggml_type T>
__launch_bounds__(GQ_NT, G3_MINB)
static __global__ void gcn_kq3_gemm(const char * __restrict__ w, const int8_t * __restrict__ yq, const int4 * __restrict__ ym,
        float * __restrict__ dst, const int M, const int N, const int K, const int64_t stride_row_w, const int64_t s1_dst,
        const int sb_per_split, const int64_t split_stride) {
    constexpr bool HAS_MIN = T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K;
    constexpr bool TWO_SC  = T == GGML_TYPE_Q6_K || T == GGML_TYPE_Q3_K; // a scale per 16 values
    constexpr int  NSW     = TWO_SC ? 2 : 1;
    __shared__ __align__(16) int aq[GQ_KB][G3_BM][G3_PADA];
    __shared__ __align__(16) int bq[GQ_KB][G3_BN][G3_PADB];
    __shared__ int    scw[G3_BM][NSW]; // the step's integer scales, byte kb (TWO_SC: byte 2*kb + half of word kb/2)
    __shared__ int2   mw2[G3_BM];      // q4_K/q5_K: the step's mins as int16 pairs (m0 | m1 << 16, m2 | m3 << 16)
    __shared__ float2 rowd[G3_BM];     // d, dmin of the row's super-block
    __shared__ float  tokd[G3_BN];     // activation scale of the step
    __shared__ int2   tokbs[G3_BN];    // activation sub-block sums of the step as int16 pairs

    const int row0 = (GQ_TOKENS_FAST ? blockIdx.y : blockIdx.x)*G3_BM;
    const int col0 = (GQ_TOKENS_FAST ? blockIdx.x : blockIdx.y)*G3_BN;
    const int nsb_all = K / QK_K;
    const int sb_beg  = blockIdx.z*sb_per_split;
    const int nsb     = min(sb_per_split, nsb_all - sb_beg);
    const int nstep   = 2*nsb;
    dst += blockIdx.z*split_stride;
    // block-uniform bases (SGPRs); the per-thread parts are 32-bit offsets rederived from the thread index where used,
    // so no per-thread address stays live across the compute loop (with 84 VGPRs they spilled)
    const char   * wbase = w  + (int64_t) row0*stride_row_w + (int64_t) sb_beg*gkq<T>::bytes;
    const int8_t * ybase = yq + (int64_t) col0*K + (int64_t) sb_beg*QK_K;
    const int4   * mbase = ym + (int64_t) col0*(K/128) + 2*sb_beg;
    const int last_row = M - 1 - row0;
    const int last_col = N - 1 - col0;
    const int wstride  = (int) stride_row_w;

    auto opaque_tid = [] {
        int t = threadIdx.x;
        asm volatile("" : "+v"(t)); // keep the derived indices from being hoisted out of the loop
        __builtin_assume(t >= 0 && t < GQ_NT);
        return t;
    };

    // raw global data of a step: A (weights: header + quants), B (32 int8 of a token's sub-block), meta
    struct raw_t { int4 a0, a1, a2, b0, b1, m; };
    auto fetch = [&](const int step) {
        raw_t R;
        int4 & ra0 = R.a0; int4 & ra1 = R.a1; int4 & ra2 = R.a2; int4 & rb0 = R.b0; int4 & rb1 = R.b1; int4 & rm = R.m;
        ra0 = ra1 = ra2 = make_int4(0, 0, 0, 0);
        const int tid  = opaque_tid();
        const int hs   = step % 2;
        const int la_u = tid % 4;
        const char * b = wbase + (step/2)*gkq<T>::bytes + (unsigned) __mul24(min(tid/4, last_row), wstride);
        const int ycol = min(tid/4, last_col);
        const int4 * pb = (const int4 *) (ybase + step*128 + (unsigned) (__mul24(ycol, K) + 32*(tid % 4)));
        rb0 = pb[0];
        rb1 = pb[1];
        rm  = mbase[step + __mul24(ycol, K/128)];
        if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K) {
            constexpr int qs_off = T == GGML_TYPE_Q5_K ? 48 : 16;
            ra0 = *(const int4 *) b;                                   // dm, scales[12]
            ra1 = *(const int4 *) (b + qs_off + 64*hs + 16*la_u);      // quarter u: pair 2*hs + u/2, values 16*(u%2)..
            if constexpr (T == GGML_TYPE_Q5_K) {
                ra2 = *(const int4 *) (b + 16 + 16*(la_u % 2));        // qh bytes 16*(u%2) .. +15
            }
        } else if constexpr (T == GGML_TYPE_IQ4_XS) {
            const int j = 4*hs + la_u;                                 // sub-block j: 16 bytes of qs
            const int2 hdr = *(const int2 *) b;                        // d | scales_h << 16, scales_l
            const int2 q01 = *(const int2 *) (b + 8 + 16*j);
            const int2 q23 = *(const int2 *) (b + 16 + 16*j);
            ra0 = make_int4(hdr.x, hdr.y, 0, 0);
            ra1 = make_int4(q01.x, q01.y, q23.x, q23.y);
        } else if constexpr (T == GGML_TYPE_Q6_K) {                    // quarter u: values 8u .. 8u+7 of 4 sub-blocks
            const int2 ql0 = gkq_ld2(b + 64*hs + 8*la_u);
            const int2 ql1 = gkq_ld2(b + 64*hs + 32 + 8*la_u);
            const int2 qh  = gkq_ld2(b + 128 + 32*hs + 8*la_u);
            const int2 sc  = gkq_ld2(b + 192 + 8*hs);                  // int8 scales of the step
            ra0 = make_int4(ql0.x, ql0.y, ql1.x, ql1.y);
            ra1 = make_int4(qh.x, qh.y, la_u % 2 ? sc.y : sc.x, *(const unsigned short *) (b + 208));
        } else {                                                       // q3_K: quarter u: values 8u .. 8u+7
            const int2 qs  = gkq_ld2(b + 32 + 32*hs + 8*la_u);
            const int2 hm  = gkq_ld2(b + 8*la_u);
            const int2 s01 = gkq_ld2(b + 96);
            ra0 = make_int4(qs.x, qs.y, hm.x, hm.y);
            ra1 = make_int4(la_u % 2 ? s01.y : s01.x, gkq_ld1(b + 104), *(const unsigned short *) (b + 108), 0);
        }
#ifdef G3_NO_GMEM // measurement only: loads replaced by register values (wrong results)
        ra0 = make_int4(0x01020304 ^ step, 0x3c003c00, tid, 7); ra1 = make_int4(tid ^ step, 5, 6, 7); ra2 = ra1;
        rb0 = make_int4(step, tid, 3, 4); rb1 = rb0; rm = make_int4(0x3c003c00, step, tid, 0);
#endif
        return R;
    };
    auto store = [&](const int step, const raw_t & R) {
        const int4 & ra0 = R.a0; const int4 & ra1 = R.a1; const int4 & ra2 = R.a2;
        const int4 & rb0 = R.b0; const int4 & rb1 = R.b1; const int4 & rm = R.m;
        const int tid    = opaque_tid();
        const int hs     = step % 2;
        const int la_row = tid / 4;
        const int la_u   = tid % 4;
        if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K) {
            // qs: byte l of pair p = value l of sub-block 2p (low nibble) and 2p+1 (high nibble)
            const int q[4] = {ra1.x, ra1.y, ra1.z, ra1.w};
            const int h[4] = {ra2.x, ra2.y, ra2.z, ra2.w};
            const int kbl = 2*(la_u / 2);
            const int ko  = 4*(la_u % 2);
            const int p   = 2*hs + la_u/2;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int lo = q[k] & 0x0F0F0F0F;
                int hi = (q[k] >> 4) & 0x0F0F0F0F;
                if constexpr (T == GGML_TYPE_Q5_K) {
                    lo |= ((h[k] >> (2*p))     & 0x01010101) << 4;
                    hi |= ((h[k] >> (2*p + 1)) & 0x01010101) << 4;
                }
                aq[kbl    ][la_row][ko + k] = lo;
                aq[kbl + 1][la_row][ko + k] = hi;
            }
            // get_scale_min_k4 of sub-block 4*hs + u
            const unsigned s0 = __builtin_amdgcn_ubfe((unsigned) ra0.y, 8*la_u, 8);
            const unsigned s1 = __builtin_amdgcn_ubfe((unsigned) ra0.z, 8*la_u, 8);
            const unsigned s2 = __builtin_amdgcn_ubfe((unsigned) ra0.w, 8*la_u, 8);
            const unsigned sc = hs ? ((s2 & 0xF) | ((s0 >> 6) << 4)) : (s0 & 63);
            const unsigned m  = hs ? ((s2 >>  4) | ((s1 >> 6) << 4)) : (s1 & 63);
            ((unsigned char  *) &scw[la_row][0])[la_u] = (unsigned char)  sc;
            ((unsigned short *) &mw2[la_row])[la_u]    = (unsigned short) m;
            const half2 dm = *(const half2 *) &ra0.x;
            rowd[la_row] = make_float2(__low2float(dm), __high2float(dm)); // the 4 quarters write the same value
        } else if constexpr (T == GGML_TYPE_IQ4_XS) {
            const int j = 4*hs + la_u;
            const int q[4] = {ra1.x, ra1.y, ra1.z, ra1.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int2 v = get_int_from_table_16(q[k], kvalues_iq4nl); // low nibbles: values 0..15, high: 16..31
                aq[la_u][la_row][k]     = v.x;
                aq[la_u][la_row][4 + k] = v.y;
            }
            const int ls = (int) ((__builtin_amdgcn_ubfe((unsigned) ra0.y, 8*(j/2), 8) >> (4*(j % 2))) & 0xF) |
                           (int) (__builtin_amdgcn_ubfe((unsigned) ra0.x, 16 + 2*j, 2) << 4);
            ((unsigned char *) &scw[la_row][0])[la_u] = (unsigned char) (ls - 32);
            rowd[la_row] = make_float2(__half2float(__ushort_as_half((unsigned short) (ra0.x & 0xFFFF))), 0.0f);
        } else if constexpr (T == GGML_TYPE_Q6_K) {
            const int a0[2] = {ra0.x, ra0.y}, a1[2] = {ra0.z, ra0.w}, hh[2] = {ra1.x, ra1.y};
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int v0 = ( a0[k]       & 0x0F0F0F0F) | (((hh[k]     ) & 0x03030303) << 4);
                const int v1 = ( a1[k]       & 0x0F0F0F0F) | (((hh[k] >> 2) & 0x03030303) << 4);
                const int v2 = ((a0[k] >> 4) & 0x0F0F0F0F) | (((hh[k] >> 4) & 0x03030303) << 4);
                const int v3 = ((a1[k] >> 4) & 0x0F0F0F0F) | (((hh[k] >> 6) & 0x03030303) << 4);
                aq[0][la_row][2*la_u + k] = gkq_sub_bytes(v0, 0x20202020u);
                aq[1][la_row][2*la_u + k] = gkq_sub_bytes(v1, 0x20202020u);
                aq[2][la_row][2*la_u + k] = gkq_sub_bytes(v2, 0x20202020u);
                aq[3][la_row][2*la_u + k] = gkq_sub_bytes(v3, 0x20202020u);
            }
            scw[la_row][la_u % 2] = ra1.z; // int8 scales, byte 2*kb + half (the row's quarters write each word twice)
            rowd[la_row] = make_float2(__half2float(__ushort_as_half((unsigned short) ra1.w)), 0.0f);
        } else { // q3_K
            const int qa[2] = {ra0.x, ra0.y}, hb[2] = {ra0.z, ra0.w};
#pragma unroll
            for (int k = 0; k < 2; ++k) {
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const int v = ((qa[k] >> (2*g)) & 0x03030303) | (((hb[k] >> (4*hs + g)) & 0x01010101) << 2); // 0..7
                    aq[g][la_row][2*la_u + k] = gkq_sub_bytes(v, 0x04040404u);
                }
            }
            const unsigned a   = ra1.x, tmp = ra1.y; // word la_u%2 of the step's 8 scales
            const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
            const unsigned wv  = ((hs ? a >> 4 : a) & km2) | (((tmp >> (4*hs + 2*(la_u % 2))) & km1) << 4);
            scw[la_row][la_u % 2] = gkq_sub_bytes((int) wv, 0x20202020u); // 6-bit scales - 32
            rowd[la_row] = make_float2(__half2float(__ushort_as_half((unsigned short) ra1.z)), 0.0f);
        }
        // B: token tid/4, sub-block tid%4
        *(int4 *) &bq[tid % 4][tid / 4][0] = rb0;
        *(int4 *) &bq[tid % 4][tid / 4][4] = rb1;
        tokd[tid / 4]  = __int_as_float(rm.x); // the 4 sub-block loaders of a token write the same values
        tokbs[tid / 4] = make_int2(rm.y, rm.z);
    };

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    raw_t nxt;
    if (G3_GPF) {
        nxt = fetch(0);
    }
    for (int step = 0; step < nstep; ++step) {
        if (G3_GPF) {
            store(step, nxt);
        } else {
            store(step, fetch(step));
        }
        g3_sync();
        if (G3_GPF == 1 && step + 1 < nstep) {
            nxt = fetch(step + 1);
        }

        const int tid = opaque_tid();
        const int rg  = tid % 16;
        const int cg  = tid / 16;
        int sw[4][NSW];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int u = 0; u < NSW; ++u) {
                sw[i][u] = scw[rg + 16*i][u];
            }
        }
        int isum[4][4];
        int s[4][4];
        // 16 rounds (sub-block kb = r/4, dword pair q = r%4) of 4 + 4 ds_read_b64 and 32 dp4a; G3_PIPE=1 loads round r+1
        // before the dp4a of round r (two buffers), 0 loads each round right before its dp4a (one buffer)
        int2 fa[2][4], fb[2][4];
        auto lds_load = [&](const int r, const int buf) {
            const int kb = r / 4, q = r % 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                fa[buf][i] = *(const int2 *) &aq[kb][rg + 16*i][2*q];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                fb[buf][j] = *(const int2 *) &bq[kb][cg + 16*j][2*q];
            }
        };
        // the step's float work, once per output: acc += d_act*(d*isum - dmin*sum_kb m*bsum)
        auto epilogue = [&] {
            float td[4];
            int2  bs[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                td[j] = tokd[cg + 16*j];
                if constexpr (HAS_MIN) {
                    bs[j] = tokbs[cg + 16*j];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int    r  = rg + 16*i;
                const float2 dm = rowd[r];
                int2 m2 = make_int2(0, 0);
                if constexpr (HAS_MIN) {
                    m2 = mw2[r];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float v = dm.x*(float) isum[i][j];
                    if constexpr (HAS_MIN) {
                        const int ms = gkq_sdot2(m2.x, bs[j].x, gkq_sdot2(m2.y, bs[j].y, 0));
                        v = fmaf(-dm.y, (float) ms, v);
                    }
                    acc[i][j] = fmaf(td[j], v, acc[i][j]);
                }
            }
        };
        if (G3_PIPE) {
            lds_load(0, 0);
        }
#pragma unroll
        for (int r = 0; r < 4*GQ_KB; ++r) {
            const int kb = r / 4, q = r % 4;
            const int buf = G3_PIPE ? r % 2 : 0;
            if (G3_PIPE) {
                if (r + 1 < 4*GQ_KB) {
                    lds_load(r + 1, (r + 1) % 2);
                }
            } else {
                lds_load(r, 0);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[i][j] = ggml_cuda_dp4a(fa[buf][i].x, fb[buf][j].x, q == 0 ? 0 : s[i][j]);
                    s[i][j] = ggml_cuda_dp4a(fa[buf][i].y, fb[buf][j].y, s[i][j]);
                }
            }
            if (q == 3 || (TWO_SC && q == 1)) {
                // TWO_SC: q == 1 closes values 0..15 of the sub-block, q == 3 values 16..31
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int sc;
                    if constexpr (TWO_SC) {
                        sc = __builtin_amdgcn_sbfe(sw[i][kb/2], 16*(kb % 2) + 8*(q/2), 8);
                    } else if constexpr (HAS_MIN) {
                        sc = __builtin_amdgcn_ubfe(sw[i][0], 8*kb, 8);
                    } else {
                        sc = __builtin_amdgcn_sbfe(sw[i][0], 8*kb, 8);
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        isum[i][j] = r == (TWO_SC ? 1 : 3) ? __mul24(s[i][j], sc) : isum[i][j] + __mul24(s[i][j], sc);
                        if constexpr (TWO_SC) {
                            s[i][j] = 0; // the next half / sub-block restarts
                        }
                    }
                }
            }
            if (r == 4*GQ_KB - 1) {
                epilogue();
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        if (G3_GPF == 2) {
            // the dp4a sums and LDS operands are dead: the next step's global loads fly during the last epilogue and
            // the barrier. Unconditional (the last step refetches itself): behind a branch the compiler sank the dp4a
            nxt = fetch(min(step + 1, nstep - 1));
        }
        g3_sync();
    }

    const int tid = threadIdx.x;
    const int rg = tid % 16;
    const int cg = tid / 16;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = col0 + cg + 16*j;
        if (col >= N) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = row0 + rg + 16*i;
            if (row < M) {
                dst[(int64_t) col*s1_dst + row] = acc[i][j];
            }
        }
    }
}

// ---- v4: exact per-32 activation scales in v3's 64 x 64 / 4 x 4 / three-blocks-per-CU structure ----
// v3's per-128 (or per-64) activation scale raised the KLD vs MMQ on Qwen3.8-27B from 0.009 (v1, per-32 = MMQ's q8_1
// precision) to 0.014 (0.0125); power-of-2 sub-scales did not fit the 84 VGPRs. v4 keeps per-32 activation scales:
// per sub-block and output acc += s*(d*sc*d_act) - dmin*m * d_act*sum(q_act) (TWO_SC: per 16-value half), the weight
// factors (d*sc, dmin*m) and token factors (d_act, d_act*sum) as float2 per sub-block in LDS. No int32 step sums, so
// 16 VGPRs fewer than v3. LDS: weight rows 8 dwords with the dword pair XOR-swizzled by (row >> 2) & 3 (the 16 rows a
// wave reads with ds_read_b64 then hit 32 distinct banks), so tiles + factors fit 20 KiB (three blocks per CU).
// Activation format: qs int8 [N][K]; dsf float2 [N][K/32] = (d, d*sum of the sub-block's quants).
static __global__ void gcn_q8x_quantize(const float * __restrict__ x, int8_t * __restrict__ qs, float2 * __restrict__ dsf,
        const int64_t K, const int64_t s_col, const int64_t s_batch = 0) {
    // 4 waves per block, a wave per 128 values, 2 values per lane, 16 lanes per 32-value sub-block; blockIdx.z: matrix
    // of a batch (gridDim.y columns each)
    x   += blockIdx.z*s_batch;
    qs  += (int64_t) blockIdx.z*gridDim.y*K;
    dsf += (int64_t) blockIdx.z*gridDim.y*(K/32);
    const int64_t col = blockIdx.y;
    const int wv = threadIdx.x >> 6, l = threadIdx.x & 63;
    const int64_t b = (int64_t) blockIdx.x*4 + wv;
    if (b*128 >= K) {
        return;
    }
    const float2 v = *(const float2 *) (x + col*s_col + b*128 + 2*l);
    float amax = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
    for (int off = 1; off < 16; off *= 2) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 64));
    }
    const float d  = amax / 127.0f;
    const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
    const int q0 = (int) roundf(v.x*id), q1 = (int) roundf(v.y*id);
    *(short *) (qs + col*K + b*128 + 2*l) = (short) ((q0 & 0xFF) | ((q1 & 0xFF) << 8));
    int s = q0 + q1;
#pragma unroll
    for (int off = 1; off < 16; off *= 2) {
        s += __shfl_xor(s, off, 64);
    }
    if ((l & 15) == 0) {
        dsf[col*(K/32) + b*4 + l/16] = make_float2(d, d*(float) s);
    }
}

// i-quant signs (7-bit index, the 8th sign the parity) -> per byte 0/1 (lsb) for values 0..3 (x) and 4..7 (y): the nibble
// times 0x00204081 puts bit k at bit 8k (the four shifted copies do not overlap). GKQ_IQ_SREG=1: the v4 tiles expand
// the signs in registers instead of an LDS table (less LDS: iq3_xxs then fits four blocks per CU)
#ifndef GKQ_IQ_SREG
#define GKQ_IQ_SREG 1
#endif
// GKQ_IQ2_PACK=1: the iq2_xs grid in LDS as 2-bit codes (every grid byte is 8, 25 or 43): 1 KiB instead of 4, so the
// tiles fit four blocks per CU; a row's 4 codes become byte selectors of v_perm over the bytes (8, 25, 43, 0)
#ifndef GKQ_IQ2_PACK
#define GKQ_IQ2_PACK 1
#endif
static __device__ __forceinline__ uint32_t gkq_iq2_unpack4(const uint32_t c8) {
    uint32_t x = c8 & 0xFF;                        // codes of 4 values, 2 bits each
    x = (x | (x << 12)) & 0x000F000Fu;             // values 0,1 -> bits 0..3, values 2,3 -> bits 16..19
    x = (x | (x <<  6)) & 0x03030303u;             // one code per byte
    return __builtin_amdgcn_perm(0u, 0x002B1908u, x);
}

static __device__ __forceinline__ uint2 gkq_iq_sign_bits(const uint32_t idx7) {
    const uint32_t s = idx7 | ((__popc(idx7) & 1) << 7);
    return make_uint2(((s & 0xF)*0x00204081u) & 0x01010101u, (((s >> 4) & 0xF)*0x00204081u) & 0x01010101u);
}

// dword d (0..7) of weight row r in the swizzled layout
static __device__ __forceinline__ int g4_swz(const int r, const int d) {
    return 2*((d >> 1) ^ ((r >> 2) & 3)) + (d & 1);
}

// MOE: MUL_MAT_ID grouped by expert (prefill: tens of tokens per expert). blockIdx.x = row tile, blockIdx.y = tile of
// moe_tiles (expert, first token of the expert's sorted slots); the tile's activation rows are moe_src[slot] of the
// quantized src1 (rows in natural order), its outputs go to dst row moe_dst[slot]; glu_epi: dst = SwiGLU-clamp(dst, result)
// (the pair's gate projection was written by the previous launch). No split-K, no batch.
#ifndef GKQ4_MOE32_MINB
#define GKQ4_MOE32_MINB 4
#endif
// BN: tokens per tile (MOE only: 32 for ~16-32 tokens per expert, the activation tile then loaded by the first two waves)
template <ggml_type T, bool MOE = false, int BN = G3_BN>
__launch_bounds__(GQ_NT, (MOE && BN == 32) ? GKQ4_MOE32_MINB : G3_MINB)
static __global__ void gcn_kq4_gemm(const char * __restrict__ w, const int8_t * __restrict__ yq, const float2 * __restrict__ ydf,
        float * __restrict__ dst, const int M, const int N, const int K, const int64_t stride_row_w, const int64_t s1_dst,
        const int sb_per_split, const int64_t split_stride, const int nsplit, const int64_t sw_batch, const int64_t sd_batch,
        const int2 * __restrict__ moe_tiles = nullptr, const int32_t * __restrict__ moe_n_tiles = nullptr,
        const int32_t * __restrict__ moe_bounds = nullptr, const int32_t * __restrict__ moe_src = nullptr,
        const int32_t * __restrict__ moe_dst = nullptr, const int64_t sw_expert = 0, const int glu_epi = 0,
        const float glu_limit = INFINITY) {
    int moe_p0 = 0, moe_ncols = 0;
    if constexpr (MOE) {
        if ((int) blockIdx.y >= *moe_n_tiles) {
            return;
        }
        const int2 te = moe_tiles[blockIdx.y];
        moe_p0    = moe_bounds[te.x] + te.y;
        moe_ncols = min(BN, moe_bounds[te.x + 1] - moe_p0);
        w += te.x*sw_expert;
    }
    constexpr bool HAS_MIN = T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K;
    // iq2_xs / iq3_xxs (GLM-5.3-Flash UD-Q2_K_XL experts): a sub-block's values are grid rows (8 / 4 magnitudes per
    // index) with 8-bit sign sets, decoded through LDS tables into the same int8 layout; iq2_xs has a scale per 16 values
    constexpr bool IS_IQ2XS  = T == GGML_TYPE_IQ2_XS;
    constexpr bool IS_IQ3XXS = T == GGML_TYPE_IQ3_XXS;
    // iq2_xxs (DeepSeek V4 Flash IQ2XXS experts): 4 grid rows of 8 magnitudes (8-bit indices) with 7-bit sign sets per
    // sub-block, one 4-bit scale per 32 values; the grid shares iq2_xs's magnitudes (2-bit codes in LDS)
    constexpr bool IS_IQ2XXS = T == GGML_TYPE_IQ2_XXS;
    constexpr bool TWO_SC  = T == GGML_TYPE_Q6_K || T == GGML_TYPE_Q3_K || IS_IQ2XS; // a scale per 16 values
    // q2_K: the 4-bit scales are multiplied into the LDS values (sc*q <= 45), so a sub-block is one integer dot product
    // times d; the 4-bit mins of the two halves go as an int16 pair against the activations' int16 half sums:
    // rowf = (half2 (d, dmin) bits, (m0, m1) int16 pair), colf = (d_act, (sum q_act of values 0..15, 16..31) int16 pair)
    constexpr bool IS_Q2K  = T == GGML_TYPE_Q2_K;
    __shared__ __align__(16) int    aq[GQ_KB][G3_BM][8];   // swizzled (g4_swz)
    static_assert(BN == G3_BN || (MOE && BN == 32), "token tile: 64, or 32 for MOE");
    constexpr int NJ = BN/16; // token columns per thread
    constexpr bool B_ALL = 4*BN == GQ_NT; // every thread loads activations (else the first 4*BN)
    __shared__ __align__(16) int    bq[GQ_KB][BN][8];
    __shared__ __align__(16) float2 rowf[GQ_KB][G3_BM];    // (d*sc, dmin*m); TWO_SC: (d*sc of values 0..15, of 16..31)
    __shared__ __align__(16) float2 colf[GQ_KB][BN];       // (d_act, d_act*sum)
    __shared__ uint2    iq_g2[IS_IQ2XS && !GKQ_IQ2_PACK ? 512 : 1];          // iq2xs_grid
    __shared__ uint16_t iq_g2p[(IS_IQ2XS && GKQ_IQ2_PACK) || IS_IQ2XXS ? 512 : 1]; // iq2xs_grid / iq2xxs_grid as 2-bit codes
    __shared__ uint32_t iq_g3[IS_IQ3XXS ? 256 : 1];         // iq3xxs_grid
    __shared__ uint2    iq_sg[(IS_IQ2XS || IS_IQ3XXS || IS_IQ2XXS) && !GKQ_IQ_SREG ? 128 : 1]; // ksigns_iq2xs as byte masks
    if constexpr (IS_IQ2XS || IS_IQ3XXS || IS_IQ2XXS) {
        if constexpr (IS_IQ2XXS) {
            for (int i = threadIdx.x; i < 256; i += GQ_NT) {
                const uint64_t g = iq2xxs_grid[i];
                uint32_t c = 0;
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const uint32_t byte = (uint32_t) (g >> (8*v)) & 0xFF;
                    c |= (byte == 0x08 ? 0u : byte == 0x19 ? 1u : 2u) << (2*v);
                }
                iq_g2p[i] = (uint16_t) c;
            }
        } else if constexpr (IS_IQ2XS) {
            for (int i = threadIdx.x; i < 512; i += GQ_NT) {
                const uint64_t g = iq2xs_grid[i];
                if (GKQ_IQ2_PACK) {
                    uint32_t c = 0;
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const uint32_t byte = (uint32_t) (g >> (8*v)) & 0xFF;
                        c |= (byte == 0x08 ? 0u : byte == 0x19 ? 1u : 2u) << (2*v);
                    }
                    iq_g2p[i] = (uint16_t) c;
                } else {
                    iq_g2[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
                }
            }
        } else {
            iq_g3[threadIdx.x] = iq3xxs_grid[threadIdx.x];
        }
        if (!GKQ_IQ_SREG && threadIdx.x < 128) {
            const uint32_t sv = ksigns_iq2xs[threadIdx.x];
            uint32_t lo = 0, hi = 0;
#pragma unroll
            for (int bb = 0; bb < 4; ++bb) {
                lo |= ((sv >> bb)       & 1u) ? (0xFFu << (8*bb)) : 0u;
                hi |= ((sv >> (bb + 4)) & 1u) ? (0xFFu << (8*bb)) : 0u;
            }
            iq_sg[threadIdx.x] = make_uint2(lo, hi);
        }
        __syncthreads();
    }

    const int row0 = MOE ? blockIdx.x*G3_BM : (GQ_TOKENS_FAST ? blockIdx.y : blockIdx.x)*G3_BM;
    const int col0 = MOE ? 0 : (GQ_TOKENS_FAST ? blockIdx.x : blockIdx.y)*G3_BN;
    // blockIdx.z = batch matrix * nsplit + K slice (split-K only for single matrices)
    const int zb = blockIdx.z / nsplit;
    const int zs = blockIdx.z % nsplit;
    w   += zb*sw_batch;
    yq  += (int64_t) zb*N*K;
    ydf += (int64_t) zb*N*(K/32);
    dst += zb*sd_batch;
    const int nsb_all = K / QK_K;
    const int sb_beg  = zs*sb_per_split;
    const int nsb     = min(sb_per_split, nsb_all - sb_beg);
    const int nstep   = 2*nsb;
    dst += zs*split_stride;
    const char   * wbase = w   + (int64_t) row0*stride_row_w + (int64_t) sb_beg*gkq<T>::bytes;
    const int8_t * ybase = yq  + (int64_t) col0*K + (int64_t) sb_beg*QK_K;
    const float2 * fbase = ydf + (int64_t) col0*(K/32) + (int64_t) sb_beg*(QK_K/32);
    const int last_row = M - 1 - row0;
    const int last_col = MOE ? moe_ncols - 1 : N - 1 - col0;
    const int wstride  = (int) stride_row_w;
    // MOE: the lane's activation row (token tid/4 of the tile, clamped) and its byte offsets, once
    unsigned moe_vy = 0, moe_vf = 0;
    if constexpr (MOE) {
        const unsigned srow = (unsigned) moe_src[moe_p0 + min((int) threadIdx.x/4, last_col)];
        moe_vy = srow*(unsigned) K + 32u*(threadIdx.x % 4);
        moe_vf = (srow*(unsigned) (K/32) + threadIdx.x % 4)*8u;
    }

    auto opaque_tid = [] {
        int t = threadIdx.x;
        asm volatile("" : "+v"(t));
        __builtin_assume(t >= 0 && t < GQ_NT);
        return t;
    };

    struct raw_t { int4 a0, a1, a2, b0, b1; float2 f; };
    auto fetch = [&](const int step) {
        raw_t R;
        R.a0 = R.a1 = R.a2 = make_int4(0, 0, 0, 0);
        const int tid  = opaque_tid();
        const int hs   = step % 2;
        const int la_u = tid % 4;
        // block-uniform byte bases (SGPRs) + one 32-bit per-lane offset each: global_load saddr form, no 64-bit VALU adds
        const char * sb   = wbase + (step/2)*gkq<T>::bytes;
        const unsigned vr = (unsigned) __mul24(min(tid/4, last_row), wstride);
        const char * b    = sb + vr;
        const int ycol    = min(tid/4, last_col);
        const unsigned vy = MOE ? moe_vy : (unsigned) (__mul24(ycol, K) + 32*(tid % 4));
        const unsigned vf = MOE ? moe_vf : (unsigned) (__mul24(ycol, K/32) + tid % 4)*8u;
        const int4 * pb   = (const int4 *) ((const char *) ybase + step*128 + vy);
        if (B_ALL || tid < 4*BN) {
            R.b0 = pb[0];
            R.b1 = pb[1];
            R.f  = *(const float2 *) ((const char *) fbase + step*32 + vf);
        }
        if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K) {
            constexpr int qs_off = T == GGML_TYPE_Q5_K ? 48 : 16;
            R.a0 = *(const int4 *) (sb + vr);
            R.a1 = *(const int4 *) (sb + qs_off + 64*hs + (vr + 16*la_u));
            if constexpr (T == GGML_TYPE_Q5_K) {
                R.a2 = *(const int4 *) (sb + 16 + (vr + 16*(la_u % 2)));
            }
        } else if constexpr (T == GGML_TYPE_IQ4_XS) {
            const int j = 4*hs + la_u;
            const int2 hdr = *(const int2 *) b;
            const int2 q01 = *(const int2 *) (b + 8 + 16*j);
            const int2 q23 = *(const int2 *) (b + 16 + 16*j);
            R.a0 = make_int4(hdr.x, hdr.y, 0, 0);
            R.a1 = make_int4(q01.x, q01.y, q23.x, q23.y);
        } else if constexpr (IS_IQ2XS) {
            // 74 bytes: d, qs[32] (9-bit grid index | 7-bit sign index << 9), scales[8]; sub-block ib = 4*hs + u
            const int  ib = 4*hs + la_u;
            const int2 q  = gkq_ld2(b + 2 + 8*ib);
            R.a0 = make_int4(q.x, q.y, 0, 0);
            R.a1 = make_int4((int) *(const unsigned short *) b, (int) *(const unsigned char *) (b + 2 + QK_K/4 + ib), 0, 0);
        } else if constexpr (IS_IQ2XXS) {
            // 66 bytes: d, then per sub-block 4 grid indices (bytes) and 4 x 7-bit sign sets | scale << 28
            const int  ib = 4*hs + la_u;
            const int2 q  = gkq_ld2(b + 2 + 8*ib);
            R.a0 = make_int4(q.x, q.y, 0, 0);
            R.a1 = make_int4((int) *(const unsigned short *) b, 0, 0, 0);
        } else if constexpr (IS_IQ3XXS) {
            // 98 bytes: d, qs[64] (8 grid indices per sub-block), then per sub-block 4 x 7-bit sign sets | scale << 28
            const int  ib = 4*hs + la_u;
            const int2 q  = gkq_ld2(b + 2 + 8*ib);
            R.a0 = make_int4(q.x, q.y, gkq_ld1(b + 2 + QK_K/4 + 4*ib), 0);
            R.a1 = make_int4((int) *(const unsigned short *) b, 0, 0, 0);
        } else if constexpr (T == GGML_TYPE_Q8_0) {
            // block 4*hs + u of the super-block (lane u's sub-block of the step)
            int q[8];
            unsigned d;
            gkq_ld_q8_0(b + sizeof(block_q8_0)*(4*hs + la_u), q, d);
            R.a0 = make_int4(q[0], q[1], q[2], q[3]);
            R.a1 = make_int4(q[4], q[5], q[6], q[7]);
            R.a2 = make_int4((int) d, 0, 0, 0);
        } else if constexpr (T == GGML_TYPE_Q6_K) {
            const int2 ql0 = gkq_ld2(b + 64*hs + 8*la_u);
            const int2 ql1 = gkq_ld2(b + 64*hs + 32 + 8*la_u);
            const int2 qh  = gkq_ld2(b + 128 + 32*hs + 8*la_u);
            const int  sc  = gkq_ld1(b + 192 + 8*hs + 2*la_u);     // scales of sub-block u's halves (bytes 0, 1)
            R.a0 = make_int4(ql0.x, ql0.y, ql1.x, ql1.y);
            R.a1 = make_int4(qh.x, qh.y, sc, *(const unsigned short *) (b + 208));
        } else if constexpr (T == GGML_TYPE_Q2_K) {
            const int2 qs = *(const int2 *) (b + 16 + 32*hs + 8*la_u); // quarter u of the 4 sub-blocks (2 bits each)
            const int2 sc = *(const int2 *) (b + 8*hs);                // the step's 8 scale bytes
            R.a0 = make_int4(qs.x, qs.y, sc.x, sc.y);
            R.a1 = make_int4(*(const int *) (b + 80), 0, 0, 0);       // d, dmin
        } else {
            const int2 qs  = gkq_ld2(b + 32 + 32*hs + 8*la_u);
            const int2 hm  = gkq_ld2(b + 8*la_u);
            const int2 s01 = gkq_ld2(b + 96);
            R.a0 = make_int4(qs.x, qs.y, hm.x, hm.y);
            R.a1 = make_int4(la_u / 2 ? s01.y : s01.x, gkq_ld1(b + 104), *(const unsigned short *) (b + 108), 0);
        }
        return R;
    };
    auto store = [&](const int step, const raw_t & R) {
        const int tid    = opaque_tid();
        const int hs     = step % 2;
        const int la_row = tid / 4;
        const int la_u   = tid % 4;
        int * arow[GQ_KB];
#pragma unroll
        for (int kb = 0; kb < GQ_KB; ++kb) {
            arow[kb] = &aq[kb][la_row][0];
        }
        float2 f; // factors of sub-block u of the step
        if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K) {
            const int q[4] = {R.a1.x, R.a1.y, R.a1.z, R.a1.w};
            const int h[4] = {R.a2.x, R.a2.y, R.a2.z, R.a2.w};
            const int kbl = 2*(la_u / 2);
            const int ko  = 4*(la_u % 2);
            const int p   = 2*hs + la_u/2;
            int lo[4], hi[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[k] = q[k] & 0x0F0F0F0F;
                hi[k] = (q[k] >> 4) & 0x0F0F0F0F;
                if constexpr (T == GGML_TYPE_Q5_K) {
                    lo[k] |= ((h[k] >> (2*p))     & 0x01010101) << 4;
                    hi[k] |= ((h[k] >> (2*p + 1)) & 0x01010101) << 4;
                }
            }
#pragma unroll
            for (int k = 0; k < 4; k += 2) {
                const int dd = g4_swz(la_row, ko + k);
                *(int2 *) &arow[kbl][dd]     = make_int2(lo[k], lo[k + 1]);
                *(int2 *) &arow[kbl + 1][dd] = make_int2(hi[k], hi[k + 1]);
            }
            const unsigned s0 = __builtin_amdgcn_ubfe((unsigned) R.a0.y, 8*la_u, 8);
            const unsigned s1 = __builtin_amdgcn_ubfe((unsigned) R.a0.z, 8*la_u, 8);
            const unsigned s2 = __builtin_amdgcn_ubfe((unsigned) R.a0.w, 8*la_u, 8);
            const unsigned sc = hs ? ((s2 & 0xF) | ((s0 >> 6) << 4)) : (s0 & 63);
            const unsigned m  = hs ? ((s2 >>  4) | ((s1 >> 6) << 4)) : (s1 & 63);
            const half2 dm = *(const half2 *) &R.a0.x;
            f = make_float2(__low2float(dm)*(float) sc, __high2float(dm)*(float) m);
        } else if constexpr (T == GGML_TYPE_IQ4_XS) {
            const int j = 4*hs + la_u;
            const int q[4] = {R.a1.x, R.a1.y, R.a1.z, R.a1.w};
            int lo[4], hi[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int2 v = get_int_from_table_16(q[k], kvalues_iq4nl);
                lo[k] = v.x;
                hi[k] = v.y;
            }
#pragma unroll
            for (int k = 0; k < 4; k += 2) {
                *(int2 *) &arow[la_u][g4_swz(la_row, k)]     = make_int2(lo[k], lo[k + 1]);
                *(int2 *) &arow[la_u][g4_swz(la_row, 4 + k)] = make_int2(hi[k], hi[k + 1]);
            }
            const int ls = (int) ((__builtin_amdgcn_ubfe((unsigned) R.a0.y, 8*(j/2), 8) >> (4*(j % 2))) & 0xF) |
                           (int) (__builtin_amdgcn_ubfe((unsigned) R.a0.x, 16 + 2*j, 2) << 4);
            f = make_float2(__half2float(__ushort_as_half((unsigned short) (R.a0.x & 0xFFFF)))*(float) (ls - 32), 0.0f);
        } else if constexpr (IS_IQ2XS) {
            // signs applied bytewise: (g ^ m) + (m & 1) negates where m = 0xFF (every grid byte is > 0: no carries)
            const uint32_t qq[4] = {(uint32_t) R.a0.x & 0xFFFF, (uint32_t) R.a0.x >> 16, (uint32_t) R.a0.y & 0xFFFF,
                                    (uint32_t) R.a0.y >> 16};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint2 g;
                if (GKQ_IQ2_PACK) {
                    const uint32_t c = iq_g2p[qq[k] & 0x1FF];
                    g = make_uint2(gkq_iq2_unpack4(c), gkq_iq2_unpack4(c >> 8));
                } else {
                    g = iq_g2[qq[k] & 0x1FF];
                }
                uint2 m, m1;
                if (GKQ_IQ_SREG) {
                    m1 = gkq_iq_sign_bits(qq[k] >> 9);
                    m  = make_uint2(m1.x*0xFFu, m1.y*0xFFu);
                } else {
                    m  = iq_sg[qq[k] >> 9];
                    m1 = make_uint2(m.x & 0x01010101u, m.y & 0x01010101u);
                }
                *(int2 *) &arow[la_u][g4_swz(la_row, 2*k)] = make_int2((int) ((g.x ^ m.x) + m1.x), (int) ((g.y ^ m.y) + m1.y));
            }
            const float d  = __half2float(__ushort_as_half((unsigned short) R.a1.x));
            const int   sc = R.a1.y;
            f = make_float2(d*0.125f*(float) (2*(sc & 0xF) + 1), d*0.125f*(float) (2*(sc >> 4) + 1));
        } else if constexpr (IS_IQ2XXS) {
            const uint32_t gi  = (uint32_t) R.a0.x; // 4 grid indices
            const uint32_t aux = (uint32_t) R.a0.y; // 4 sign sets | scale << 28
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t c = iq_g2p[(gi >> (8*k)) & 0xFF];
                const uint2 g = make_uint2(gkq_iq2_unpack4(c), gkq_iq2_unpack4(c >> 8));
                uint2 m, m1;
                if (GKQ_IQ_SREG) {
                    m1 = gkq_iq_sign_bits((aux >> (7*k)) & 0x7F);
                    m  = make_uint2(m1.x*0xFFu, m1.y*0xFFu);
                } else {
                    m  = iq_sg[(aux >> (7*k)) & 0x7F];
                    m1 = make_uint2(m.x & 0x01010101u, m.y & 0x01010101u);
                }
                *(int2 *) &arow[la_u][g4_swz(la_row, 2*k)] = make_int2((int) ((g.x ^ m.x) + m1.x), (int) ((g.y ^ m.y) + m1.y));
            }
            const float d = __half2float(__ushort_as_half((unsigned short) R.a1.x));
            f = make_float2(d*0.125f*(float) (2*(int) (aux >> 28) + 1), 0.0f);
        } else if constexpr (IS_IQ3XXS) {
            const uint32_t ii[2] = {(uint32_t) R.a0.x, (uint32_t) R.a0.y};
            const uint32_t aux   = (uint32_t) R.a0.z;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t w  = ii[k/2] >> (16*(k % 2));
                const uint32_t gl = iq_g3[w & 0xFF];
                const uint32_t gh = iq_g3[(w >> 8) & 0xFF];
                uint2 m, m1;
                if (GKQ_IQ_SREG) {
                    m1 = gkq_iq_sign_bits((aux >> (7*k)) & 0x7F);
                    m  = make_uint2(m1.x*0xFFu, m1.y*0xFFu);
                } else {
                    m  = iq_sg[(aux >> (7*k)) & 0x7F];
                    m1 = make_uint2(m.x & 0x01010101u, m.y & 0x01010101u);
                }
                *(int2 *) &arow[la_u][g4_swz(la_row, 2*k)] = make_int2((int) ((gl ^ m.x) + m1.x), (int) ((gh ^ m.y) + m1.y));
            }
            const float d = __half2float(__ushort_as_half((unsigned short) R.a1.x));
            f = make_float2(d*0.25f*(float) (2*(int) (aux >> 28) + 1), 0.0f);
        } else if constexpr (T == GGML_TYPE_Q8_0) {
            // the int8 quants as they are, in value order (dword k = values 4k .. 4k + 3)
            *(int2 *) &arow[la_u][g4_swz(la_row, 0)] = make_int2(R.a0.x, R.a0.y);
            *(int2 *) &arow[la_u][g4_swz(la_row, 2)] = make_int2(R.a0.z, R.a0.w);
            *(int2 *) &arow[la_u][g4_swz(la_row, 4)] = make_int2(R.a1.x, R.a1.y);
            *(int2 *) &arow[la_u][g4_swz(la_row, 6)] = make_int2(R.a1.z, R.a1.w);
            f = make_float2(__half2float(__ushort_as_half((unsigned short) R.a2.x)), 0.0f);
        } else if constexpr (T == GGML_TYPE_Q6_K) {
            const int a0[2] = {R.a0.x, R.a0.y}, a1[2] = {R.a0.z, R.a0.w}, hh[2] = {R.a1.x, R.a1.y};
            int v[4][2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                v[0][k] = gkq_sub_bytes(( a0[k]       & 0x0F0F0F0F) | (((hh[k]     ) & 0x03030303) << 4), 0x20202020u);
                v[1][k] = gkq_sub_bytes(( a1[k]       & 0x0F0F0F0F) | (((hh[k] >> 2) & 0x03030303) << 4), 0x20202020u);
                v[2][k] = gkq_sub_bytes(((a0[k] >> 4) & 0x0F0F0F0F) | (((hh[k] >> 4) & 0x03030303) << 4), 0x20202020u);
                v[3][k] = gkq_sub_bytes(((a1[k] >> 4) & 0x0F0F0F0F) | (((hh[k] >> 6) & 0x03030303) << 4), 0x20202020u);
            }
            const int dd = g4_swz(la_row, 2*la_u);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                *(int2 *) &arow[g][dd] = make_int2(v[g][0], v[g][1]);
            }
            const float d = __half2float(__ushort_as_half((unsigned short) R.a1.w));
            f = make_float2(d*(float) (int8_t) (R.a1.z & 0xFF), d*(float) (int8_t) ((R.a1.z >> 8) & 0xFF));
        } else if constexpr (T == GGML_TYPE_Q2_K) {
            const int qa[2] = {R.a0.x, R.a0.y};
            const int hh = la_u / 2; // the 16-value half of every sub-block that quarter u lies in
            int v[4][2];
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                // scale byte 2g + hh (word g/2): its low nibble
                const unsigned sc = __builtin_amdgcn_ubfe((unsigned) (g < 2 ? R.a0.z : R.a0.w), 16*(g % 2) + 8*hh, 4);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    v[g][k] = (int) ((((unsigned) qa[k] >> (2*g)) & 0x03030303u)*sc);
                }
            }
            const int dd = g4_swz(la_row, 2*la_u);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                *(int2 *) &arow[g][dd] = make_int2(v[g][0], v[g][1]);
            }
            // sub-block u's mins: high nibbles of scale bytes 2u, 2u + 1
            const unsigned sw = (unsigned) (la_u < 2 ? R.a0.z : R.a0.w);
            const int m0 = (int) __builtin_amdgcn_ubfe(sw, 16*(la_u % 2) + 4,  4);
            const int m1 = (int) __builtin_amdgcn_ubfe(sw, 16*(la_u % 2) + 12, 4);
            f = make_float2(__int_as_float(R.a1.x), __int_as_float(m0 | (m1 << 16)));
        } else { // q3_K
            const int qa[2] = {R.a0.x, R.a0.y}, hb[2] = {R.a0.z, R.a0.w};
            int v[4][2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    v[g][k] = gkq_sub_bytes(((qa[k] >> (2*g)) & 0x03030303) | (((hb[k] >> (4*hs + g)) & 0x01010101) << 2),
                        0x04040404u);
                }
            }
            const int dd = g4_swz(la_row, 2*la_u);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                *(int2 *) &arow[g][dd] = make_int2(v[g][0], v[g][1]);
            }
            // the 6-bit scales 8*hs + 2*u, +1 (the halves of sub-block u), minus 32
            const unsigned a   = R.a1.x, tmp = R.a1.y; // word u/2 of the step's 8 scales
            const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
            const unsigned wv  = ((hs ? a >> 4 : a) & km2) | (((tmp >> (4*hs + 2*(la_u / 2))) & km1) << 4);
            const int sc0 = (int) ((wv >> (16*(la_u % 2)))     & 0xFF) - 32;
            const int sc1 = (int) ((wv >> (16*(la_u % 2) + 8)) & 0xFF) - 32;
            const float d = __half2float(__ushort_as_half((unsigned short) R.a1.z));
            f = make_float2(d*(float) sc0, d*(float) sc1);
        }
        rowf[la_u][la_row] = f;
        if (!B_ALL && tid >= 4*BN) {
            return;
        }
        *(int4 *) &bq[tid % 4][tid / 4][0] = R.b0;
        *(int4 *) &bq[tid % 4][tid / 4][4] = R.b1;
        float2 cfv = R.f;
        if constexpr (IS_Q2K) {
            // the integer sums of the sub-block's two 16-value halves (|sum| <= 2032) as an int16 pair
            int s0 = 0, s1 = 0;
            s0 = ggml_cuda_dp4a(R.b0.x, 0x01010101, s0); s0 = ggml_cuda_dp4a(R.b0.y, 0x01010101, s0);
            s0 = ggml_cuda_dp4a(R.b0.z, 0x01010101, s0); s0 = ggml_cuda_dp4a(R.b0.w, 0x01010101, s0);
            s1 = ggml_cuda_dp4a(R.b1.x, 0x01010101, s1); s1 = ggml_cuda_dp4a(R.b1.y, 0x01010101, s1);
            s1 = ggml_cuda_dp4a(R.b1.z, 0x01010101, s1); s1 = ggml_cuda_dp4a(R.b1.w, 0x01010101, s1);
            cfv.y = __int_as_float((s0 & 0xFFFF) | (s1 << 16));
        }
        colf[tid % 4][tid / 4] = cfv;
    };

    float acc[4][NJ];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int step = 0; step < nstep; ++step) {
        store(step, fetch(step));
        g3_sync();

        const int tid = opaque_tid();
        const int rg  = tid % 16;
        const int cg  = tid / 16;
        const int sw8 = 8*((rg >> 2) & 3); // byte offset of the swizzle of the thread's rows (the same for rg + 16*i)
        int s[4][NJ];
        // 16 rounds (sub-block kb = r/4, dword pair q = r%4) of 4 + 4 ds_read_b64 and 32 dp4a; G4_PIPE=1 loads round
        // r+1 before the dp4a of round r (two buffers)
#ifndef G4_PIPE
#define G4_PIPE 1
#endif
        int2 fa[2][4], fb[2][NJ];
        auto lds_load = [&](const int r, const int buf) {
            const int kb = r / 4, q = r % 4;
            const char * ab = (const char *) &aq[kb][rg][0] + ((8*q) ^ sw8);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                fa[buf][i] = *(const int2 *) (ab + 16*i*32);
            }
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                fb[buf][j] = *(const int2 *) &bq[kb][cg + 16*j][2*q];
            }
        };
        if (G4_PIPE) {
            lds_load(0, 0);
        }
#pragma unroll
        for (int r = 0; r < 4*GQ_KB; ++r) {
            const int kb = r / 4, q = r % 4;
            const int buf = G4_PIPE ? r % 2 : 0;
            if (G4_PIPE) {
                if (r + 1 < 4*GQ_KB) {
                    lds_load(r + 1, (r + 1) % 2);
                }
            } else {
                lds_load(r, 0);
            }
            const int2 * a = fa[buf];
            const int2 * bb = fb[buf];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    s[i][j] = ggml_cuda_dp4a(a[i].x, bb[j].x, (q == 0 || (TWO_SC && q == 2)) ? 0 : s[i][j]);
                    s[i][j] = ggml_cuda_dp4a(a[i].y, bb[j].y, s[i][j]);
                }
            }
            if (q == 3 || (TWO_SC && q == 1)) {
                // the sub-block's (TWO_SC: half's) float work
                float2 rf[4], cf[NJ];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    rf[i] = rowf[kb][rg + 16*i];
                }
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    cf[j] = colf[kb][cg + 16*j];
                }
                if constexpr (IS_Q2K) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const half2 dmh = __builtin_bit_cast(half2, __float_as_int(rf[i].x));
                        const float d_w = __low2float(dmh), m_w = __high2float(dmh);
                        const int   mp  = __float_as_int(rf[i].y);
#pragma unroll
                        for (int j = 0; j < NJ; ++j) {
                            const int mi = __builtin_amdgcn_sdot2(__builtin_bit_cast(gkq_short2, mp),
                                __builtin_bit_cast(gkq_short2, __float_as_int(cf[j].y)), 0, false);
                            const float t = fmaf(d_w, (float) s[i][j], -m_w*(float) mi);
                            acc[i][j] = fmaf(cf[j].x, t, acc[i][j]);
                        }
                    }
                } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float fa = TWO_SC ? (q == 1 ? rf[i].x : rf[i].y) : rf[i].x;
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
                        acc[i][j] = fmaf((float) s[i][j], fa*cf[j].x, acc[i][j]);
                        if constexpr (HAS_MIN) {
                            acc[i][j] = fmaf(-rf[i].y, cf[j].y, acc[i][j]);
                        }
                    }
                }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        g3_sync();
    }

    const int tid = threadIdx.x;
    const int rg = tid % 16;
    const int cg = tid / 16;
    if constexpr (MOE) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int c = cg + 16*j;
            if (c >= moe_ncols) {
                continue;
            }
            float * o = dst + (int64_t) moe_dst[moe_p0 + c]*s1_dst;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = row0 + rg + 16*i;
                if (row < M) {
                    if (glu_epi) {
                        // silu(min(g, limit)) * clamp(u, -limit, limit) (INFINITY: plain SwiGLU)
                        const float g = fminf(o[row], glu_limit);
                        const float u = fmaxf(fminf(acc[i][j], glu_limit), -glu_limit);
                        o[row] = g/(1.0f + expf(-g))*u;
                    } else {
                        o[row] = acc[i][j];
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int col = col0 + cg + 16*j;
        if (col >= N) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = row0 + rg + 16*i;
            if (row < M) {
                dst[(int64_t) col*s1_dst + row] = acc[i][j];
            }
        }
    }
}

// GGML_CUDA_GCN_KQ: 1 (default) all of q4_K, q5_K, q6_K, iq4_xs, iq4_nl, q3_K, q2_K (v4 only); 0 off (MMQ); or a bitmask 1 q4_K,
// 2 q5_K, 4 q6_K, 8 iq4_xs, 16 iq4_nl, 32 q3_K, 64 q2_K when > 1 (e.g. 2 = q5_K only)
bool ggml_cuda_gcn_kq_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const int env = [] { const char * e = getenv("GGML_CUDA_GCN_KQ"); return e ? atoi(e) : 1; }();
    static const int min_rows = [] { const char * e = getenv("GGML_CUDA_GCN_GEMM_MIN_M"); return e ? atoi(e) : 128; }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!env || !GGML_CUDA_CC_IS_GCN(cc) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int mask = env == 1 ? 255 : env;
    int bit = 0;
    switch (src0->type) {
        case GGML_TYPE_Q4_K:   bit = 1; break;
        case GGML_TYPE_Q5_K:   bit = 2; break;
        case GGML_TYPE_Q6_K:   bit = 4; break;
        case GGML_TYPE_IQ4_XS: bit = 8; break;
        case GGML_TYPE_IQ4_NL: bit = 16; break;
        case GGML_TYPE_Q3_K:   bit = 32; break;
        case GGML_TYPE_Q2_K:   bit = 64; break;
        case GGML_TYPE_Q8_0:   bit = 128; break;
        default: return false;
    }
    if (!(mask & bit)) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    // a batch of matrices (e.g. DeepSeek V4's grouped output projection: 8 x (1024 x 4096)) runs as blockIdx.z of v4,
    // without broadcasting
    const int64_t nbatch = src0->ne[2];
    if (M < min_rows || K % QK_K != 0 || N < 32 || src0->ne[3] != 1 || src1->ne[2] != nbatch || src1->ne[3] != 1 ||
            dst->ne[2] != nbatch || dst->nb[0] != sizeof(float) || !ggml_is_contiguous(src0) || src1->nb[0] != sizeof(float) ||
            src0->nb[1] % 4 != 0 || ((uintptr_t) src0->data) % 4 != 0 || (nbatch > 1 && src1->nb[2] % 8 != 0)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    // GGML_CUDA_GCN_KQV: 4 (default) v4: exact per-32 activation scales, 64 x 64 tiles, three blocks per CU; 3: v3,
    // activation scales per 128 values (~12% faster GEMMs, but KLD vs MMQ 0.014 instead of 0.009 on Qwen3.8-27B);
    // 1: v1 (float scales per 32, 128 x 64 tiles). iq4_nl (a float scale per 32 values) always runs v1, and v4 falls
    // back to v1 for iq4_xs at <= 128 tokens (v1's 8 x 4 register tile was 7-10% faster there, equal at 512-2048)
    static const int kqv_env = [] { const char * e = getenv("GGML_CUDA_GCN_KQV"); return e ? atoi(e) : 4; }();
    const int  wal  = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K ? 16 : src0->type == GGML_TYPE_IQ4_XS ? 8 : 4;
    const bool v34  = src0->type != GGML_TYPE_IQ4_NL && src1->nb[1] % 8 == 0 && ((uintptr_t) src1->data) % 8 == 0 &&
        src0->nb[1] % wal == 0 && ((uintptr_t) src0->data) % wal == 0;
    const bool v4 = v34 && kqv_env == 4 && !(src0->type == GGML_TYPE_IQ4_XS && N <= 128);
    const bool v3 = v34 && kqv_env == 3;
    if ((src0->type == GGML_TYPE_Q2_K || src0->type == GGML_TYPE_Q8_0 || nbatch > 1) && !v4) {
        return false; // q2_K, q8_0 and batches: v4 only
    }
    const int  BM  = v3 || v4 ? G3_BM : GQ_BM;
    const int  BN  = v3 || v4 ? G3_BN : GQ_BN;
    const int  bpc = v3 || v4 ? G3_MINB : 2; // resident blocks per CU
    ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool());
    ggml_cuda_pool_alloc<int8_t>     yq(ctx.pool());
    ggml_cuda_pool_alloc<int4>       ym(ctx.pool());
    ggml_cuda_pool_alloc<float2>     yf(ctx.pool());
    if (v4) {
        yq.alloc(nbatch*N*K);
        yf.alloc(nbatch*N*(K/32));
        gcn_q8x_quantize<<<dim3((K/128 + 3)/4, N, nbatch), 256, 0, stream>>>((const float *) src1->data, yq.get(), yf.get(), K,
            src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float));
    } else if (v3) {
        yq.alloc(N*K);
        ym.alloc(N*(K/128));
        gcn_q8s_quantize<<<dim3((K/128 + 3)/4, N), 256, 0, stream>>>((const float *) src1->data, yq.get(), ym.get(), K,
            src1->nb[1]/sizeof(float));
    } else {
        y.alloc(N*(K/QK8_1));
        if (src1->nb[1] % 16 == 0 && ((uintptr_t) src1->data) % 16 == 0) {
            gcn_q8_quantize4<<<dim3((K + 1023)/1024, N), 256, 0, stream>>>((const float *) src1->data, y.get(), K, src1->nb[1]/sizeof(float));
        } else {
            gcn_q8_quantize<<<dim3((K + 255)/256, N), 256, 0, stream>>>((const float *) src1->data, y.get(), K, src1->nb[1]/sizeof(float));
        }
    }
    // split K (in whole super-blocks) when the tiles alone cannot fill the GPU: the split count minimizing rounds of
    // resident blocks (2 per CU) x super-blocks per block. 8704x5120 at 32/64 tokens (68 tiles): no split 568/573 us ->
    // 3-4 splits 405/420 (q5_K); prefill tile counts (>= 2 rounds, 544 tiles at 512 tokens) keep 1
    const int nsm   = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int tiles = (int) (((M + BM - 1)/BM) * ((N + BN - 1)/BN));
    const int nsb   = (int) (K / QK_K);
    int nsplit = 1;
    if (nbatch == 1 && tiles < 2*bpc*nsm) { // fewer than 2 rounds: each extra slice also costs a partial write and the reduction
        int64_t best = INT64_MAX;
        for (int ns = 1; ns <= 8 && ns <= nsb; ++ns) {
            const int64_t cost = (int64_t) ((tiles*ns + bpc*nsm - 1)/(bpc*nsm)) * ((nsb + ns - 1)/ns) + (ns > 1 ? ns : 0);
            if (cost < best) {
                best = cost;
                nsplit = ns;
            }
        }
    }
    static const int split_env = [] { const char * e = getenv("GGML_CUDA_GCN_GEMM_SPLIT"); return e ? atoi(e) : 0; }();
    if (split_env > 0 && nbatch == 1) {
        nsplit = split_env;
    }
    const int sb_per_split = (nsb + nsplit - 1) / nsplit;
    nsplit = (nsb + sb_per_split - 1) / sb_per_split;
    const dim3 grid(GQ_TOKENS_FAST ? (N + BN - 1)/BN : (M + BM - 1)/BM,
                    GQ_TOKENS_FAST ? (M + BM - 1)/BM : (N + BN - 1)/BN, nsplit*nbatch);
    float * out = (float *) dst->data;
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    int64_t s1 = dst->nb[1]/sizeof(float), split_stride = 0;
    if (nsplit > 1) {
        part.alloc(nsplit*M*N);
        out = part.get();
        s1 = M;
        split_stride = M*N;
    }
    const char * w = (const char *) src0->data;
    if (v4) {
#define GKQ4_LAUNCH(TT) gcn_kq4_gemm<TT><<<grid, GQ_NT, 0, stream>>>(w, yq.get(), yf.get(), out, (int) M, (int) N, (int) K, \
            src0->nb[1], s1, sb_per_split, split_stride, nsplit, src0->nb[2], dst->nb[2]/sizeof(float))
        switch (src0->type) {
            case GGML_TYPE_Q4_K:   GKQ4_LAUNCH(GGML_TYPE_Q4_K);   break;
            case GGML_TYPE_Q5_K:   GKQ4_LAUNCH(GGML_TYPE_Q5_K);   break;
            case GGML_TYPE_Q6_K:   GKQ4_LAUNCH(GGML_TYPE_Q6_K);   break;
            case GGML_TYPE_Q3_K:   GKQ4_LAUNCH(GGML_TYPE_Q3_K);   break;
            case GGML_TYPE_Q2_K:   GKQ4_LAUNCH(GGML_TYPE_Q2_K);   break;
            case GGML_TYPE_Q8_0:   GKQ4_LAUNCH(GGML_TYPE_Q8_0);   break;
            default:               GKQ4_LAUNCH(GGML_TYPE_IQ4_XS); break;
        }
#undef GKQ4_LAUNCH
    } else if (v3) {
#define GKQ3_LAUNCH(TT) gcn_kq3_gemm<TT><<<grid, GQ_NT, 0, stream>>>(w, yq.get(), ym.get(), out, (int) M, (int) N, (int) K, \
            src0->nb[1], s1, sb_per_split, split_stride)
        switch (src0->type) {
            case GGML_TYPE_Q4_K:   GKQ3_LAUNCH(GGML_TYPE_Q4_K);   break;
            case GGML_TYPE_Q5_K:   GKQ3_LAUNCH(GGML_TYPE_Q5_K);   break;
            case GGML_TYPE_Q6_K:   GKQ3_LAUNCH(GGML_TYPE_Q6_K);   break;
            case GGML_TYPE_Q3_K:   GKQ3_LAUNCH(GGML_TYPE_Q3_K);   break;
            default:               GKQ3_LAUNCH(GGML_TYPE_IQ4_XS); break;
        }
#undef GKQ3_LAUNCH
    } else switch (src0->type) {
        case GGML_TYPE_Q4_K:
            gcn_kq_gemm<GGML_TYPE_Q4_K><<<grid, GQ_NT, 0, stream>>>(w, y.get(), out, (int) M, (int) N, (int) K, src0->nb[1], s1, sb_per_split, split_stride);
            break;
        case GGML_TYPE_Q5_K:
            gcn_kq_gemm<GGML_TYPE_Q5_K><<<grid, GQ_NT, 0, stream>>>(w, y.get(), out, (int) M, (int) N, (int) K, src0->nb[1], s1, sb_per_split, split_stride);
            break;
        case GGML_TYPE_Q6_K:
            gcn_kq_gemm<GGML_TYPE_Q6_K><<<grid, GQ_NT, 0, stream>>>(w, y.get(), out, (int) M, (int) N, (int) K, src0->nb[1], s1, sb_per_split, split_stride);
            break;
        case GGML_TYPE_IQ4_NL:
            gcn_kq_gemm<GGML_TYPE_IQ4_NL><<<grid, GQ_NT, 0, stream>>>(w, y.get(), out, (int) M, (int) N, (int) K, src0->nb[1], s1, sb_per_split, split_stride);
            break;
        case GGML_TYPE_Q3_K:
            gcn_kq_gemm<GGML_TYPE_Q3_K><<<grid, GQ_NT, 0, stream>>>(w, y.get(), out, (int) M, (int) N, (int) K, src0->nb[1], s1, sb_per_split, split_stride);
            break;
        default:
            gcn_kq_gemm<GGML_TYPE_IQ4_XS><<<grid, GQ_NT, 0, stream>>>(w, y.get(), out, (int) M, (int) N, (int) K, src0->nb[1], s1, sb_per_split, split_stride);
            break;
    }
    if (nsplit > 1) {
        gcn_q8_splitk_reduce<<<(M*N + 255)/256, 256, 0, stream>>>(part.get(), (float *) dst->data, nsplit, M, N, dst->nb[1]/sizeof(float));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_gcn_q8_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // default on GCN (4x MI50 pp2048: 1750 -> 1988 t/s); GGML_CUDA_GCN_GEMM=0 falls back to MMQ
    static const int env = [] { const char * e = getenv("GGML_CUDA_GCN_GEMM"); return e ? atoi(e) : 1; }();
    static const int min_rows = [] { const char * e = getenv("GGML_CUDA_GCN_GEMM_MIN_M"); return e ? atoi(e) : 128; }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!env || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (M < min_rows || K % QK8_0 != 0 || N < 32 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src0) || src1->nb[0] != sizeof(float) ||
            src0->nb[1] % 4 != 0 || ((uintptr_t) src0->data) % 4 != 0) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool(), N*(K/QK8_1));
    if (src1->nb[1] % 16 == 0 && ((uintptr_t) src1->data) % 16 == 0) {
        gcn_q8_quantize4<<<dim3((K + 1023)/1024, N), 256, 0, stream>>>((const float *) src1->data, y.get(), K, src1->nb[1]/sizeof(float));
    } else {
        gcn_q8_quantize<<<dim3((K + 255)/256, N), 256, 0, stream>>>((const float *) src1->data, y.get(), K, src1->nb[1]/sizeof(float));
    }
    // split K when the tiles alone cannot fill the GPU (few rows): up to 8 slices of >= 8 blocks each
    const int nsm   = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int tiles = (int) (((M + GQ_BM - 1)/GQ_BM) * ((N + GQ_BN - 1)/GQ_BN));
    const int nkb   = (int) (K / QK8_0);
    int nsplit = 1;
    while (nsplit < 8 && tiles*nsplit*2 <= 2*nsm && nkb/(nsplit*2) >= 8) {
        nsplit *= 2;
    }
    static const int split_env = [] { const char * e = getenv("GGML_CUDA_GCN_GEMM_SPLIT"); return e ? atoi(e) : 0; }();
    if (split_env > 0) {
        nsplit = split_env;
    }
    const int kb_per_split = (nkb + nsplit - 1) / nsplit;
    nsplit = (nkb + kb_per_split - 1) / kb_per_split;
    const dim3 grid(GQ_TOKENS_FAST ? (N + GQ_BN - 1)/GQ_BN : (M + GQ_BM - 1)/GQ_BM,
                    GQ_TOKENS_FAST ? (M + GQ_BM - 1)/GQ_BM : (N + GQ_BN - 1)/GQ_BN, nsplit);
    if (nsplit == 1) {
        gcn_q8_gemm<<<grid, GQ_NT, 0, stream>>>((const char *) src0->data, y.get(), (float *) dst->data, (int) M, (int) N, (int) K,
            src0->nb[1], dst->nb[1]/sizeof(float), kb_per_split, 0);
    } else {
        ggml_cuda_pool_alloc<float> part(ctx.pool(), nsplit*M*N);
        gcn_q8_gemm<<<grid, GQ_NT, 0, stream>>>((const char *) src0->data, y.get(), part.get(), (int) M, (int) N, (int) K,
            src0->nb[1], M, kb_per_split, M*N);
        gcn_q8_splitk_reduce<<<(M*N + 255)/256, 256, 0, stream>>>(part.get(), (float *) dst->data, nsplit, M, N, dst->nb[1]/sizeof(float));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// prefill HC up projection -> RESHAPE -> gated DSV4_HC_PRE on GCN (the 4E x N F32 gate, 13 MB per call at 320
// tokens, is never stored or re-read)
bool ggml_cuda_gcn_hc_up_mix_supported(const int cc, const ggml_tensor * mm, const ggml_tensor * pre) {
    static const int env = [] { const char * e = getenv("GGML_CUDA_GCN_HC_UP_MIX"); return e ? atoi(e) : 1; }();
    static const int gemm_env = [] { const char * e = getenv("GGML_CUDA_GCN_GEMM"); return e ? atoi(e) : 1; }();
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * lo = mm->src[1];
    const ggml_tensor * xn = pre->src[0];
    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    return env && gemm_env && GGML_CUDA_CC_IS_GCN(cc) && w->type == GGML_TYPE_Q8_0 && lo->type == GGML_TYPE_F32 &&
        xn->type == GGML_TYPE_F32 && pre->type == GGML_TYPE_F32 && ggml_get_op_params_i32(pre, 1) != 0 && // gated
        xn->ne[1] == 4 && xn->ne[3] == 1 && w->ne[1] == 4*xn->ne[0] && xn->ne[0] % 32 == 0 && w->ne[0] % QK8_0 == 0 &&
        ggml_is_contiguous(w) && w->nb[1] % 4 == 0 && ((uintptr_t) w->data) % 4 == 0 && w->ne[2] == 1 && w->ne[3] == 1 &&
        lo->nb[0] == sizeof(float) && lo->ne[2] == 1 && lo->ne[3] == 1 && lo->ne[1] == xn->ne[2] && lo->ne[1] >= 32 &&
        lo->ne[0] == w->ne[0] && xn->nb[0] == sizeof(float) && pre->nb[0] == sizeof(float) && pre->ne[0] == xn->ne[0] &&
        pre->ne[1] == xn->ne[2] && !overlaps(pre, xn) && !overlaps(pre, lo);
}

void ggml_cuda_gcn_hc_up_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * pre) {
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * lo = mm->src[1];
    const ggml_tensor * xn = pre->src[0];
    const int64_t K = w->ne[0], E = xn->ne[0], N = lo->ne[1];
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool(), N*(K/QK8_1));
    if (lo->nb[1] % 16 == 0 && ((uintptr_t) lo->data) % 16 == 0) {
        gcn_q8_quantize4<<<dim3((K + 1023)/1024, N), 256, 0, stream>>>((const float *) lo->data, y.get(), K, lo->nb[1]/sizeof(float));
    } else {
        gcn_q8_quantize<<<dim3((K + 255)/256, N), 256, 0, stream>>>((const float *) lo->data, y.get(), K, lo->nb[1]/sizeof(float));
    }
    gcn_q8_mix_args mix;
    mix.xn    = (const float *) xn->data;
    mix.pre   = (float *) pre->data;
    mix.E     = (int) E;
    mix.sx1   = xn->nb[1]/sizeof(float);
    mix.sx2   = xn->nb[2]/sizeof(float);
    mix.sp1   = pre->nb[1]/sizeof(float);
    mix.scale = ggml_get_op_params_f32(pre, 0);
    const int nkb = (int) (K / QK8_0);
    const dim3 grid(GQ_TOKENS_FAST ? (N + GQ_BN - 1)/GQ_BN : E/32, GQ_TOKENS_FAST ? E/32 : (N + GQ_BN - 1)/GQ_BN, 1);
    gcn_q8_gemm<true><<<grid, GQ_NT, 0, stream>>>((const char *) w->data, y.get(), nullptr, (int) (4*E), (int) N, (int) K,
        w->nb[1], 0, nkb, 0, mix);
    CUDA_CHECK(cudaGetLastError());
}

// ---- MUL_MAT_ID on the v4 tiles, grouped by expert ----
// prefill MoE (GLM-5-Next TP4: 288 experts, top 8, ~28 tokens per expert at ub 1024): the row-lane kernel is latency-bound
// at ~6-7 TFLOPS and MMQ's q6_K MoE runs at ~1.6; here the expert's tokens form 64-wide v4 tiles (tile map over the
// expert bounds), activations are quantized once per src1 row and gathered per lane, outputs scattered by ids_dst

static __global__ void gkq_moe_tile_map(const int32_t * __restrict__ expert_bounds, int2 * __restrict__ tiles,
        int32_t * __restrict__ n_tiles, const int n_experts, const int nc) {
    extern __shared__ int gkq_moe_scan[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const int per = (n_experts + nt - 1) / nt;
    const int e0  = min(tid*per, n_experts);
    const int e1  = min(e0 + per, n_experts);
    int cnt = 0;
    for (int e = e0; e < e1; ++e) {
        cnt += (expert_bounds[e + 1] - expert_bounds[e] + nc - 1) / nc;
    }
    gkq_moe_scan[tid] = cnt;
    __syncthreads();
    for (int off = 1; off < nt; off <<= 1) {
        const int v = tid >= off ? gkq_moe_scan[tid - off] : 0;
        __syncthreads();
        gkq_moe_scan[tid] += v;
        __syncthreads();
    }
    int t = gkq_moe_scan[tid] - cnt;
    for (int e = e0; e < e1; ++e) {
        const int n = expert_bounds[e + 1] - expert_bounds[e];
        for (int c = 0; c < n; c += nc) {
            tiles[t++] = make_int2(e, c);
        }
    }
    if (tid == nt - 1) {
        *n_tiles = gkq_moe_scan[tid];
    }
}

bool ggml_cuda_gcn_kq_moe_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                    const ggml_tensor * dst) {
    static const int env     = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MOE"); return e ? atoi(e) : 1; }();
    static const int min_avg = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MOE_MIN_AVG"); return e ? atoi(e) : 12; }();
    if (!env || !GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    // GGML_CUDA_GCN_KQ_MOE_IQ=0: the i-quants (iq2_xs, iq3_xxs, iq4_xs: GLM-5.3-Flash UD-Q2_K_XL) on MMQ as before
    static const int env_iq  = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MOE_IQ"); return e ? atoi(e) : 1; }();
    const bool iq = src0->type == GGML_TYPE_IQ2_XS || src0->type == GGML_TYPE_IQ3_XXS || src0->type == GGML_TYPE_IQ4_XS ||
                    src0->type == GGML_TYPE_IQ2_XXS;
    // q2_K (DeepSeek V4's expert down) with the K-quant threshold; GGML_CUDA_GCN_KQ_MOE_Q2K=0 off
    static const int env_q2k = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MOE_Q2K"); return e ? atoi(e) : 1; }();
    const bool type_ok = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K ||
                         (env_q2k && src0->type == GGML_TYPE_Q2_K) || (env_iq && iq);
    // row / expert / base alignment the tile loads need: 16-byte int4 loads (q4_K, q5_K), int2 (iq4_xs), dwords around
    // 2-byte aligned blocks (q6_K, iq2_xs, iq3_xxs)
    const size_t wal = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K ? 16 : src0->type == GGML_TYPE_IQ4_XS ? 8 : 4;
    const int64_t K = src0->ne[0], n_tokens = src1->ne[2], n_used = ids->ne[0], n_exp = src0->ne[2];
    // the i-quants have no mid-size MoE path but MMQ, which the v4 tiles beat from ~1 token per expert (Q2 TP4 prefill:
    // pp128 299 -> 404, pp256 405 -> 581 t/s); the K-quants keep their threshold (GGML_CUDA_GCN_KQ_MOE_MIN_AVG_IQ)
    static const int min_avg_iq = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MOE_MIN_AVG_IQ"); return e ? atoi(e) : 1; }();
    // up to 64 tokens (GGML_CUDA_MOE_IQ_DEC_MAX) stay on moe_coal_iq (decode, MTP verify, several sequences, short
    // prompt tails): the gate/up fusion asks this function first; at one token the tiles cost GLM-5.3 Q2 decode 62 -> 36
    // t/s, and up to ~96 tokens the per-slot kernel wins (was > 4 tokens here: 5-token forward 62.9 -> 32.5 ms)
    static const int iq_dec_max = [] { const char * e = getenv("GGML_CUDA_MOE_IQ_DEC_MAX"); return e && atoi(e) > 0 ? std::min(atoi(e), 256) : 64; }();
    // (iq2_xxs: moe_coal_iq takes up to 32 tokens)
    const int64_t dec_max = src0->type == GGML_TYPE_IQ2_XXS && !getenv("GGML_CUDA_MOE_IQ_DEC_MAX") ? 32 : iq_dec_max;
    const int64_t avg = (n_tokens*n_used + n_exp - 1)/n_exp;
    // iq2_xxs: the row-lane tiles win below ~32 tokens per expert (DeepSeek V4 TP4 pp2048: ub 512 (12 per expert) 745 ->
    // 706 t/s with the tiles here, ub 1024 796 -> 789, ub 2048 (48) 798 -> 811); GGML_CUDA_GCN_KQ_MOE_MIN_AVG_IQ2XXS
    static const int min_avg_xxs = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MOE_MIN_AVG_IQ2XXS"); return e ? atoi(e) : 32; }();
    const int64_t min_avg_t = src0->type == GGML_TYPE_IQ2_XXS ? min_avg_xxs : min_avg_iq;
    return type_ok && (iq ? n_tokens > dec_max && avg >= min_avg_t : avg >= min_avg) && K % QK_K == 0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && ids->ne[1] == n_tokens && src0->ne[3] == 1 &&
        src1->ne[3] == 1 && ggml_is_contiguous(src0) && src0->nb[1] % wal == 0 && src0->nb[2] % wal == 0 &&
        ((uintptr_t) src0->data) % wal == 0 && src1->nb[0] == sizeof(float) && src1->nb[2] == src1->ne[1]*src1->nb[1] &&
        src1->nb[1] % 8 == 0 && ((uintptr_t) src1->data) % 8 == 0 && dst->nb[0] == sizeof(float) &&
        dst->nb[2] == dst->ne[1]*dst->nb[1] && dst->ne[0] == src0->ne[1] && dst->ne[1] == n_used && dst->ne[2] == n_tokens &&
        (src1->ne[1] == 1 || src1->ne[1] == n_used) && src1->ne[1]*src1->ne[2]*K < (int64_t) INT32_MAX;
}

void ggml_cuda_gcn_kq_moe(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                          const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * up_src0, const float glu_limit) {
    cudaStream_t stream = ctx.stream();
    const int64_t K = src0->ne[0], M = src0->ne[1], n_exp = src0->ne[2];
    const int64_t ne11 = src1->ne[1], n_tokens = src1->ne[2], n_used = ids->ne[0];
    const int64_t n_slots = n_tokens*n_used, n_rows = ne11*n_tokens;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_exp + 1);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
        (int) n_exp, (int) n_tokens, (int) n_used, (int) ne11, (int) (ids->nb[1]/sizeof(int32_t)),
        (int) (src1->nb[2]/src1->nb[1]), /*write_inverse =*/ false, stream, &ctx.pool());

    // every src1 row once (gate/up: one per token; down: one per token slot), in natural order
    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), n_rows*K);
    ggml_cuda_pool_alloc<float2> yf(ctx.pool(), n_rows*(K/32));
    gcn_q8x_quantize<<<dim3((K/128 + 3)/4, n_rows, 1), 256, 0, stream>>>((const float *) src1->data, yq.get(), yf.get(), K,
        src1->nb[1]/sizeof(float), 0);

    // tile width: 32 tokens unless the average expert gets > 64 (GGML_CUDA_GCN_KQ_MOE_BN=32/64 forces one): at ub 1024
    // (~28 tokens per expert) 32-wide tiles run 13.7 TFLOPS on q4_K vs 9.1, at ub 2048 16.3 vs 15.8
    static const int bn_env = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MOE_BN"); return e ? atoi(e) : 0; }();
    const int64_t avg = (n_slots + n_exp - 1)/n_exp;
    const int bn = bn_env == 32 || bn_env == 64 ? bn_env : (avg <= 64 ? 32 : 64);
    const int max_tiles = (int) (n_slots/bn + n_exp);
    ggml_cuda_pool_alloc<int2>    tiles(ctx.pool(), max_tiles);
    ggml_cuda_pool_alloc<int32_t> n_tiles(ctx.pool(), 1);
    gkq_moe_tile_map<<<1, 256, 256*sizeof(int), stream>>>(bounds.get(), tiles.get(), n_tiles.get(), (int) n_exp, bn);

    const dim3 grid((M + G3_BM - 1)/G3_BM, max_tiles, 1);
    const int64_t s1 = dst->nb[1]/sizeof(float);
    auto launch = [&](const ggml_tensor * w, const int epi) {
#define GKQ4M_LAUNCH(TT, BNT) gcn_kq4_gemm<TT, true, BNT><<<grid, GQ_NT, 0, stream>>>((const char *) w->data, yq.get(), yf.get(), \
            (float *) dst->data, (int) M, (int) n_rows, (int) K, w->nb[1], s1, (int) (K/QK_K), 0, 1, 0, 0, \
            tiles.get(), n_tiles.get(), bounds.get(), ids_src1.get(), ids_dst.get(), w->nb[2], epi, glu_limit)
#define GKQ4M_BN(TT) if (bn == 32) { GKQ4M_LAUNCH(TT, 32); } else { GKQ4M_LAUNCH(TT, 64); }
        switch (w->type) {
            case GGML_TYPE_Q4_K:    GKQ4M_BN(GGML_TYPE_Q4_K);    break;
            case GGML_TYPE_Q5_K:    GKQ4M_BN(GGML_TYPE_Q5_K);    break;
            case GGML_TYPE_IQ2_XS:  GKQ4M_BN(GGML_TYPE_IQ2_XS);  break;
            case GGML_TYPE_IQ2_XXS: GKQ4M_BN(GGML_TYPE_IQ2_XXS); break;
            case GGML_TYPE_IQ3_XXS: GKQ4M_BN(GGML_TYPE_IQ3_XXS); break;
            case GGML_TYPE_IQ4_XS:  GKQ4M_BN(GGML_TYPE_IQ4_XS);  break;
            case GGML_TYPE_Q2_K:    GKQ4M_BN(GGML_TYPE_Q2_K);    break;
            default:                GKQ4M_BN(GGML_TYPE_Q6_K);    break;
        }
#undef GKQ4M_BN
#undef GKQ4M_LAUNCH
    };
    launch(src0, 0);
    if (up_src0) {
        launch(up_src0, 1);
    }
    CUDA_CHECK(cudaGetLastError());
}
