#include "gcn-kq-matvec.cuh"
#include "mmvq.cuh"
#include "vecdotq.cuh"
#include "unary.cuh"

// Dense K-quant / iq4_xs x q8_1 matvec for 1..4 tokens on GCN (gfx906). MMVQ reads these block formats in 4..8-byte
// pieces per lane and reaches ~400 GB/s on large matrices (q5_K 2048 x 4096: 13.6 us for 5.8 MB) and 130..240 GB/s on
// small or short-K ones; a dense model like Qwen3.8-27B decodes at about that rate. Here an item is one 16-byte run of
// a row's quants (32 values: two runs of 16 that sit in q8_1 blocks of the activation), LPR lanes of a row take
// consecutive items, so a wave streams whole super-blocks; the item's scales come from the super-block header (cached).
// The activations (the shared q8_1 copy) are staged in LDS once per block as int8 runs, their q8_1 scales and the sums
// of each 16-value run (for the K-quants' mins and q6_K's -32 offset), and a block walks RG row groups (long spans).
// Every weight item is decoded once for all NT tokens. GGML_CUDA_GCN_KQ_MV=0: off (MMVQ); GGML_CUDA_GCN_KQ_MV_LPR /
// GGML_CUDA_GCN_KQ_MV_RG: lanes per row / row groups per block.

struct kqmv_item {
    int   qa[4], qb[4]; // 16 int8 values of run A and of run B
    int   ra, rb;       // their 16-value run index within the row (k/16)
    float fa, fb;       // scale of each run
    float ma, mb;       // multiplier of each run's activation sum (K-quant mins, q6_K's -32 offset); 0 for iq4_xs
};

// 16 bytes at a 2-byte aligned address (q6_K: 210-byte super-blocks) from 4..5 aligned dwords
static __device__ __forceinline__ void kqmv_load16_a2(const char * p, int * out) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * w = (const uint32_t *) (a & ~(uintptr_t) 3);
    const int sh = (int) (a & 3);
    const uint32_t v0 = w[0], v1 = w[1], v2 = w[2], v3 = w[3], v4 = w[sh ? 4 : 3];
    out[0] = (int) __builtin_amdgcn_alignbyte(v1, v0, sh);
    out[1] = (int) __builtin_amdgcn_alignbyte(v2, v1, sh);
    out[2] = (int) __builtin_amdgcn_alignbyte(v3, v2, sh);
    out[3] = (int) __builtin_amdgcn_alignbyte(v4, v3, sh);
}

// q4_K / q5_K (as moe_coal_kq_item): item = 16 bytes of qs, low nibbles = 16 values of sub-block 2p, high nibbles = the
// same 16 positions of sub-block 2p + 1 (q5_K: 5th bits from qh)
template <bool Q5>
static __device__ __forceinline__ void kqmv_decode_q45k(const char * __restrict__ row, const int it, kqmv_item & r) {
    constexpr int SB = Q5 ? sizeof(block_q5_K) : sizeof(block_q4_K);
    constexpr int QS = Q5 ? 48 : 16;
    const int sb = it >> 3;
    const int q  = it & 7;
    const int p  = q >> 1;
    const int h  = q & 1;
    const char * b = row + sb*SB;
    const int4 hd = *(const int4 *) b;
    const int4 qw = *(const int4 *) (b + QS + 16*q);
    const int qa[4] = {qw.x, qw.y, qw.z, qw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r.qa[i] = qa[i] & 0x0F0F0F0F;
        r.qb[i] = (qa[i] >> 4) & 0x0F0F0F0F;
    }
    if constexpr (Q5) {
        const int4 qh4 = *(const int4 *) (b + 16 + 16*h);
        const int qh[4] = {qh4.x, qh4.y, qh4.z, qh4.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            r.qa[i] |= ((qh[i] >> (2*p    )) & 0x01010101) << 4;
            r.qb[i] |= ((qh[i] >> (2*p + 1)) & 0x01010101) << 4;
        }
    }
    const float d    = __half2float(__ushort_as_half((unsigned short) (hd.x & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half((unsigned short) ((unsigned) hd.x >> 16)));
    auto sc8 = [&](const int k) {
        const int w = k < 4 ? hd.y : (k < 8 ? hd.z : hd.w);
        return (w >> (8*(k & 3))) & 0xFF;
    };
    int sc[2], mn[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const int s_ = 2*p + t;
        if (s_ < 4) {
            sc[t] = sc8(s_) & 63;
            mn[t] = sc8(s_ + 4) & 63;
        } else {
            sc[t] = (sc8(s_ + 4) & 0xF) | ((sc8(s_ - 4) >> 6) << 4);
            mn[t] = (sc8(s_ + 4) >>  4) | ((sc8(s_    ) >> 6) << 4);
        }
    }
    r.fa = d*sc[0];    r.fb = d*sc[1];
    r.ma = dmin*mn[0]; r.mb = dmin*mn[1];
    r.ra = (sb*8 + 2*p)*2 + h;
    r.rb = r.ra + 2;
}

// q6_K (210-byte super-blocks: ql[128], qh[64], scales[16], d): item = 16 bytes of ql in half n, chunk c: low nibbles =
// values 128n + 16c + 0..15, high nibbles = values 128n + 64 + 16c + 0..15, upper 2 bits from qh; values are q - 32
static __device__ __forceinline__ void kqmv_decode_q6k(const char * __restrict__ row, const int it, kqmv_item & r) {
    const int sb = it >> 3;
    const int q  = it & 7;
    const int n  = q >> 2;
    const int c  = q & 3;
    const char * b = row + sb*(int) sizeof(block_q6_K);
    int ql[4], qh[4];
    kqmv_load16_a2(b + 64*n + 16*c, ql);
    kqmv_load16_a2(b + 128 + 32*n + 16*(c & 1), qh);
    const int shl = c < 2 ? 0 : 2;
    const int shh = c < 2 ? 4 : 6;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r.qa[i] = ( ql[i]       & 0x0F0F0F0F) | (((qh[i] >> shl) & 0x03030303) << 4);
        r.qb[i] = ((ql[i] >> 4) & 0x0F0F0F0F) | (((qh[i] >> shh) & 0x03030303) << 4);
    }
    const int8_t * sc = (const int8_t *) (b + 192 + 8*n);
    const float d = __half2float(__ushort_as_half(*(const unsigned short *) (b + 208)));
    const int is = c & 1;
    r.fa = d*sc[(c < 2 ? 0 : 2) + is];
    r.fb = d*sc[(c < 2 ? 4 : 6) + is];
    r.ma = 32.0f*r.fa;
    r.mb = 32.0f*r.fb;
    r.ra = sb*16 + 8*n + c;
    r.rb = r.ra + 4;
}

// iq4_xs (136-byte super-blocks: d, scales_h, scales_l[4], qs[128]): item = sub-block ib, low nibbles = values 0..15,
// high nibbles = values 16..31, non-linear values from kvalues_iq4nl by byte permutes
static __device__ __forceinline__ void kqmv_decode_iq4xs(const char * __restrict__ row, const int it, kqmv_item & r) {
    const int sb = it >> 3;
    const int ib = it & 7;
    const char * b = row + sb*(int) sizeof(block_iq4_xs);
    const uint32_t h = *(const uint32_t *) b;
    const uint2 q0 = *(const uint2 *) (b + 8 + 16*ib);
    const uint2 q1 = *(const uint2 *) (b + 16 + 16*ib);
    const int qs[4] = {(int) q0.x, (int) q0.y, (int) q1.x, (int) q1.y};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int2 v = get_int_from_table_16(qs[i], kvalues_iq4nl);
        r.qa[i] = v.x;
        r.qb[i] = v.y;
    }
    const int ls = ((((uint8_t) b[4 + ib/2]) >> (4*(ib & 1))) & 0xF) | (((h >> (16 + 2*ib)) & 3) << 4);
    const float d = __half2float(__ushort_as_half((unsigned short) (h & 0xFFFF)));
    r.fa = r.fb = d*(ls - 32);
    r.ma = r.mb = 0.0f;
    r.ra = (sb*8 + ib)*2;
    r.rb = r.ra + 1;
}

template <ggml_type type>
static __device__ __forceinline__ void kqmv_decode(const char * __restrict__ row, const int it, kqmv_item & r) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        kqmv_decode_q45k<false>(row, it, r);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        kqmv_decode_q45k<true>(row, it, r);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        kqmv_decode_q6k(row, it, r);
    } else {
        static_assert(type == GGML_TYPE_IQ4_XS, "unsupported type");
        kqmv_decode_iq4xs(row, it, r);
    }
}

template <ggml_type type, int NT, int LPR>
__launch_bounds__(256)
static __global__ void gcn_kq_mv(const char * __restrict__ W, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const int RG, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride,
        const float * add, const int64_t a_stride) { // add: an addend of dst's shape (may be dst itself) or nullptr
    constexpr bool MIN = type != GGML_TYPE_IQ4_XS;
    extern __shared__ int4 kqmv_lds[];
    const int nyb = K / QK8_1;
    int8_t * yq   = (int8_t *) kqmv_lds;      // NT x K int8
    float  * yd   = (float *) (yq + NT*K);    // NT x K/32 q8_1 scales
    int    * ysum = (int *) (yd + NT*nyb);    // NT x K/16 run sums
    for (int i = threadIdx.x; i < NT*nyb; i += 256) {
        const int t = i / nyb;
        const int j = i - t*nyb;
        const block_q8_1 * bq = y + t*y_stride + j;
        const int * src = (const int *) bq;
        int v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = src[1 + k];
        }
        int4 * dq = (int4 *) (yq + t*K + 32*j);
        dq[0] = make_int4(v[0], v[1], v[2], v[3]);
        dq[1] = make_int4(v[4], v[5], v[6], v[7]);
        yd[t*nyb + j] = __low2float(bq->ds);
        if constexpr (MIN) {
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s0 = ggml_cuda_dp4a(v[k],     0x01010101, s0);
                s1 = ggml_cuda_dp4a(v[k + 4], 0x01010101, s1);
            }
            ysum[t*2*nyb + 2*j]     = s0;
            ysum[t*2*nyb + 2*j + 1] = s1;
        }
    }
    __syncthreads();

    const int sub = threadIdx.x % LPR;
    const int nit = (K / QK_K)*8;
#pragma unroll 1
    for (int g = 0; g < RG; ++g) {
        const int row = (blockIdx.x*RG + g)*(256/LPR) + threadIdx.x / LPR;
        if (row >= M) {
            break; // whole lane groups; nothing after the loop needs the block
        }
        const char * wr = W + (int64_t) row*w_stride;
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[t] = 0.0f;
        }
#pragma unroll 2
        for (int it = sub; it < nit; it += LPR) {
            kqmv_item r;
            kqmv_decode<type>(wr, it, r);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int4 A = *(const int4 *) (yq + t*K + 16*r.ra);
                const int4 B = *(const int4 *) (yq + t*K + 16*r.rb);
                int sa = 0, sb = 0;
                sa = ggml_cuda_dp4a(r.qa[0], A.x, sa); sa = ggml_cuda_dp4a(r.qa[1], A.y, sa);
                sa = ggml_cuda_dp4a(r.qa[2], A.z, sa); sa = ggml_cuda_dp4a(r.qa[3], A.w, sa);
                sb = ggml_cuda_dp4a(r.qb[0], B.x, sb); sb = ggml_cuda_dp4a(r.qb[1], B.y, sb);
                sb = ggml_cuda_dp4a(r.qb[2], B.z, sb); sb = ggml_cuda_dp4a(r.qb[3], B.w, sb);
                const float da = yd[t*nyb + (r.ra >> 1)];
                const float db = yd[t*nyb + (r.rb >> 1)];
                float v = da*(r.fa*sa) + db*(r.fb*sb);
                if constexpr (MIN) {
                    v -= da*(r.ma*ysum[t*2*nyb + r.ra]) + db*(r.mb*ysum[t*2*nyb + r.rb]);
                }
                acc[t] += v;
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[t];
#pragma unroll
            for (int off = LPR/2; off > 0; off >>= 1) {
                a += __shfl_xor(a, off, LPR);
            }
            if (sub == t) {
                dst[t*d_stride + row] = add ? a + add[t*a_stride + row] : a;
            }
        }
    }
}

// q4_K / q5_K 6-bit scale and min of sub-block s_ (0..7) from the header (d, dmin, scales[12] = hd.y .. hd.w), by
// shifts and selects (component selection by a runtime index went through scratch memory)
static __device__ __forceinline__ void kqmv_scmin(const int4 hd, const int s_, int & sc, int & mn) {
    const int j  = 8*(s_ & 3);
    const int y_ = (hd.y >> j) & 0xFF;
    const int z_ = (hd.z >> j) & 0xFF;
    const int w_ = (hd.w >> j) & 0xFF;
    const bool lo = s_ < 4;
    sc = lo ? y_ & 63 : (w_ & 0xF) | ((y_ >> 6) << 4);
    mn = lo ? z_ & 63 : (w_ >>  4) | ((z_ >> 6) << 4);
}

// One token, q4_K / q5_K (GGML_CUDA_GCN_KQ_MV1=0: MMVQ): 4 lanes per super-block, lane p takes the sub-block pair (2p, 2p+1)
// whole (32 bytes of qs, q5_K: all 32 bytes of qh, the 16-byte header), so the q8_1 block sums of the activation give
// the min terms directly (no run sums) and the scales are decoded once per 64 values; the activation comes from L1
// (no LDS copy: the copy and its barrier delayed every block's weight stream). A wave walks R rows as one flat list of
// super-blocks, 16 per step, so short rows still fill every lane group. MI50 vs MMVQ: q4_K 8704 x 5120 47.1 -> 42.2 us,
// 4096 x 4096 20.1 -> 16.5, q5_K 8704 x 5120 54.1 -> 48.6, 2048 x 4096 13.2 -> 10.5, 5120 x 8704 54.0 -> 52.2 (a 16-byte
// contiguous stream through LDS was no faster). With the gate/up + SwiGLU fusion: Qwen3.8-27B TP2 tg 43.1 -> 44.0 t/s.
template <bool Q5, int R, bool GLU, int NT>
__launch_bounds__(256)
static __global__ void gcn_kq_mv1(const char * __restrict__ W, const char * __restrict__ Wg, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int nsb, const int M, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride,
        const float glu_limit) {
    typedef int kqmv_i4a __attribute__((ext_vector_type(4), aligned(4)));
    constexpr int SB = Q5 ? (int) sizeof(block_q5_K) : (int) sizeof(block_q4_K);
    constexpr int QS = Q5 ? 48 : 16;
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int g    = lane / 4;
    const int p    = lane % 4;
    const int row0 = (blockIdx.x*4 + wave)*R;
    if (row0 >= M) {
        return;
    }
    const int nrows = min(R, M - row0);
    float acc[NT][R], accg[NT][R];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            acc[t][rr]  = 0.0f;
            accg[t][rr] = 0.0f;
        }
    }
    // GLU: the gate rows (pass 0) then the up rows (pass 1), one flat walk
    const int npass = GLU ? 2 : 1;
    int r  = g / nsb;
    int sb = g - r*nsb;
    for (; r < npass*nrows; ) {
        const bool gp = GLU && r < nrows;
        const int  rl = GLU && !gp ? r - nrows : r;
        const char * b = (gp ? Wg : W) + (int64_t) (row0 + rl)*w_stride + (int64_t) sb*SB;
        const int4 hd = *(const int4 *) b;
        const int4 q0 = *(const int4 *) (b + QS + 32*p);
        const int4 q1 = *(const int4 *) (b + QS + 32*p + 16);
        int4 h0 = make_int4(0, 0, 0, 0), h1 = make_int4(0, 0, 0, 0);
        if constexpr (Q5) {
            h0 = *(const int4 *) (b + 16);
            h1 = *(const int4 *) (b + 32);
        }
        // the weights are decoded once for all NT tokens
        const int qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const int hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        int va[8], vb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            va[i] = qw[i] & 0x0F0F0F0F;
            vb[i] = (qw[i] >> 4) & 0x0F0F0F0F;
            if constexpr (Q5) {
                va[i] |= ((hw[i] >> (2*p    )) & 0x01010101) << 4;
                vb[i] |= ((hw[i] >> (2*p + 1)) & 0x01010101) << 4;
            }
        }
        const float d    = __half2float(__ushort_as_half((unsigned short) (hd.x & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((unsigned short) ((unsigned) hd.x >> 16)));
        int sc[2], mn[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            kqmv_scmin(hd, 2*p + k, sc[k], mn[k]);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const block_q8_1 * ya = y + t*y_stride + sb*8 + 2*p;
            const block_q8_1 * yb = ya + 1;
            const kqmv_i4a ya0 = ((const kqmv_i4a *) ya->qs)[0], ya1 = ((const kqmv_i4a *) ya->qs)[1];
            const kqmv_i4a yb0 = ((const kqmv_i4a *) yb->qs)[0], yb1 = ((const kqmv_i4a *) yb->qs)[1];
            const float2 dsa = __half22float2(ya->ds);
            const float2 dsb = __half22float2(yb->ds);
            const int yaw[8] = {ya0[0], ya0[1], ya0[2], ya0[3], ya1[0], ya1[1], ya1[2], ya1[3]};
            const int ybw[8] = {yb0[0], yb0[1], yb0[2], yb0[3], yb1[0], yb1[1], yb1[2], yb1[3]};
            int sa = 0, sbs = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sa  = ggml_cuda_dp4a(va[i], yaw[i], sa);
                sbs = ggml_cuda_dp4a(vb[i], ybw[i], sbs);
            }
            const float v = d*(dsa.x*(float) (sc[0]*sa) + dsb.x*(float) (sc[1]*sbs)) - dmin*(dsa.y*(float) mn[0] + dsb.y*(float) mn[1]);
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                if (GLU && gp) {
                    accg[t][rr] += rr == rl ? v : 0.0f;
                } else {
                    acc[t][rr]  += rr == rl ? v : 0.0f;
                }
            }
        }
        sb += 16;
        while (sb >= nsb) {
            sb -= nsb;
            r++;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            float a = warp_reduce_sum<64>(acc[t][rr]);
            if constexpr (GLU) {
                const float gv = warp_reduce_sum<64>(accg[t][rr]);
                // SWIGLU_CLAMP when the limit is finite (as MMVQ's fused path), else SWIGLU
                a = isinf(glu_limit) ? a*ggml_cuda_op_silu_single(gv) : ggml_cuda_op_swiglu_clamp_single(gv, a, glu_limit);
            }
            if (lane == 0 && rr < nrows) {
                dst[t*d_stride + row0 + rr] = a;
            }
        }
    }
}


// 2..4 tokens (GGML_CUDA_GCN_KQ_MVS, default on where faster): as gcn_kq_mv1, but the tokens' activations are staged once
// per block in LDS (int8 values, 16-byte aligned rows, and the q8_1 (d, sum) pairs) and the blocks walk row groups with a
// grid stride (a few blocks per CU, so the staging is paid ~4x per CU instead of per 4-8 rows): the per-token y loads
// from L1 were what made gcn_kq_mv1 lose from 3 tokens.
template <ggml_type TYPE, int R, bool GLU, int NT>
__launch_bounds__(256)
static __global__ void gcn_kq_mvs(const char * __restrict__ W, const char * __restrict__ Wg, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int nsb, const int M, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride,
        const float glu_limit) {
    constexpr bool Q5 = TYPE == GGML_TYPE_Q5_K;
    constexpr bool I4 = TYPE == GGML_TYPE_IQ4_XS; // 136-byte blocks: d, scales_h, scales_l[4], qs[128]; no mins
    typedef int kqmvs_i4a __attribute__((ext_vector_type(4), aligned(4)));
    constexpr int SB = I4 ? (int) sizeof(block_iq4_xs) : Q5 ? (int) sizeof(block_q5_K) : (int) sizeof(block_q4_K);
    constexpr int QS = Q5 ? 48 : 16;
    extern __shared__ int4 kqmvs_lds[];
    const int K   = nsb*QK_K;
    const int nyb = K/QK8_1;
    int8_t  * yq = (int8_t *) kqmvs_lds;          // NT x K int8
    half2   * yd = (half2 *) (yq + NT*K);         // NT x K/32 (d, sum)
    for (int i = threadIdx.x; i < NT*nyb; i += 256) {
        const int t = i / nyb;
        const int j = i - t*nyb;
        const block_q8_1 * bq = y + t*y_stride + j;
        const int * src = (const int *) bq;
        int4 * dq = (int4 *) (yq + t*K + 32*j);
        dq[0] = make_int4(src[1], src[2], src[3], src[4]);
        dq[1] = make_int4(src[5], src[6], src[7], src[8]);
        yd[t*nyb + j] = bq->ds;
    }
    __syncthreads();

    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int g    = lane / 4;
    const int p    = lane % 4;
    const int ngroups = (M + 4*R - 1)/(4*R);
    const int npass = GLU ? 2 : 1;
    for (int grp = blockIdx.x; grp < ngroups; grp += gridDim.x) {
        const int row0 = (grp*4 + wave)*R;
        if (row0 >= M) {
            continue;
        }
        const int nrows = min(R, M - row0);
        float acc[NT][R], accg[NT][R];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                acc[t][rr]  = 0.0f;
                accg[t][rr] = 0.0f;
            }
        }
        int r  = g / nsb;
        int sb = g - r*nsb;
        for (; r < npass*nrows; ) {
            const bool gp = GLU && r < nrows;
            const int  rl = GLU && !gp ? r - nrows : r;
            const char * b = (gp ? Wg : W) + (int64_t) (row0 + rl)*w_stride + (int64_t) sb*SB;
            int va[8], vb[8];
            float d = 0.0f, dmin = 0.0f;
            int sc[2] = {0, 0}, mn[2] = {0, 0};
            if constexpr (I4) {
                // sub-blocks 2p and 2p + 1: 16 bytes each (low nibbles = values 0..15, high = 16..31), 8-byte aligned
                const uint2 hd2 = *(const uint2 *) b;
                const kqmvs_i4a q0 = *(const kqmvs_i4a *) (b + 8 + 32*p);
                const kqmvs_i4a q1 = *(const kqmvs_i4a *) (b + 8 + 32*p + 16);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int2 v0 = get_int_from_table_16(q0[i], kvalues_iq4nl);
                    const int2 v1 = get_int_from_table_16(q1[i], kvalues_iq4nl);
                    va[i] = v0.x; va[4 + i] = v0.y;
                    vb[i] = v1.x; vb[4 + i] = v1.y;
                }
                d = __half2float(__ushort_as_half((unsigned short) (hd2.x & 0xFFFF)));
                const uint32_t sh = hd2.x >> 16;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int ib = 2*p + k;
                    sc[k] = (int) (((hd2.y >> (8*(ib/2) + 4*(ib & 1))) & 0xF) | (((sh >> (2*ib)) & 3) << 4)) - 32;
                }
            } else {
                const int4 hd = *(const int4 *) b;
                const int4 q0 = *(const int4 *) (b + QS + 32*p);
                const int4 q1 = *(const int4 *) (b + QS + 32*p + 16);
                int4 h0 = make_int4(0, 0, 0, 0), h1 = make_int4(0, 0, 0, 0);
                if constexpr (Q5) {
                    h0 = *(const int4 *) (b + 16);
                    h1 = *(const int4 *) (b + 32);
                }
                const int qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
                const int hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    va[i] = qw[i] & 0x0F0F0F0F;
                    vb[i] = (qw[i] >> 4) & 0x0F0F0F0F;
                    if constexpr (Q5) {
                        va[i] |= ((hw[i] >> (2*p    )) & 0x01010101) << 4;
                        vb[i] |= ((hw[i] >> (2*p + 1)) & 0x01010101) << 4;
                    }
                }
                d    = __half2float(__ushort_as_half((unsigned short) (hd.x & 0xFFFF)));
                dmin = __half2float(__ushort_as_half((unsigned short) ((unsigned) hd.x >> 16)));
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    kqmv_scmin(hd, 2*p + k, sc[k], mn[k]);
                }
            }
            const int ja = sb*8 + 2*p; // q8_1 block of sub-block 2p (2p + 1: ja + 1)
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int4 * ya = (const int4 *) (yq + t*K + 32*ja);
                const int4 a0 = ya[0], a1 = ya[1], b0 = ya[2], b1 = ya[3];
                const float2 dsa = __half22float2(yd[t*nyb + ja]);
                const float2 dsb = __half22float2(yd[t*nyb + ja + 1]);
                const int yaw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const int ybw[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
                int sa = 0, sbs = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sa  = ggml_cuda_dp4a(va[i], yaw[i], sa);
                    sbs = ggml_cuda_dp4a(vb[i], ybw[i], sbs);
                }
                const float v = I4 ? d*(dsa.x*(float) (sc[0]*sa) + dsb.x*(float) (sc[1]*sbs)) :
                    d*(dsa.x*(float) (sc[0]*sa) + dsb.x*(float) (sc[1]*sbs)) - dmin*(dsa.y*(float) mn[0] + dsb.y*(float) mn[1]);
#pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    if (GLU && gp) {
                        accg[t][rr] += rr == rl ? v : 0.0f;
                    } else {
                        acc[t][rr]  += rr == rl ? v : 0.0f;
                    }
                }
            }
            sb += 16;
            while (sb >= nsb) {
                sb -= nsb;
                r++;
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                float a = warp_reduce_sum<64>(acc[t][rr]);
                if constexpr (GLU) {
                    const float gv = warp_reduce_sum<64>(accg[t][rr]);
                    a = isinf(glu_limit) ? a*ggml_cuda_op_silu_single(gv) : ggml_cuda_op_swiglu_clamp_single(gv, a, glu_limit);
                }
                if (lane == 0 && rr < nrows) {
                    dst[t*d_stride + row0 + rr] = a;
                }
            }
        }
    }
}

// launch gcn_kq_mv1 for N (1..4) tokens: rows per wave from M (>= ~2048 waves) and capped for the accumulators of NT tokens
template <bool GLU>
static void kqmv1_launch(const ggml_type type, const char * W, const char * Wg, const block_q8_1 * yq, float * dst, const int nsb,
        const int64_t M, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride, const int N, cudaStream_t st,
        const float glu_limit = INFINITY) {
    static const int r_env = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MV1_R"); return e ? atoi(e) : 0; }();
    int R = r_env ? r_env : 1;
    const int r_max = N >= 3 ? 2 : N == 2 ? 4 : 8;
    while (!r_env && R < r_max && M/(2*R) >= 2048) {
        R *= 2;
    }
    const dim3 grid((unsigned) ((M + 4*R - 1)/(4*R)));
    static const int mvs_env = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MVS"); return e ? atoi(e) : 1; }();
    static const int mvs_bpc = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MVS_BPC"); return e ? atoi(e) : 2; }();
    const size_t mvs_lds = (size_t) N*(nsb*QK_K + (nsb*QK_K/QK8_1)*sizeof(half2));
    // large matrices only (>= 16M weights): on small ones the LDS copy costs more than it saves (2 tokens, q5_K: 4096 x 2048
    // 16.1 -> 17.3 us, 512 x 4096 6.4 -> 6.8; 2048 x 4096 15.3 -> 14.6)
    if (mvs_env && N >= 2 && mvs_lds <= 32*1024 && M*(int64_t) nsb*QK_K >= 16*1024*1024) {
        const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        const int ngroups = (int) ((M + 4*R - 1)/(4*R));
        const dim3 grid_s((unsigned) std::min(ngroups, mvs_bpc*nsm));
#define KQMVS(T_, R_, NT_) gcn_kq_mvs<T_, R_, GLU, NT_><<<grid_s, 256, mvs_lds, st>>>(W, Wg, yq, dst, nsb, (int) M, w_stride, y_stride, d_stride, glu_limit)
#define KQMVS_R(T_, NT_) switch (R) { case 1: KQMVS(T_, 1, NT_); break; case 2: KQMVS(T_, 2, NT_); break; \
        default: KQMVS(T_, 4, NT_); break; }
#define KQMVS_N(T_) switch (N) { case 2: KQMVS_R(T_, 2) break; case 3: KQMVS_R(T_, 3) break; default: KQMVS_R(T_, 4) break; }
        if (type == GGML_TYPE_Q5_K) {
            KQMVS_N(GGML_TYPE_Q5_K)
        } else if (type == GGML_TYPE_IQ4_XS) {
            KQMVS_N(GGML_TYPE_IQ4_XS)
        } else {
            KQMVS_N(GGML_TYPE_Q4_K)
        }
#undef KQMVS_N
#undef KQMVS_R
#undef KQMVS
        return;
    }
    GGML_ASSERT(type != GGML_TYPE_IQ4_XS && "iq4_xs: gcn_kq_mvs only (see kqmv1_ok)");
#define KQMV1L(Q5_, R_, NT_) gcn_kq_mv1<Q5_, R_, GLU, NT_><<<grid, 256, 0, st>>>(W, Wg, yq, dst, nsb, (int) M, w_stride, y_stride, d_stride, glu_limit)
#define KQMV1L_R(Q5_, NT_) switch (R) { case 1: KQMV1L(Q5_, 1, NT_); break; case 2: KQMV1L(Q5_, 2, NT_); break; \
        case 4: KQMV1L(Q5_, 4, NT_); break; default: KQMV1L(Q5_, 8, NT_); break; }
#define KQMV1L_N(Q5_) switch (N) { case 1: KQMV1L_R(Q5_, 1) break; case 2: KQMV1L_R(Q5_, 2) break; case 3: KQMV1L_R(Q5_, 3) break; \
        default: KQMV1L_R(Q5_, 4) break; }
    if (type == GGML_TYPE_Q5_K) {
        KQMV1L_N(true)
    } else {
        KQMV1L_N(false)
    }
#undef KQMV1L_N
#undef KQMV1L_R
#undef KQMV1L
}

// which token counts use gcn_kq_mv1 (GGML_CUDA_GCN_KQ_MV1: 0 off, 1 default (1..2 tokens), 2 every N in 1..4)
static bool kqmv1_ok(const ggml_tensor * src0, const int64_t N) {
    static const int env1 = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MV1"); return e ? atoi(e) : 1; }();
    const int64_t K = src0->ne[0];
    static const int env_i4 = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MVS_IQ4"); return e ? atoi(e) : 1; }();
    if (src0->type == GGML_TYPE_IQ4_XS) {
        // gcn_kq_mvs only (no gcn_kq_mv1 instance), 2..3 tokens, >= 16M weights, and only where gcn_kq_mv's LDS copy is too big
        // for it (it is faster where it applies: 3 tokens 8704 x 5120 58.0 vs 76.1 us; here 5120 x 8704 97.2 (MMVQ) -> 68.4)
        return env1 && env_i4 && K >= 2048 && K % QK_K == 0 && src0->nb[1] % 8 == 0 && ((uintptr_t) src0->data) % 8 == 0 &&
            (N == 2 || N == 3) && src0->ne[1]*K >= 16*1024*1024 && N*(K + (K/QK8_1)*(int64_t) sizeof(half2)) <= 32*1024 &&
            (double) N*K*1.375 > 24*1024;
    }
    if (!env1 || (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K) || K < 2048 || K % QK_K != 0 ||
            src0->nb[1] % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0 || N < 1 || N > 4) {
        return false;
    }
    // 1 token: gcn_kq_mv1; 2 tokens, and 3 for q5_K: gcn_kq_mvs (activations in LDS). MI50, us, previous path -> here:
    // 2 tokens q5_K 8704 x 5120 71.7 -> ~62, 5120 x 8704 72.1 -> ~56, q4_K 60.7 -> ~54, 66.7 -> ~51; 3 tokens q5_K 88.5 -> ~80,
    // 89.0 -> ~74 (q4_K at 3 tokens: gcn_kq_mv / MMVQ stay, 70.4 vs ~75 at 8704 x 5120); 4 tokens: the LDS copy outgrows it
    return env1 == 2 || N <= 2 || (N == 3 && src0->type == GGML_TYPE_Q5_K && src0->ne[1]*K >= 16*1024*1024);
}

static bool kqmv_type_ok(const ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ4_XS;
}

bool ggml_cuda_gcn_kq_matvec_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int env = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MV"); return e ? atoi(e) : 1; }();
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const size_t al = src0->type == GGML_TYPE_Q6_K ? 2 : src0->type == GGML_TYPE_IQ4_XS ? 8 : 16;
    // where it beats MMVQ (gfx906 us, MMVQ -> here): short K, all types (q5_K 4096 x 512 at 1 / 2 tokens 10.9 -> 6.7 /
    // 14.5 -> 7.5, q6_K 9.5 -> 7.8 / 12.9 -> 8.8); iq4_xs (8704 x 5120 at 1 / 3 tokens 44.2 -> 43.4 / 90.8 -> 58.9) and
    // q4_K from 2 tokens with >= 4096 rows (8704 x 5120 at 3: 83.4 -> 71.9) while the LDS copy stays small (<= 24 KB:
    // more costs occupancy, 5120 x 8704 q4_K at 3 tokens 83.5 -> 126.6). Long-K q5_K / q6_K stay on MMVQ (its ~550 GB/s
    // on large matrices; the per-item header and qh loads cost more here: q5_K 8704 x 5120 54.3 -> 58.8).
    // GGML_CUDA_GCN_KQ_MV=2: every supported shape (experiments)
    const bool small_lds = (double) N*K*1.375 <= 24*1024;
    const bool mv1 = kqmv1_ok(src0, N);
    const bool win = env == 2 || mv1 || K <= 1024 || (src0->type == GGML_TYPE_IQ4_XS && small_lds) ||
        (src0->type == GGML_TYPE_Q4_K && N >= 2 && M >= 4096 && small_lds);
    return env && win && GGML_CUDA_CC_IS_GCN(cc) && kqmv_type_ok(src0->type) && N >= 1 && N <= 4 && K % QK_K == 0 &&
        K <= 32768 && src0->ne[1] <= INT_MAX && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src0->nb[1] % al == 0 && ((uintptr_t) src0->data) % al == 0 &&
        src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float) && dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float);
}

// MUL_MAT + an addend of its output's shape (ADD) as one launch where the plain MUL_MAT would run gcn_kq_mv anyway (same
// sums, one rounding for the add): 2..4 tokens (GLM-5.3's shared expert down + the routed output at the MTP verify)
bool ggml_cuda_gcn_kq_matvec_add_supported(int cc, const ggml_tensor * mm, const ggml_tensor * add_node, const ggml_tensor * addend) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_KQMV_ADD"); return !e || atoi(e) != 0; }();
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const int64_t N = src1->ne[1];
    return env && mm->op == GGML_OP_MUL_MAT && N >= 2 && ggml_cuda_gcn_kq_matvec_supported(cc, src0, src1, mm) && !kqmv1_ok(src0, N) &&
        add_node->type == GGML_TYPE_F32 && addend->type == GGML_TYPE_F32 && ggml_are_same_shape(add_node, mm) &&
        ggml_are_same_shape(addend, mm) && add_node->nb[0] == sizeof(float) && addend->nb[0] == sizeof(float) &&
        mm->ne[2] == 1 && mm->ne[3] == 1;
}

void ggml_cuda_gcn_kq_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * add) {
    static const int lpr_env = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MV_LPR"); return e ? atoi(e) : 0; }();
    static const int rg_env  = [] { const char * e = getenv("GGML_CUDA_GCN_KQ_MV_RG"); return e ? atoi(e) : 0; }();
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> yq_local(ctx.pool());
    const block_q8_1 * yq = (const block_q8_1 *) ggml_cuda_q8_1_activation(ctx, src1, src0->type, K_pad, yq_local);
    const int64_t ys = K_pad / QK8_1;
    const int64_t ds = dst->nb[1] / sizeof(float);
    GGML_ASSERT(!add || !kqmv1_ok(src0, N));
    const int64_t as = add ? add->nb[1] / sizeof(float) : 0;
    if (kqmv1_ok(src0, N)) {
        kqmv1_launch<false>(src0->type, (const char *) src0->data, nullptr, yq, (float *) dst->data, (int) (K / QK_K), M, src0->nb[1],
            ys, ds, (int) N, ctx.stream());
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int nit = (int) (K / QK_K)*8;
    // lanes per row: about 8 items per lane (one item = 32 values)
    const int lpr = lpr_env ? lpr_env : nit >= 512 ? 64 : nit >= 256 ? 32 : nit >= 96 ? 16 : 8;
    const int64_t groups = (M + 256/lpr - 1)/(256/lpr);
    // row groups per block: long spans, but >= ~480 blocks (8 per CU) where the rows allow
    const int rg = rg_env ? rg_env : (int) std::max<int64_t>(1, std::min<int64_t>(8, groups/480));
    const bool mins = src0->type != GGML_TYPE_IQ4_XS;
    // tokens per launch: the LDS copy (int8 runs + scales + run sums) must fit
    const int64_t per_tok = K + (K/QK8_1)*4 + (mins ? (K/16)*4 : 0);
    const int nt_max = (int) std::max<int64_t>(1, std::min<int64_t>(4, (48*1024)/per_tok));
    cudaStream_t stream = ctx.stream();
    for (int64_t t0 = 0; t0 < N; t0 += nt_max) {
        const int nt = (int) std::min<int64_t>(nt_max, N - t0);
        const size_t smem = (size_t) nt*per_tok;
        const block_q8_1 * yt = yq + t0*ys;
        float * dt = (float *) dst->data + t0*ds;
        const float * at = add ? (const float *) add->data + t0*as : nullptr;
        const dim3 grid((unsigned) ((groups + rg - 1)/rg));
#define KQMV(T_, NT_, L_) gcn_kq_mv<T_, NT_, L_><<<grid, 256, smem, stream>>>((const char *) src0->data, yt, dt, (int) K, (int) M, rg, \
            src0->nb[1], ys, ds, at, as)
#define KQMV_L(T_, NT_) switch (lpr) { case 64: KQMV(T_, NT_, 64); break; case 32: KQMV(T_, NT_, 32); break; \
            case 16: KQMV(T_, NT_, 16); break; default: KQMV(T_, NT_, 8); break; }
#define KQMV_N(T_) switch (nt) { case 1: KQMV_L(T_, 1) break; case 2: KQMV_L(T_, 2) break; case 3: KQMV_L(T_, 3) break; \
            default: KQMV_L(T_, 4) break; }
        switch (src0->type) {
            case GGML_TYPE_Q4_K: KQMV_N(GGML_TYPE_Q4_K) break;
            case GGML_TYPE_Q5_K: KQMV_N(GGML_TYPE_Q5_K) break;
            case GGML_TYPE_Q6_K: KQMV_N(GGML_TYPE_Q6_K) break;
            default:             KQMV_N(GGML_TYPE_IQ4_XS) break;
        }
#undef KQMV_N
#undef KQMV_L
#undef KQMV
    }
    CUDA_CHECK(cudaGetLastError());
}

// gate/up + SwiGLU at one token (q4_K / q5_K, same type and shape): one launch, dst = up * silu(gate) (MMVQ's fused order)
bool ggml_cuda_gcn_kq_matvec_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, ggml_tensor * dst, const float glu_limit) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = up->ne[0], M = up->ne[1], N = src1->ne[1];
    if (!GGML_CUDA_CC_IS_GCN(cc) || !kqmv1_ok(up, N) || !kqmv1_ok(gate, N) || gate->type != up->type ||
            !ggml_are_same_shape(up, gate) || up->ne[2] != 1 || up->ne[3] != 1 || !ggml_is_contiguous(up) ||
            !ggml_is_contiguous(gate) || up->nb[1] != gate->nb[1] ||
            src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 || src1->ne[3] != 1 || src1->ne[0] != K || src1->nb[0] != sizeof(float) ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != M || dst->ne[1] != N) {
        return false;
    }
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> yq_local(ctx.pool());
    const block_q8_1 * yq = (const block_q8_1 *) ggml_cuda_q8_1_activation(ctx, src1, up->type, K_pad, yq_local);
    kqmv1_launch<true>(up->type, (const char *) up->data, (const char *) gate->data, yq, (float *) dst->data, (int) (K / QK_K), M,
        up->nb[1], K_pad / QK8_1, dst->nb[1] / sizeof(float), (int) N, ctx.stream(), glu_limit);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
