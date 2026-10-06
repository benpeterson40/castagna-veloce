// Dense prefill flash attention for gfx906 (MI50): D = 256 (K and V), f16 K/V, f32 Q and output, GQA ratio G with
// G | 48 (48 rows = TQ tokens x G query heads of one KV head per block). The attention core is qsa_attn's
// (fattn-qsa-gcn.cu) on contiguous cells instead of a per-group cell list:
//   1. fdg_prep    (one block per TQ-token group, grid.y = 2): y = 0: kv_end[g] = 1 + the last cell with a finite mask
//                  value for any token of the group (backward scan in 256-cell blocks, stops at the first block with one:
//                  a few blocks for causal masks); y = 1: Q*scale in f16 (scale in f32 first, then round: the tile
//                  kernel's order), rows padded to whole groups with zeros.
//   2. fdg_attn    (group, KV head, segment of [0, kv_end)): 64-cell chunks, QK in f32 (v_dot2_f32_f16) + the exact f16 mask
//                  value, online softmax, P in f16, PV accumulated in f32.
//   3. fdg_combine (nseg > 1): merges the segments in fixed order (deterministic). The only writer of dst (or fdg_attn
//                  itself when nseg == 1).
// Semantics equal FLASH_ATTN_EXT without sinks/ALiBi/softcap; only the precision (f32 accumulation instead of the tile
// kernel's half2 VKQ) and the summation order differ. A row with no visible cell gives 0 (like the CPU backend).
// All launch geometry depends on shapes only (HIP graph capture is safe). The file also builds standalone (FDG_STANDALONE,
// ~/mi50-engine/tools/fdg/fdg_bench.hip includes it).

#ifdef FDG_STANDALONE
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include "fattn-dense-gcn.cuh"
#endif

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(GGML_USE_HIP) || defined(FDG_STANDALONE)

// device code for gfx906 only (v_dot2_f32_f16, wave64 DPP); other targets compile empty kernels
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__)
#define FDG_DEVICE 1
#else
#define FDG_DEVICE 0
#endif

#ifndef FDG_QK_UNROLL
#define FDG_QK_UNROLL 2
#endif
#ifndef FDG_PV_UNROLL
#define FDG_PV_UNROLL 2
#endif
#define FDG_PSTR(x) #x
#define FDG_PRAGMA(x) _Pragma(FDG_PSTR(x))

namespace fdg {

constexpr int D   = 256;          // head size (K and V)
constexpr int NT  = 256;          // threads per block
constexpr int R   = 48;           // rows per block (TQ tokens x G heads)
constexpr int C   = 64;           // cells per chunk
constexpr int DS  = 64;           // dims per K stage
#ifndef FDG_KP
#define FDG_KP (DS/2 + 2)
#endif
// half2 per K/Q row in LDS: 34 (136 B, 8-B aligned rows: staged with b64 stores) makes the QK K reads (16 cells, b64)
// cover all 32 banks once; 36 (144 B) put cells cg and cg + 8 on the same banks
constexpr int KP  = FDG_KP;
constexpr int VP  = D + 4;        // half2 per V cell-pair row
constexpr int PP  = C + 8;        // f16 per P row
constexpr int SCAN = 256;         // kv_end scan block (cells)

typedef _Float16 h2v __attribute__((ext_vector_type(2)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));   // SSA-friendly 16-B vector (HIP's uint4 is a struct)

static __device__ __forceinline__ float dot2(const h2v a, const h2v b, const float c) {
    return __builtin_amdgcn_fdot2(a, b, c, false);
}

static __device__ __forceinline__ h2v as_h2(const unsigned x) {
    return __builtin_bit_cast(h2v, x);
}

// LDS-only barrier: __syncthreads() after a global store would also drain all loads in flight (s_waitcnt vmcnt(0))
static __device__ __forceinline__ void lds_barrier() {
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
}

template <int ctrl>
static __device__ __forceinline__ float dpp_f(const float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), ctrl, 0xf, 0xf, true));
}
// reductions over the 16 lanes of a DPP row (quad_perm xor 1, xor 2, row_half_mirror, row_mirror): every lane gets it
static __device__ __forceinline__ float row16_max(float v) {
    v = fmaxf(v, dpp_f<0xB1>(v));
    v = fmaxf(v, dpp_f<0x4E>(v));
    v = fmaxf(v, dpp_f<0x141>(v));
    v = fmaxf(v, dpp_f<0x140>(v));
    return v;
}
static __device__ __forceinline__ float row16_sum(float v) {
    v += dpp_f<0xB1>(v);
    v += dpp_f<0x4E>(v);
    v += dpp_f<0x141>(v);
    v += dpp_f<0x140>(v);
    return v;
}

template <int G> struct shape {
    static_assert(R % G == 0, "G must divide 48");
    static constexpr int TQ = R/G;    // tokens per group
    static_assert(TQ*8 <= NT, "fdg_attn stages the chunk's mask with one 16-B piece per thread");
};

// 1. prep: one block per group g. mask: f16 bits, (cell c, token t) at t*m_s1 + c, m_rows rows. Q: f32, element (d, t, h)
// at t*q_s1 + h*q_s2 + d. Out: kv_end[g], q16[(g*TQ + tq)*Hq + h][D] = f16(Q*scale), zero for tokens >= T.
template <int G, bool QV4>
static __global__ void __launch_bounds__(NT) fdg_prep(
        const float * __restrict__ Q, const int64_t q_s1, const int64_t q_s2, const int Hq, const float scale,
        const unsigned short * __restrict__ mask, const int64_t m_s1, const int m_rows, const int n_kv, const int T,
        _Float16 * __restrict__ q16, int32_t * __restrict__ kv_end) {
#if FDG_DEVICE
    constexpr int TQ = shape<G>::TQ;
    const int g = blockIdx.x, tid = threadIdx.x;
    const int t0 = g*TQ;
    const int ntq = min(TQ, min(T, m_rows) - t0);
    if (blockIdx.y == 0) {
        __shared__ int s_last;
        if (tid == 0) {
            s_last = -1;
        }
        __syncthreads();
        for (int cb = ((n_kv + SCAN - 1)/SCAN - 1)*SCAN; cb >= 0; cb -= SCAN) {
            const int c = cb + tid;
            bool vis = false;
            if (c < n_kv) {
                for (int tq = 0; tq < ntq; ++tq) {
                    vis = vis || mask[(size_t) (t0 + tq)*m_s1 + c] != 0xFC00;
                }
            }
            if (vis) {
                atomicMax(&s_last, c);
            }
            __syncthreads();
            if (s_last >= 0) {
                break;
            }
        }
        if (tid == 0) {
            kv_end[g] = s_last + 1;
        }
        return;
    }
    // Thread = (4 dims d, heads h0 + 4k) of 2 tokens per pass: 2*QB float4 loads in flight
    constexpr int QB = 6;
    const int d  = (tid & 63)*4;
    const int hh = tid >> 6;
    for (int tp = 0; tp < TQ; tp += 2) {
        for (int hb = 0; hb < Hq; hb += 4*QB) {
            float4 v[2][QB];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int t = t0 + tp + u;
                const float * qi = Q + (size_t) min(t, T - 1)*q_s1 + d;
#pragma unroll
                for (int k = 0; k < QB; ++k) {
                    const int h = hb + hh + 4*k;
                    v[u][k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if (tp + u < TQ && t < T && h < Hq) {
                        const float * qp = qi + h*q_s2;
                        if constexpr (QV4) {
                            v[u][k] = *(const float4 *) qp;
                        } else {
                            v[u][k] = make_float4(qp[0], qp[1], qp[2], qp[3]);
                        }
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                if (tp + u >= TQ) {
                    continue;
                }
                _Float16 * qo = q16 + (size_t) (t0 + tp + u)*Hq*D + d;
#pragma unroll
                for (int k = 0; k < QB; ++k) {
                    const int h = hb + hh + 4*k;
                    if (h < Hq) {
                        const h2v a = {(_Float16) (v[u][k].x*scale), (_Float16) (v[u][k].y*scale)};
                        const h2v b = {(_Float16) (v[u][k].z*scale), (_Float16) (v[u][k].w*scale)};
                        *(uint2 *) (qo + h*D) = make_uint2(__builtin_bit_cast(unsigned, a), __builtin_bit_cast(unsigned, b));
                    }
                }
            }
        }
    }
#else
    (void) Q; (void) q_s1; (void) q_s2; (void) Hq; (void) scale; (void) mask; (void) m_s1; (void) m_rows; (void) n_kv;
    (void) T; (void) q16; (void) kv_end;
#endif
}

// 2. attention: grid (group, KV head, segment), 256 threads (4 wave64).
// QK role: rows rg*3 + i (rg = 4*wave + lane/16), cells cg + 16j of the chunk (cg = lane % 16).
// PV role: rows 12*wave + h (h < 12), dims 4l..4l+3: O[12][4] in f32.
// Row r of the block = token t0 + r/G, query head hk*G + r%G.
// Segment s covers cells [s*seg, min(kv_end, (s+1)*seg)), seg = pad64(ceil(kv_end/nseg)).
template <int G>
static __global__ void __launch_bounds__(NT, 2) fdg_attn(
        const _Float16 * __restrict__ q16, const char * __restrict__ Kc, const char * __restrict__ Vc,
        const unsigned k_nb1, const unsigned k_nb2, const unsigned v_nb1, const unsigned v_nb2,
        const unsigned short * __restrict__ mask, const int64_t m_s1, const int m_rows, const int32_t * __restrict__ kv_end,
        const int T, const int Hq, const int n_kvh, const int nseg,
        float * __restrict__ dst, float * __restrict__ part, float2 * __restrict__ meta) {
#if FDG_DEVICE
    constexpr int TQ = shape<G>::TQ;
    const int g = blockIdx.x, hk = blockIdx.y, s = blockIdx.z;
    const int tid = threadIdx.x, w = tid >> 6, l = tid & 63;

    const int n   = kv_end[g];
    const int seg = ((n + nseg - 1)/nseg + C - 1) & ~(C - 1);
    const int c0  = s*seg;
    const int c1  = min(n, c0 + seg);
    const size_t pm = (size_t) (g*n_kvh + hk)*nseg + s;
    if (c0 >= c1 && nseg > 1) {
        if (tid < R) {
            meta[pm*R + tid] = make_float2(-FLT_MAX/2, 0.0f); // empty segment: neutral
        }
        return;
    }

    __shared__ h2v sQ[R*KP];                            // Q quarter [48][64 dims]
    __shared__ h2v sKV[(C*KP > 8*VP) ? C*KP : 8*VP];    // K quarter [64 cells][64 dims] or V pairs [8][256 dims]
    __shared__ _Float16 sP[R*PP];                       // P [row][cell]
    __shared__ float sScale[R];
    __shared__ float sL[R];
    __shared__ unsigned short sM[TQ][C];                // chunk: f16 mask bits per (token, cell)

    const int rg = w*4 + (l >> 4);
    const int cg = l & 15;
    float m[3], ls[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        m[i]  = -FLT_MAX/2;
        ls[i] = 0.0f;
    }
    float O[12][4];
#pragma unroll
    for (int h = 0; h < 12; ++h) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            O[h][k] = 0.0f;
        }
    }

    const int t0 = g*TQ;
    const unsigned khead = (unsigned) hk*k_nb2, vhead = (unsigned) hk*v_nb2;
    // Q staging pieces (fixed): piece tid (row tid/8 < 48) and tid + 256 (a row < 48 for tid < 128); q16 rows are padded
    // to whole groups, so no clamp
    const int qo = tid & 7;
    const int qr0 = tid >> 3, qr1 = (tid + NT) >> 3;
    const unsigned qoff0 = (unsigned) (((t0 + qr0/G)*Hq + hk*G + qr0%G)*D + qo*8);
    const unsigned qoff1 = tid < 128 ? (unsigned) (((t0 + qr1/G)*Hq + hk*G + qr1%G)*D + qo*8) : 0u;
    // K staging: cells kc and kc + 32, 16-B piece ko of the 128-B quarter row
    const int kc = tid >> 3, ko = tid & 7;
    // V staging: cell pair pr of the 16-cell stage, 16-B piece dc of the 512-B row
    const int pr = tid >> 5, dc = tid & 31;
    // mask staging: token mt, 8-cell piece mc (threads < TQ*8); rows past the mask (tokens >= T) read as -inf
    const int mt = tid >> 3, mc = tid & 7;
    const bool m_ld = tid < TQ*8 && t0 + mt < min(T, m_rows);
    const unsigned short * mrow = mask + (size_t) (m_ld ? t0 + mt : 0)*m_s1 + mc*8;

    for (int cb = c0; cb < c1; cb += C) {
        float S[3][4];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                S[i][j] = 0.0f;
            }
        }
        const unsigned kof0 = (unsigned) (cb + kc)*k_nb1 + khead + (unsigned) (ko*16);
        const unsigned kof1 = kof0 + 32u*k_nb1;
#pragma unroll 1
        for (int qd = 0; qd < D/DS; ++qd) {
#ifdef FDG_NO_GMEM // bench-only: no global loads in the loop (latency-free upper bound)
            const unsigned z_ = (unsigned) (cb + qd) & 0x3u;
            const uint4 ka = make_uint4(0x3c003c00u ^ z_, kof0, 0x3c003c00u, 0x3c003c00u);
            const uint4 kb = make_uint4(0x3c003c00u ^ z_, kof1, 0x3c003c00u, 0x3c003c00u);
            const uint4 qa = make_uint4(0x3c003c00u, qoff0 ^ z_, 0x3c003c00u, 0x3c003c00u);
            uint4 qb = make_uint4(0x3c003c00u, qoff1 ^ z_, 0x3c003c00u, 0x3c003c00u);
#else
            const uint4 ka = *(const uint4 *) (Kc + (kof0 + (unsigned) (qd*DS*2)));
            const uint4 kb = *(const uint4 *) (Kc + (kof1 + (unsigned) (qd*DS*2)));
            const uint4 qa = *(const uint4 *) (q16 + (qoff0 + (unsigned) (qd*DS)));
            uint4 qb = make_uint4(0, 0, 0, 0);
            if (tid < 128) {
                qb = *(const uint4 *) (q16 + (qoff1 + (unsigned) (qd*DS)));
            }
#endif
            // the chunk's mask values, loaded with the last quarter (short live range: no registers held over the QK work)
            uint4 mv = make_uint4(0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u);
            if (qd == D/DS - 1 && m_ld) {
                mv = *(const uint4 *) (mrow + cb);
            }
            lds_barrier(); // the previous readers of sKV / sQ (and of sM at qd == 0) are done
            *(uint2 *) &sKV[kc*KP + ko*4]            = make_uint2(ka.x, ka.y);
            *(uint2 *) &sKV[kc*KP + ko*4 + 2]        = make_uint2(ka.z, ka.w);
            *(uint2 *) &sKV[(kc + 32)*KP + ko*4]     = make_uint2(kb.x, kb.y);
            *(uint2 *) &sKV[(kc + 32)*KP + ko*4 + 2] = make_uint2(kb.z, kb.w);
            *(uint2 *) &sQ[qr0*KP + qo*4]            = make_uint2(qa.x, qa.y);
            *(uint2 *) &sQ[qr0*KP + qo*4 + 2]        = make_uint2(qa.z, qa.w);
            if (tid < 128) {
                *(uint2 *) &sQ[qr1*KP + qo*4]     = make_uint2(qb.x, qb.y);
                *(uint2 *) &sQ[qr1*KP + qo*4 + 2] = make_uint2(qb.z, qb.w);
            }
            if (qd == D/DS - 1 && tid < TQ*8) {
                *(uint4 *) &sM[mt][mc*8] = mv;
            }
            lds_barrier();
            // 4-dim steps (b64 LDS operands): 3 Q + 4 K operands, 24 dot2
#ifndef FDG_SKIP_QK
            FDG_PRAGMA(unroll FDG_QK_UNROLL)
            for (int d4 = 0; d4 < DS/4; ++d4) {
                uint2 qx[3], kx[4];
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    qx[i] = *(const uint2 *) &sQ[(rg*3 + i)*KP + d4*2];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    kx[j] = *(const uint2 *) &sKV[(cg + 16*j)*KP + d4*2];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int i = 0; i < 3; ++i) {
                        S[i][j] = dot2(as_h2(qx[i].x), as_h2(kx[j].x), S[i][j]);
                        S[i][j] = dot2(as_h2(qx[i].y), as_h2(kx[j].y), S[i][j]);
                    }
                }
            }
#endif
        }

        // online softmax: + the token's exact f16 mask value; P (f16) -> sP, per-row rescale -> sScale
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int row = rg*3 + i;
            const int tq  = row/G;
            float mx = -FLT_MAX/2;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                S[i][j] += (float) __builtin_bit_cast(_Float16, sM[tq][cg + 16*j]);
                mx = fmaxf(mx, S[i][j]);
            }
            mx = row16_max(mx);
            const float mnew = fmaxf(m[i], mx);
            const float sc   = __expf(m[i] - mnew);
            m[i] = mnew;
            float ps = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const _Float16 p = (_Float16) __expf(S[i][j] - mnew);
                ps += (float) p; // normalize by the weights actually applied
                sP[row*PP + cg + 16*j] = p;
            }
            ls[i] = ls[i]*sc + ps;
            if (cg == 0) {
                sScale[row] = sc;
            }
        }

        // PV: 4 stages of 16 cells, V as half2(V[c][d], V[c+1][d]) cell pairs
        const unsigned vof = (unsigned) (cb + 2*pr)*v_nb1 + vhead;
#pragma unroll 1
        for (int vs = 0; vs < C/16; ++vs) {
#ifdef FDG_NO_GMEM
            const uint4 va = make_uint4(0x3c003c00u ^ (unsigned) vs, vof, 0x3c003c00u, (unsigned) cb);
            const uint4 vb = make_uint4(0x3c003c00u, vof ^ (unsigned) vs, 0x3c003c00u, (unsigned) cb);
#else
            const uint4 va = *((const uint4 *) (Vc + (vof + (unsigned) (vs*16)*v_nb1)) + dc);
            const uint4 vb = *((const uint4 *) (Vc + (vof + (unsigned) (vs*16 + 1)*v_nb1)) + dc);
#endif
            lds_barrier();
            {
                const unsigned a[4] = {va.x, va.y, va.z, va.w}, b[4] = {vb.x, vb.y, vb.z, vb.w};
                unsigned o[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    o[2*k]     = __builtin_amdgcn_perm(b[k], a[k], 0x05040100u);
                    o[2*k + 1] = __builtin_amdgcn_perm(b[k], a[k], 0x07060302u);
                }
                uint4 * vd = (uint4 *) &sKV[pr*VP + dc*8];
                vd[0] = make_uint4(o[0], o[1], o[2], o[3]);
                vd[1] = make_uint4(o[4], o[5], o[6], o[7]);
            }
            lds_barrier();
            if (vs == 0) {
#pragma unroll
                for (int h = 0; h < 12; ++h) {
                    const float sc = sScale[w*12 + h];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        O[h][k] *= sc;
                    }
                }
            }
#ifndef FDG_SKIP_PV
            FDG_PRAGMA(unroll FDG_PV_UNROLL)
            for (int p = 0; p < 8; ++p) {
                const uint4 vv = *(const uint4 *) &sKV[p*VP + l*4];
                const h2v v4[4] = {as_h2(vv.x), as_h2(vv.y), as_h2(vv.z), as_h2(vv.w)};
#pragma unroll
                for (int h = 0; h < 12; ++h) {
                    const h2v pp = *(const h2v *) &sP[(w*12 + h)*PP + vs*16 + 2*p];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        O[h][k] = dot2(pp, v4[k], O[h][k]);
                    }
                }
            }
#endif
        }
    }

    lds_barrier(); // the last chunk's readers of sScale are done
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float sum = row16_sum(ls[i]);
        if (cg == 0) {
            sL[rg*3 + i]     = sum;
            sScale[rg*3 + i] = m[i];
        }
    }
    lds_barrier();
    if (nseg == 1) {
#pragma unroll
        for (int h = 0; h < 12; ++h) {
            const int row = w*12 + h;
            const int t   = t0 + row/G;
            if (t < T) {
                const float lsum = sL[row];
                const float inv  = lsum > 0.0f ? 1.0f/lsum : 0.0f;
                *(float4 *) &dst[((size_t) t*Hq + hk*G + row%G)*D + 4*l] =
                    make_float4(O[h][0]*inv, O[h][1]*inv, O[h][2]*inv, O[h][3]*inv);
            }
        }
    } else {
        float * pt = part + pm*R*D;
#pragma unroll
        for (int h = 0; h < 12; ++h) {
            *(float4 *) &pt[(w*12 + h)*D + 4*l] = make_float4(O[h][0], O[h][1], O[h][2], O[h][3]);
        }
        if (l < 12) {
            meta[pm*R + w*12 + l] = make_float2(sScale[w*12 + l], sL[w*12 + l]);
        }
    }
#else
    (void) q16; (void) Kc; (void) Vc; (void) k_nb1; (void) k_nb2; (void) v_nb1; (void) v_nb2; (void) mask; (void) m_s1;
    (void) m_rows; (void) kv_end; (void) T; (void) Hq; (void) n_kvh; (void) nseg; (void) dst; (void) part; (void) meta;
#endif
}

// 2b. attention, one block per CU (occupancy 1: 256 VGPRs) with all of Q resident in LDS and every global load issued one
// compute phase ahead (software pipelined): K quarter qd+1 while QK of quarter qd runs, V stage vs+1 during PV stage vs,
// the next chunk's K quarter 0 and mask during the last PV stage. 128-cell chunks.
// QK role: rows rg*3 + i (rg = tid/16), cells 32j + 2cg + e (cg = tid % 16, j < 4, e < 2): 3 x 8 tile, 11 b64 LDS reads
// per 48 dot2. PV role: rows 12*wave + h, dims 4l..4l+3 (as fdg_attn). grid (group, KV head, segment).
constexpr int C1  = 128;          // cells per chunk
constexpr int QP  = D/2 + 4;      // half2 per resident Q row (16 B pad)
constexpr int PP1 = C1 + 8;       // f16 per P row

template <int G>
static __global__ void __launch_bounds__(NT, 1) __attribute__((amdgpu_waves_per_eu(1, 1))) fdg_attn1(
        const _Float16 * __restrict__ q16, const char * __restrict__ Kc, const char * __restrict__ Vc,
        const unsigned k_nb1, const unsigned k_nb2, const unsigned v_nb1, const unsigned v_nb2,
        const unsigned short * __restrict__ mask, const int64_t m_s1, const int m_rows, const int32_t * __restrict__ kv_end,
        const int T, const int Hq, const int n_kvh, const int nseg,
        float * __restrict__ dst, float * __restrict__ part, float2 * __restrict__ meta) {
#if FDG_DEVICE
    constexpr int TQ = shape<G>::TQ;
    const int g = blockIdx.x, hk = blockIdx.y, s = blockIdx.z;
    const int tid = threadIdx.x, w = tid >> 6, l = tid & 63;

    const int n   = kv_end[g];
    const int seg = ((n + nseg - 1)/nseg + C1 - 1) & ~(C1 - 1);
    const int c0  = s*seg;
    const int c1  = min(n, c0 + seg);
    const size_t pm = (size_t) (g*n_kvh + hk)*nseg + s;
    if (c0 >= c1 && nseg > 1) {
        if (tid < R) {
            meta[pm*R + tid] = make_float2(-FLT_MAX/2, 0.0f); // empty segment: neutral
        }
        return;
    }

    __shared__ h2v sQ[R*QP];                              // all of Q [48][256 dims]
    __shared__ h2v sKV[(C1*KP > 8*VP) ? C1*KP : 8*VP];    // K quarter [128 cells][64 dims] or V pairs [8][256 dims]
    __shared__ _Float16 sP[R*PP1];                        // P [row][cell]
    __shared__ float sScale[R];
    __shared__ float sL[R];
    __shared__ unsigned short sM[TQ][C1];                 // chunk: f16 mask bits per (token, cell)

    const int t0 = g*TQ;
    const unsigned khead = (unsigned) hk*k_nb2, vhead = (unsigned) hk*v_nb2;

    // resident Q: 48 rows x 32 pieces of 16 B (q16 rows are padded to whole groups); the loop's first barrier orders it
    for (int e = tid; e < R*32; e += NT) {
        const int row = e >> 5, pc = e & 31;
        const uint4 q = *(const uint4 *) (q16 + (size_t) ((t0 + row/G)*Hq + hk*G + row%G)*D + pc*8);
        *(uint4 *) &sQ[row*QP + pc*4] = q;
    }

    const int rg = tid >> 4;
    const int cg = tid & 15;
    // K staging: cells kc + 32k (k < 4), 16-B piece ko of the 128-B quarter row
    const int kc = tid >> 3, ko = tid & 7;
    const unsigned kof = (unsigned) kc*k_nb1 + khead + (unsigned) (ko*16);
    // V staging: cell pair pr of the 16-cell stage, 16-B piece dc of the 512-B row
    const int pr = tid >> 5, dc = tid & 31;
    const unsigned vof = (unsigned) (2*pr)*v_nb1 + vhead + (unsigned) (dc*16);
    // mask staging: token mt, 8-cell piece mc (threads < TQ*16); rows past the mask (tokens >= T) read as -inf
    const int mt = tid >> 4, mc = tid & 15;
    const bool m_ld = tid < TQ*16 && t0 + mt < min(T, m_rows);
    const unsigned mof = m_ld ? (unsigned) ((t0 + mt)*m_s1 + mc*8) : 0u;

    // prefetch registers (macros, not lambdas: arrays captured by reference ended up in scratch)
    v4u kp0, kp1, kp2, kp3, vp0, vp1, mv;
#define FDG_LOAD_K(cb_, qd_) do { \
        const unsigned o_ = (unsigned) (cb_)*k_nb1 + kof + (unsigned) ((qd_)*DS*2); \
        kp0 = *(const v4u *) (Kc + o_); \
        kp1 = *(const v4u *) (Kc + (o_ + 32u*k_nb1)); \
        kp2 = *(const v4u *) (Kc + (o_ + 64u*k_nb1)); \
        kp3 = *(const v4u *) (Kc + (o_ + 96u*k_nb1)); \
    } while (0)
#define FDG_LOAD_V(cb_, vs_) do { \
        const unsigned o_ = (unsigned) ((cb_) + (vs_)*16)*v_nb1 + vof; \
        vp0 = *(const v4u *) (Vc + o_); \
        vp1 = *(const v4u *) (Vc + (o_ + v_nb1)); \
    } while (0)
#define FDG_LOAD_M(cb_) do { \
        mv = (v4u) {0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u, 0xFC00FC00u}; \
        if (m_ld) { \
            mv = *(const v4u *) (mask + (mof + (unsigned) (cb_))); \
        } \
    } while (0)

    float m[3], ls[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        m[i]  = -FLT_MAX/2;
        ls[i] = 0.0f;
    }
    float O[12][4];
#pragma unroll
    for (int h = 0; h < 12; ++h) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            O[h][k] = 0.0f;
        }
    }

    if (c0 < c1) {
        FDG_LOAD_K(c0, 0);
        FDG_LOAD_M(c0);
    }
    for (int cb = c0; cb < c1; cb += C1) {
        float S[3][8];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                S[i][j] = 0.0f;
            }
        }
#pragma unroll 1
        for (int qd = 0; qd < D/DS; ++qd) {
            lds_barrier(); // the previous readers of sKV (and of sM at qd == 0) are done
            *(uint2 *) &sKV[(kc +  0)*KP + ko*4]     = make_uint2(kp0.x, kp0.y);
            *(uint2 *) &sKV[(kc +  0)*KP + ko*4 + 2] = make_uint2(kp0.z, kp0.w);
            *(uint2 *) &sKV[(kc + 32)*KP + ko*4]     = make_uint2(kp1.x, kp1.y);
            *(uint2 *) &sKV[(kc + 32)*KP + ko*4 + 2] = make_uint2(kp1.z, kp1.w);
            *(uint2 *) &sKV[(kc + 64)*KP + ko*4]     = make_uint2(kp2.x, kp2.y);
            *(uint2 *) &sKV[(kc + 64)*KP + ko*4 + 2] = make_uint2(kp2.z, kp2.w);
            *(uint2 *) &sKV[(kc + 96)*KP + ko*4]     = make_uint2(kp3.x, kp3.y);
            *(uint2 *) &sKV[(kc + 96)*KP + ko*4 + 2] = make_uint2(kp3.z, kp3.w);
            if (qd == 0 && tid < TQ*16) {
                *(v4u *) &sM[mt][mc*8] = mv;
            }
            lds_barrier();
            if (qd < D/DS - 1) {
                FDG_LOAD_K(cb, qd + 1);
            } else {
                FDG_LOAD_V(cb, 0);
            }
            FDG_PRAGMA(unroll FDG_QK_UNROLL)
            for (int d4 = 0; d4 < DS/4; ++d4) {
                uint2 qx[3], kx[8];
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    qx[i] = *(const uint2 *) &sQ[(rg*3 + i)*QP + qd*(DS/2) + d4*2];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        kx[2*j + e] = *(const uint2 *) &sKV[(32*j + 2*cg + e)*KP + d4*2];
                    }
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
#pragma unroll
                    for (int i = 0; i < 3; ++i) {
                        S[i][j] = dot2(as_h2(qx[i].x), as_h2(kx[j].x), S[i][j]);
                        S[i][j] = dot2(as_h2(qx[i].y), as_h2(kx[j].y), S[i][j]);
                    }
                }
            }
        }

        // online softmax: + the token's exact f16 mask value; P (f16 cell pairs) -> sP, per-row rescale -> sScale
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int row = rg*3 + i;
            const int tq  = row/G;
            float mx = -FLT_MAX/2;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const h2v mk = *(const h2v *) &sM[tq][32*j + 2*cg];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    S[i][2*j + e] += (float) mk[e];
                    mx = fmaxf(mx, S[i][2*j + e]);
                }
            }
            mx = row16_max(mx);
            const float mnew = fmaxf(m[i], mx);
            const float sc   = __expf(m[i] - mnew);
            m[i] = mnew;
            float ps = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                h2v pj;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    pj[e] = (_Float16) __expf(S[i][2*j + e] - mnew);
                    ps += (float) pj[e]; // normalize by the weights actually applied
                }
                *(h2v *) &sP[row*PP1 + 32*j + 2*cg] = pj;
            }
            ls[i] = ls[i]*sc + ps;
            if (cg == 0) {
                sScale[row] = sc;
            }
        }

        // PV: 8 stages of 16 cells, V as half2(V[c][d], V[c+1][d]) cell pairs
#pragma unroll 1
        for (int vs = 0; vs < C1/16; ++vs) {
            lds_barrier();
            {
                const unsigned a[4] = {vp0.x, vp0.y, vp0.z, vp0.w}, b[4] = {vp1.x, vp1.y, vp1.z, vp1.w};
                unsigned o[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    o[2*k]     = __builtin_amdgcn_perm(b[k], a[k], 0x05040100u);
                    o[2*k + 1] = __builtin_amdgcn_perm(b[k], a[k], 0x07060302u);
                }
                uint4 * vd = (uint4 *) &sKV[pr*VP + dc*8];
                vd[0] = make_uint4(o[0], o[1], o[2], o[3]);
                vd[1] = make_uint4(o[4], o[5], o[6], o[7]);
            }
            lds_barrier();
            if (vs < C1/16 - 1) {
                FDG_LOAD_V(cb, vs + 1);
            } else if (cb + C1 < c1) {
                FDG_LOAD_K(cb + C1, 0);
                FDG_LOAD_M(cb + C1);
            }
            if (vs == 0) {
#pragma unroll
                for (int h = 0; h < 12; ++h) {
                    const float sc = sScale[w*12 + h];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        O[h][k] *= sc;
                    }
                }
            }
            FDG_PRAGMA(unroll FDG_PV_UNROLL)
            for (int p = 0; p < 8; ++p) {
                const uint4 vv = *(const uint4 *) &sKV[p*VP + l*4];
                const h2v v4[4] = {as_h2(vv.x), as_h2(vv.y), as_h2(vv.z), as_h2(vv.w)};
#pragma unroll
                for (int h = 0; h < 12; ++h) {
                    const h2v pp = *(const h2v *) &sP[(w*12 + h)*PP1 + vs*16 + 2*p];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        O[h][k] = dot2(pp, v4[k], O[h][k]);
                    }
                }
            }
        }
    }

    lds_barrier(); // the last chunk's readers of sScale are done
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float sum = row16_sum(ls[i]);
        if (cg == 0) {
            sL[rg*3 + i]     = sum;
            sScale[rg*3 + i] = m[i];
        }
    }
    lds_barrier();
    if (nseg == 1) {
#pragma unroll
        for (int h = 0; h < 12; ++h) {
            const int row = w*12 + h;
            const int t   = t0 + row/G;
            if (t < T) {
                const float lsum = sL[row];
                const float inv  = lsum > 0.0f ? 1.0f/lsum : 0.0f;
                *(float4 *) &dst[((size_t) t*Hq + hk*G + row%G)*D + 4*l] =
                    make_float4(O[h][0]*inv, O[h][1]*inv, O[h][2]*inv, O[h][3]*inv);
            }
        }
    } else {
        float * pt = part + pm*R*D;
#pragma unroll
        for (int h = 0; h < 12; ++h) {
            *(float4 *) &pt[(w*12 + h)*D + 4*l] = make_float4(O[h][0], O[h][1], O[h][2], O[h][3]);
        }
        if (l < 12) {
            meta[pm*R + w*12 + l] = make_float2(sScale[w*12 + l], sL[w*12 + l]);
        }
    }
#else
    (void) q16; (void) Kc; (void) Vc; (void) k_nb1; (void) k_nb2; (void) v_nb1; (void) v_nb2; (void) mask; (void) m_s1;
    (void) m_rows; (void) kv_end; (void) T; (void) Hq; (void) n_kvh; (void) nseg; (void) dst; (void) part; (void) meta;
#endif
}

#undef FDG_LOAD_K
#undef FDG_LOAD_V
#undef FDG_LOAD_M

// 3. segment merge: one wave per output row r = t*Hq + hq (4 rows per block), lane = 4 dims (float4); segments in fixed
// order (deterministic); l == 0 partials (empty segment, or fully masked for this row) are skipped; a row with no
// visible cell gives 0.
template <int G>
static __global__ void __launch_bounds__(NT) fdg_combine(
        const float * __restrict__ part, const float2 * __restrict__ meta, float * __restrict__ dst,
        const int nrows, const int Hq, const int n_kvh, const int nseg) {
#if FDG_DEVICE
    constexpr int TQ = shape<G>::TQ;
    const int r = __builtin_amdgcn_readfirstlane(blockIdx.x*(NT/64) + (threadIdx.x >> 6));
    const int l = threadIdx.x & 63;
    if (r >= nrows) {
        return;
    }
    const int t = r / Hq, hq = r - t*Hq;
    const int g = t / TQ, hk = hq / G;
    const int row = (t % TQ)*G + (hq - hk*G);
    const size_t b = (size_t) (g*n_kvh + hk)*nseg;
    const float2 * mp = meta + b*R + row;
    const float4 * pp = (const float4 *) (part + (b*R + row)*D) + l;
    float mx = -FLT_MAX/2;
    for (int s = 0; s < nseg; ++s) {
        mx = fmaxf(mx, mp[s*R].x);
    }
    float4 num = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float den = 0.0f;
    for (int s0 = 0; s0 < nseg; s0 += 4) {
        float4 v[4];
        float  f[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int s = s0 + u;
            const float2 ms = s < nseg ? mp[s*R] : make_float2(-FLT_MAX/2, 0.0f);
            f[u] = __expf(ms.x - mx);
            den += f[u]*ms.y;
            f[u] = ms.y > 0.0f ? f[u] : 0.0f;
            v[u] = ms.y > 0.0f ? pp[(size_t) s*(R*D/4)] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            num.x += f[u]*v[u].x;
            num.y += f[u]*v[u].y;
            num.z += f[u]*v[u].z;
            num.w += f[u]*v[u].w;
        }
    }
    const float inv = den > 0.0f ? 1.0f/den : 0.0f;
    *(float4 *) &dst[(size_t) r*D + 4*l] = make_float4(num.x*inv, num.y*inv, num.z*inv, num.w*inv);
#else
    (void) part; (void) meta; (void) dst; (void) nrows; (void) Hq; (void) n_kvh; (void) nseg;
#endif
}

struct params {
    const unsigned short * mask; int64_t m_s1; int m_rows;    // KQ mask (f16 bits)
    const float * Q; int64_t q_s1, q_s2; int q_vec4;          // f32, strides in elements; q_vec4: 16-B aligned rows and heads
    const char * K; const char * V;                           // f16, cell/head strides in bytes
    unsigned k_nb1, k_nb2, v_nb1, v_nb2;
    int T, n_kv, Hq, n_kvh, G; float scale;
    int var;                                                  // 0: fdg_attn (2 blocks/CU), 1: fdg_attn1 (1 block/CU)
    _Float16 * q16; int32_t * kv_end;                         // pool
    float * part; float2 * meta; int nseg;                    // pool (nseg > 1)
    float * dst;
};

static int n_groups(const int T, const int G) {
    const int TQ = R/G;
    return (T + TQ - 1)/TQ;
}

// segments per (group, KV head): the count minimizing rounds of resident blocks x (cells per segment + ~64 cells of fixed
// per-segment cost); slots = resident blocks (2 per CU for fdg_attn, 1 for fdg_attn1). Measured at T = 512, 2 KV heads, GQA 6,
// causal (MI50, 60 CUs), best nseg: 512 cells 3 (233 us vs 275 at 1), 1K 3, 1.5K 3-4, 4K 6, 4.6K 6, 66K 15 (28.4 ms vs
// 29.1 at 6). A shape-only rule (HIP graph capture).
static int nseg_auto(const int T, const int n_kv, const int n_kvh, const int G, const int slots = 120) {
    const int b1 = std::max(1, n_groups(T, G)*n_kvh);
    int best = 1;
    double best_cost = 1e300;
    for (int ns = 1; ns <= 32; ++ns) {
        const double rounds = (double) ((b1*ns + slots - 1)/slots);
        const double cost   = rounds*((double) n_kv/ns + 64.0);
        if (cost < best_cost*0.999) {
            best_cost = cost;
            best = ns;
        }
    }
    return best;
}

static size_t part_elems(const int ng, const int n_kvh, const int nseg) {
    return (size_t) ng*n_kvh*nseg*R*D;
}

template <int G>
static void launch_g(const params & p, hipStream_t st, const int which) {
    const int ng = n_groups(p.T, G);
    if (which & 1) {
        if (p.q_vec4) {
            fdg_prep<G, true><<<dim3(ng, 2), NT, 0, st>>>(p.Q, p.q_s1, p.q_s2, p.Hq, p.scale, p.mask, p.m_s1, p.m_rows,
                p.n_kv, p.T, p.q16, p.kv_end);
        } else {
            fdg_prep<G, false><<<dim3(ng, 2), NT, 0, st>>>(p.Q, p.q_s1, p.q_s2, p.Hq, p.scale, p.mask, p.m_s1, p.m_rows,
                p.n_kv, p.T, p.q16, p.kv_end);
        }
    }
    if ((which & 2) && p.var == 0) {
        fdg_attn<G><<<dim3(ng, p.n_kvh, p.nseg), NT, 0, st>>>(p.q16, p.K, p.V, p.k_nb1, p.k_nb2, p.v_nb1, p.v_nb2,
            p.mask, p.m_s1, p.m_rows, p.kv_end, p.T, p.Hq, p.n_kvh, p.nseg, p.dst, p.part, p.meta);
    }
    if constexpr (G >= 3) { // fdg_attn1 stages the mask with one 16-B piece per thread: TQ*16 <= 256
        if ((which & 2) && p.var == 1) {
            fdg_attn1<G><<<dim3(ng, p.n_kvh, p.nseg), NT, 0, st>>>(p.q16, p.K, p.V, p.k_nb1, p.k_nb2, p.v_nb1, p.v_nb2,
                p.mask, p.m_s1, p.m_rows, p.kv_end, p.T, p.Hq, p.n_kvh, p.nseg, p.dst, p.part, p.meta);
        }
    }
    if ((which & 4) && p.nseg > 1) {
        const int nrows = p.T*p.Hq;
        fdg_combine<G><<<(nrows + NT/64 - 1)/(NT/64), NT, 0, st>>>(p.part, p.meta, p.dst, nrows, p.Hq, p.n_kvh, p.nseg);
    }
}

static bool supported_g(const int G, const int var = 0) {
    return G == 2 || G == 3 || G == 4 || G == 6 || G == 8 || G == 12 ? (var == 0 || G >= 3) : false;
}

// which: 1 prep, 2 attention, 4 combine (the bench times them one by one)
static void launch(const params & p, hipStream_t st, const int which = 7) {
    switch (p.G) {
        case  2: launch_g< 2>(p, st, which); break;
        case  3: launch_g< 3>(p, st, which); break;
        case  4: launch_g< 4>(p, st, which); break;
        case  6: launch_g< 6>(p, st, which); break;
        case  8: launch_g< 8>(p, st, which); break;
        case 12: launch_g<12>(p, st, which); break;
        default: break;
    }
}

} // namespace fdg

#endif // GGML_USE_HIP || FDG_STANDALONE

#ifndef FDG_STANDALONE

#ifdef GGML_USE_HIP

static int fdg_env(const char * name, const int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

// every condition of the dense GCN path; returns nullptr when supported, else the reason
static const char * fdg_unsupported(const ggml_tensor * dst, const int min_t, const int var) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    if (ggml_is_empty(dst) || ggml_is_empty(Q) || ggml_is_empty(K) || ggml_is_empty(V)) {
        return "empty attention slice";
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (dst->src[4] != nullptr || max_bias != 0.0f || softcap != 0.0f) {
        return "sinks/alibi/softcap";
    }
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return "dst type/layout";
    }
    if (Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || Q->ne[0] != fdg::D || Q->ne[3] != 1) {
        return "Q type/shape";
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || K->ne[0] != fdg::D || V->ne[0] != fdg::D || K->ne[3] != 1 ||
            V->ne[3] != 1 || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] || Q->ne[2] % K->ne[2] != 0) {
        return "K/V type/shape";
    }
    const int G = (int) (Q->ne[2]/K->ne[2]);
    if (!fdg::supported_g(G, var)) {
        return "GQA ratio";
    }
    if (K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || (K->nb[1] | K->nb[2] | V->nb[1] | V->nb[2]) % 16 != 0 ||
            ((uintptr_t) K->data | (uintptr_t) V->data) % 16 != 0) {
        return "K/V alignment";
    }
    // chunks run up to the next chunk boundary past the last visible cell (masked there)
    if (K->ne[1] % (var == 1 ? fdg::C1 : fdg::C) != 0) {
        return "n_kv not a multiple of the chunk";
    }
    if (mask == nullptr || mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) || mask->ne[0] < K->ne[1] ||
            mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[1] % 16 != 0 || (uintptr_t) mask->data % 16 != 0) {
        return "mask type/shape/alignment";
    }
    if (Q->ne[1] < min_t) {
        return "T < MIN_T";
    }
    // 32-bit offsets
    const int64_t k_max = (K->ne[1] - 1)*(int64_t) K->nb[1] + (K->ne[2] - 1)*(int64_t) K->nb[2] + fdg::D*(int64_t) sizeof(half);
    const int64_t v_max = (V->ne[1] - 1)*(int64_t) V->nb[1] + (V->ne[2] - 1)*(int64_t) V->nb[2] + fdg::D*(int64_t) sizeof(half);
    const int64_t m_max = mask->ne[1]*(mask->nb[1]/(int64_t) sizeof(half));
    const int64_t q_max = (Q->ne[1] + fdg::R)*Q->ne[2]*(int64_t) fdg::D;
    if (k_max > INT32_MAX || v_max > INT32_MAX || m_max > INT32_MAX || q_max > INT32_MAX) {
        return "too large for 32-bit offsets";
    }
    return nullptr;
}

bool ggml_cuda_flash_attn_ext_dense_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const int mode   = fdg_env("GGML_CUDA_FA_DENSE_GCN",       1);
    static const int min_t  = fdg_env("GGML_CUDA_FA_DENSE_GCN_MIN_T", 64);
    static const int nseg_e = fdg_env("GGML_CUDA_FA_DENSE_GCN_NSEG",  0);
    static const int var    = fdg_env("GGML_CUDA_FA_DENSE_GCN_VAR",   0);
    static const int stats  = fdg_env("GGML_CUDA_FA_DENSE_GCN_STATS", 0);
    if (!mode || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_VEGA20 || ctx.curr_stream_no != 0) {
        return false;
    }
    const char * why = fdg_unsupported(dst, min_t, var);
    if (why) {
        if (stats) {
            fprintf(stderr, "fdg: dev %d %s: fallback: %s\n", ctx.device, dst->name, why);
        }
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    fdg::params p = {};
    p.mask   = (const unsigned short *) mask->data;
    p.m_s1   = mask->nb[1]/sizeof(half);
    p.m_rows = (int) mask->ne[1];
    p.Q      = (const float *) Q->data;
    p.q_s1   = Q->nb[1]/sizeof(float);
    p.q_s2   = Q->nb[2]/sizeof(float);
    p.q_vec4 = ((uintptr_t) Q->data % 16 == 0 && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0) ? 1 : 0;
    p.K      = (const char *) K->data;
    p.V      = (const char *) V->data;
    p.k_nb1  = (unsigned) K->nb[1];
    p.k_nb2  = (unsigned) K->nb[2];
    p.v_nb1  = (unsigned) V->nb[1];
    p.v_nb2  = (unsigned) V->nb[2];
    p.T      = (int) Q->ne[1];
    p.n_kv   = (int) K->ne[1];
    p.Hq     = (int) Q->ne[2];
    p.n_kvh  = (int) K->ne[2];
    p.G      = p.Hq/p.n_kvh;
    p.var    = var;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    const int ng    = fdg::n_groups(p.T, p.G);
    const int slots = (var == 1 ? 1 : 2)*ggml_cuda_info().devices[ctx.device].nsm;
    p.nseg = nseg_e > 0 ? std::min(nseg_e, 64) : fdg::nseg_auto(p.T, p.n_kv, p.n_kvh, p.G, slots);
    p.dst  = (float *) dst->data;

    ggml_cuda_pool_alloc<half>    q16   (ctx.pool(), (size_t) ng*(fdg::R/p.G)*p.Hq*fdg::D);
    ggml_cuda_pool_alloc<int32_t> kv_end(ctx.pool(), ng);
    ggml_cuda_pool_alloc<float>   part  (ctx.pool());
    ggml_cuda_pool_alloc<float2>  meta  (ctx.pool());
    if (p.nseg > 1) {
        part.alloc(fdg::part_elems(ng, p.n_kvh, p.nseg));
        meta.alloc((size_t) ng*p.n_kvh*p.nseg*fdg::R);
    }
    p.q16    = (_Float16 *) q16.get();
    p.kv_end = kv_end.get();
    p.part   = part.ptr;
    p.meta   = meta.ptr;

    fdg::launch(p, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    if (stats) {
        fprintf(stderr, "fdg: dev %d %s: T %d n_kv %d Hq %d n_kvh %d G %d nseg %d var %d\n", ctx.device, dst->name, p.T, p.n_kv,
            p.Hq, p.n_kvh, p.G, p.nseg, var);
    }
    return true;
}

#else

bool ggml_cuda_flash_attn_ext_dense_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

#endif // GGML_USE_HIP

#endif // !FDG_STANDALONE
