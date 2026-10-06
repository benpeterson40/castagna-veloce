// Exact sparse QSA prefill attention for gfx906 (MI50), plan: sparse-plan.md (block-sparse-new).
//
// The qwen4exp indexer builds CONT(TOP_K) -> FILL(kq_mask, -inf) -> SET_ROWS(zeros at the selected cells) -> ADD(kq_mask)
// -> FLASH_ATTN_EXT: every query row sees mask[t][j] at its selected cells (2051 of n_kv) and -inf elsewhere. The dense
// attention still streams all n_kv K/V rows. Here the whole chain runs as three kernels, the [n_kv, T] mask is never built:
//   1. qsa_build   (one block per 4-token group): the sorted union of the group's selections (LDS nibble bitmap over
//                  32K-cell windows, ballot-prefix compaction), each entry with the exact f16 mask value of every token of
//                  the group (-inf where that token did not select the cell), and Q*scale in f16. Writes pool memory only.
//   2. qsa_attn    (group, KV head, segment of the union): 48 rows (4 tokens x 12 query heads) per block, 64-cell chunks,
//                  QK in f32 (v_dot2_f32_f16), online softmax, P in f16, PV accumulated in f32.
//   3. qsa_combine (nseg > 1): merges the segments in fixed order (deterministic). The only writer of dst (or qsa_attn
//                  itself when nseg == 1).
// Semantics equal hc_qsa_mask + the dense FA (the union with per-token mask values is the same visible set and values);
// only the precision (f32 accumulation instead of half2 VKQ) and the summation order differ. A row with no visible cell
// gives 0 (like the CPU backend) instead of NaN.
//
// Aliasing: FA dst may lie over the TOP_K result, the FILL/SET_ROWS buffer and the KQ mask (all dead after the chain in
// the unfused order). The builder reads those first and writes only pool memory; attention reads Q (through the builder's
// copy), K, V and pool memory; dst is written last. All pool memory lives within one call (shape-only launch geometry, so
// HIP graph capture is safe).
//
// The file also builds standalone (QSA_STANDALONE, ~/mi50-engine/tools/qsa/qsa_bench.hip includes it), so the bench runs
// exactly these kernels and the same launcher.

#ifdef QSA_STANDALONE
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include "fattn-qsa-gcn.cuh"
#include "hc-persist.cuh"
#endif

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(GGML_USE_HIP) || defined(QSA_STANDALONE)

// device code for gfx906 only (v_dot2_f32_f16, wave64 DPP); other targets compile empty kernels
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__)
#define QSA_DEVICE 1
#else
#define QSA_DEVICE 0
#endif

// tuning knobs (clang 21 -O3, gfx906; every choice keeps qsa_attn at <= 128 VGPRs with no scratch):
// QK unroll 1/2/4: 1.904/1.755/1.946 ms (4 spills 72 B), PV unroll 2: -1%, next-chunk list prefetch: -1%, expf: +2%
// (8448 cells, U4 2694, 24 heads; see ~/mi50-engine/tools/qsa)
#ifndef QSA_QK_UNROLL
#define QSA_QK_UNROLL 2
#endif
#ifndef QSA_PV_UNROLL
#define QSA_PV_UNROLL 2
#endif
#ifndef QSA_PREFETCH_LIST
#define QSA_PREFETCH_LIST 1
#endif
#ifndef QSA_EXP
#define QSA_EXP __expf
#endif
#define QSA_PSTR(x) #x
#define QSA_PRAGMA(x) _Pragma(QSA_PSTR(x))

namespace qsa {

constexpr int D   = 256;          // head size (K and V)
constexpr int G   = 12;           // query heads per KV head
constexpr int TQ  = 4;            // query tokens per group
constexpr int NT  = 256;          // threads per block
constexpr int R   = TQ*G;         // 48 rows per block
constexpr int C   = 64;           // cells per chunk
constexpr int DS  = 64;           // dims per K stage
constexpr int KP  = DS/2 + 4;     // half2 per K/Q row in LDS (16 B pad)
constexpr int VP  = D + 4;        // half2 per V cell-pair row
constexpr int PP  = C + 8;        // f16 per P row
constexpr int WIN = 32768;        // builder window (cells): 16 KB nibble bitmap

typedef _Float16 h2v __attribute__((ext_vector_type(2)));

#ifdef QSA_TS // bench-only phase stamps (s_memrealtime, 25 MHz on gfx906)
__device__ unsigned long long qsa_ts[1024][8];
#define QSA_STAMP(k) if (threadIdx.x == 0 && blockIdx.x < 1024) { qsa_ts[blockIdx.x][k] = __builtin_amdgcn_s_memrealtime(); }
#else
#define QSA_STAMP(k)
#endif

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

// 1. union builder: one block per 4-token group g.
// topk: [T][width] cells (row stride tk_s1 elements); mask: f16 bits [T][n_kv] (row stride m_s1); Q: f32, element
// (d, t, h) at t*q_s1 + h*q_s2 + d.
// Out: list[g][cap] sorted cells, mval[g][cap] 4 x f16 (token tq in bits 16*tq of the uint2 pair), count[g],
// q16[(g*4 + tq)*Hq + h][D] = f16(Q*scale), zero for tokens >= T (grid.y = 1 blocks). count <= min(n_kv, 4*width) = cap.
// QV4: Q rows and head strides 16-B aligned (one dwordx4 per float4; a runtime choice of the two paths compiled to 4 dword
// loads per float4). clang 21 also turns the QV4 = false form into dwordx4 (legal at 4-B alignment on gfx906): today both
// instantiations are the same code, the template keeps the aligned path explicit
template <bool QV4>
static __global__ void __launch_bounds__(NT) qsa_build(
        const int32_t * __restrict__ topk, const int64_t tk_s1, const int width,
        const unsigned short * __restrict__ mask, const int64_t m_s1, const int n_kv, const int T,
        const float * __restrict__ Q, const int64_t q_s1, const int64_t q_s2, const int Hq, const float scale,
        int32_t * __restrict__ list, uint2 * __restrict__ mval, int32_t * __restrict__ count, _Float16 * __restrict__ q16,
        const int cap) {
#if QSA_DEVICE
    constexpr int EC = 3968;          // entry buffer: flushed when a round (<= NT*8 entries) might not fit; 32,288 B LDS
    __shared__ unsigned nib[WIN/8];   // 4 bits per cell: bit tq = token t0 + tq selected it
    __shared__ unsigned ent[EC];      // pending entries: cell | bits << 28
    __shared__ int      wsum[2][NT/64];

    const int g = blockIdx.x, tid = threadIdx.x, lane = tid & 63, wv = tid >> 6;
    const int t0  = g*TQ;
    const int ntq = min(TQ, T - t0);
    if (blockIdx.y == 1) {
        // D: the group's Q*scale in f16 (scale in f32 first, then round: the tile kernel's order), in blocks of their own:
        // this stream is bandwidth-bound and overlaps the latency-bound union blocks
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
                        if (t < T && h < Hq) {
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
        QSA_STAMP(3);
        return;
    }
    int32_t * lst = list + (size_t) g*cap;
    uint2   * mv  = mval + (size_t) g*cap;
    const unsigned short * mrow[TQ];
#pragma unroll
    for (int tq = 0; tq < TQ; ++tq) {
        mrow[tq] = mask + (size_t) min(t0 + tq, T - 1)*m_s1;
    }

    QSA_STAMP(0);
    // the group's top_k indices stay in registers across the windows when width <= KPT*NT (all loads in flight at once)
    constexpr int KPT = 9;
    const bool resident = width <= KPT*NT;
    int jv[TQ][KPT];
    if (resident) {
#pragma unroll
        for (int tq = 0; tq < TQ; ++tq) {
            const int32_t * row = topk + (size_t) (t0 + min(tq, ntq - 1))*tk_s1;
#pragma unroll
            for (int k = 0; k < KPT; ++k) {
                jv[tq][k] = tq < ntq && tid + k*NT < width ? row[tid + k*NT] : -1;
            }
        }
    }
    int base = 0;   // entries written to the list
    int pend = 0;   // entries in ent[]
    int round = 0;
    for (int w0 = 0; w0 < n_kv; w0 += WIN) {
        const int wlen = min(WIN, n_kv - w0);
        const int nw   = (wlen + 7) >> 3;
        for (int k = tid; k < nw; k += NT) {
            nib[k] = 0u;
        }
        lds_barrier();
        // A: selection bits (the bounds test 0 <= j < n_kv of hc_qsa_mask, per window)
        if (resident) {
#pragma unroll
            for (int tq = 0; tq < TQ; ++tq) {
#pragma unroll
                for (int k = 0; k < KPT; ++k) {
                    const unsigned j = (unsigned) jv[tq][k] - (unsigned) w0;
                    if (j < (unsigned) wlen) {
                        atomicOr(&nib[j >> 3], 1u << (((j & 7) << 2) + tq));
                    }
                }
            }
        } else {
            for (int tq = 0; tq < ntq; ++tq) {
                const int32_t * row = topk + (size_t) (t0 + tq)*tk_s1;
#pragma unroll 8
                for (int k = tid; k < width; k += NT) {
                    const unsigned j = (unsigned) row[k] - (unsigned) w0;
                    if (j < (unsigned) wlen) {
                        atomicOr(&nib[j >> 3], 1u << (((j & 7) << 2) + tq));
                    }
                }
            }
        }
        lds_barrier();
        QSA_STAMP(1);
        // B: ordered compaction, 8 cells (one word) per thread per round, into ent[]
        for (int r0 = 0; r0 < nw; r0 += NT, ++round) {
            const int      i   = r0 + tid;
            const unsigned x   = i < nw ? nib[i] : 0u;
            const unsigned nz  = (x | x >> 1 | x >> 2 | x >> 3) & 0x11111111u;
            const int      cnt = __popc(nz);
            int pre = 0, tot = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) { // bit-sliced wave prefix of cnt (<= 8)
                const unsigned long long bal = __ballot((cnt >> b) & 1);
                pre += (int) __builtin_amdgcn_mbcnt_hi((unsigned) (bal >> 32), __builtin_amdgcn_mbcnt_lo((unsigned) bal, 0u)) << b;
                tot += __popcll(bal) << b;
            }
            int * ws = wsum[round & 1]; // double-buffered: a wave may run one round ahead of a slower one's reads
            if (lane == 0) {
                ws[wv] = tot;
            }
            lds_barrier();
            int pos = pend + pre, rt = 0;
#pragma unroll
            for (int q = 0; q < NT/64; ++q) {
                const int s = ws[q];
                pos += q < wv ? s : 0;
                rt  += s;
            }
            for (unsigned y = nz; y; y &= y - 1) {
                const int b = __builtin_ctz(y); // 4*k for cell k of the word
                ent[pos++] = (unsigned) (w0 + i*8 + (b >> 2)) | ((x >> b) & 0xFu) << 28;
            }
            pend += rt;
            const bool last = r0 + NT >= nw && w0 + WIN >= n_kv;
            if (pend > EC - NT*8 || (last && pend > 0)) {
                lds_barrier();
                QSA_STAMP(4);
                // C: the exact mask values of the pending entries. The CU*TQ gathers are unconditional (the cell is < n_kv,
                // entry 0 past the end reads cell 0, the rows are clamped), so all of them are in flight at once; the
                // selection bit picks afterwards. A conditional gather is a branch of its own with a vmcnt(0) inside.
                // CU = 4: CU = 8 needs 138 VGPRs (occupancy 1, or scratch under __launch_bounds__(NT, 2))
                constexpr int CU = 4;
                for (int q0 = 0; q0 < pend; q0 += CU*NT) {
                    unsigned e[CU], h[CU][TQ];
#pragma unroll
                    for (int u = 0; u < CU; ++u) {
                        const int q = q0 + u*NT + tid;
                        e[u] = q < pend ? ent[q] : 0u;
                    }
#pragma unroll
                    for (int u = 0; u < CU; ++u) {
#pragma unroll
                        for (int tq = 0; tq < TQ; ++tq) {
                            h[u][tq] = mrow[tq][e[u] & 0x0FFFFFFFu];
                        }
                    }
#pragma unroll
                    for (int u = 0; u < CU; ++u) {
#pragma unroll
                        for (int tq = 0; tq < TQ; ++tq) {
                            h[u][tq] = (e[u] >> (28 + tq)) & 1 ? h[u][tq] : 0xFC00u; // not selected by this token: -inf
                        }
                    }
#pragma unroll
                    for (int u = 0; u < CU; ++u) {
                        const int q = q0 + u*NT + tid;
                        if (q < pend) {
                            lst[base + q] = (int) (e[u] & 0x0FFFFFFFu);
                            mv[base + q]  = make_uint2(h[u][0] | h[u][1] << 16, h[u][2] | h[u][3] << 16);
                        }
                    }
                }
                base += pend;
                pend  = 0;   // ent[] is rewritten only after the next round's barrier
            }
        }
    }
    QSA_STAMP(2);
    if (tid == 0) {
        count[g] = base;
    }
#else
    (void) topk; (void) tk_s1; (void) width; (void) mask; (void) m_s1; (void) n_kv; (void) T; (void) Q; (void) q_s1; (void) q_s2;
    (void) Hq; (void) scale; (void) list; (void) mval; (void) count; (void) q16; (void) cap;
#endif
}

// 2. attention: grid (group, KV head, segment), 256 threads (4 wave64).
// QK role: rows rg*3 + i (token = wave w, heads 3*(lane/16) + i), cells cg + 16j of the chunk.
// PV role: token w, dims 4l..4l+3, all 12 heads: O[12][4] in f32.
// Segment s covers entries [s*seg, min(count, (s+1)*seg)), seg = pad64(ceil(count/nseg)), balanced on the device.
static __global__ void __launch_bounds__(NT, 2) qsa_attn(
        const _Float16 * __restrict__ q16, const char * __restrict__ Kc, const char * __restrict__ Vc,
        const unsigned k_nb1, const unsigned k_nb2, const unsigned v_nb1, const unsigned v_nb2,
        const int32_t * __restrict__ list, const uint2 * __restrict__ mval, const int32_t * __restrict__ count, const int cap,
        const int T, const int Hq, const int n_kvh, const int nseg,
        float * __restrict__ dst, float * __restrict__ part, float2 * __restrict__ meta) {
#if QSA_DEVICE
    const int g = blockIdx.x, hk = blockIdx.y, s = blockIdx.z;
    const int tid = threadIdx.x, w = tid >> 6, l = tid & 63;

    const int n   = count[g];
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
    __shared__ unsigned sKo[C];                         // chunk cells: K / V byte offsets (head included)
    __shared__ unsigned sVo[C];
    __shared__ unsigned short sMh[TQ][C];               // chunk cells: f16 mask bits per token

    const int rg = w*4 + (l >> 4);
    const int cg = l & 15;
    float m[3], ls[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        m[i]  = -FLT_MAX/2;
        ls[i] = 0.0f;
    }
    float O[G][4];
#pragma unroll
    for (int h = 0; h < G; ++h) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            O[h][k] = 0.0f;
        }
    }

    const int t0 = g*TQ;
    const int32_t * lst = list + (size_t) g*cap;
    const uint2   * mvp = mval + (size_t) g*cap;
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

#if QSA_PREFETCH_LIST
    // chunk list entries (lanes < 64): the next chunk's are loaded during the current chunk's PV stages
    const int cell_tail = lst[c0];
    int   nx_cell = cell_tail;
    uint2 nx_m    = make_uint2(0xFC00FC00u, 0xFC00FC00u);
    if (tid < C && c0 + tid < c1) {
        nx_cell = lst[c0 + tid];
        nx_m    = mvp[c0 + tid];
    }
#endif
    for (int cb = c0; cb < c1; cb += C) {
        if (tid < C) {
#if QSA_PREFETCH_LIST
            const int   cell = nx_cell;
            const uint2 mm   = nx_m;
#else
            // tail entries: a live cell of the list (finite data) under a -inf mask
            const int nc = min(C, c1 - cb);
            int   cell = lst[c0];
            uint2 mm   = make_uint2(0xFC00FC00u, 0xFC00FC00u);
            if (tid < nc) {
                cell = lst[cb + tid];
                mm   = mvp[cb + tid];
            }
#endif
            sKo[tid] = (unsigned) cell*k_nb1 + khead;
            sVo[tid] = (unsigned) cell*v_nb1 + vhead;
            sMh[0][tid] = (unsigned short) (mm.x & 0xFFFF);
            sMh[1][tid] = (unsigned short) (mm.x >> 16);
            sMh[2][tid] = (unsigned short) (mm.y & 0xFFFF);
            sMh[3][tid] = (unsigned short) (mm.y >> 16);
        }
        lds_barrier();

        float S[3][4];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                S[i][j] = 0.0f;
            }
        }
#pragma unroll 1
        for (int qd = 0; qd < D/DS; ++qd) {
            const uint4 ka = *(const uint4 *) (Kc + (sKo[kc]      + (unsigned) (qd*DS*2 + ko*16)));
            const uint4 kb = *(const uint4 *) (Kc + (sKo[kc + 32] + (unsigned) (qd*DS*2 + ko*16)));
            const uint4 qa = *(const uint4 *) (q16 + (qoff0 + (unsigned) (qd*DS)));
            uint4 qb = make_uint4(0, 0, 0, 0);
            if (tid < 128) {
                qb = *(const uint4 *) (q16 + (qoff1 + (unsigned) (qd*DS)));
            }
            lds_barrier(); // the previous readers of sKV / sQ are done
            *(uint4 *) &sKV[kc*KP + ko*4]        = ka;
            *(uint4 *) &sKV[(kc + 32)*KP + ko*4] = kb;
            *(uint4 *) &sQ[qr0*KP + qo*4]        = qa;
            if (tid < 128) {
                *(uint4 *) &sQ[qr1*KP + qo*4] = qb;
            }
            lds_barrier();
            // 4-dim steps (b64 LDS operands): 3 Q + 4 K operands, 24 dot2
            QSA_PRAGMA(unroll QSA_QK_UNROLL)
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
        }

        // online softmax: + the token's exact f16 mask value; P (f16) -> sP, per-row rescale -> sScale
        float mk[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            mk[j] = (float) __builtin_bit_cast(_Float16, sMh[w][cg + 16*j]);
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int row = rg*3 + i;
            float mx = -FLT_MAX/2;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                S[i][j] += mk[j];
                mx = fmaxf(mx, S[i][j]);
            }
            mx = row16_max(mx);
            const float mnew = fmaxf(m[i], mx);
            const float sc   = QSA_EXP(m[i] - mnew);
            m[i] = mnew;
            float ps = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const _Float16 p = (_Float16) QSA_EXP(S[i][j] - mnew);
                ps += (float) p; // normalize by the weights actually applied
                sP[row*PP + cg + 16*j] = p;
            }
            ls[i] = ls[i]*sc + ps;
            if (cg == 0) {
                sScale[row] = sc;
            }
        }

        // PV: 4 stages of 16 cells, V as half2(V[c][d], V[c+1][d]) cell pairs
#pragma unroll 1
        for (int vs = 0; vs < C/16; ++vs) {
            const uint4 va = *((const uint4 *) (Vc + sVo[vs*16 + 2*pr])     + dc);
            const uint4 vb = *((const uint4 *) (Vc + sVo[vs*16 + 2*pr + 1]) + dc);
#if QSA_PREFETCH_LIST
            if (vs == 0 && tid < C) {
                const int e = cb + C + tid;
                nx_cell = cell_tail;
                nx_m    = make_uint2(0xFC00FC00u, 0xFC00FC00u);
                if (e < c1) {
                    nx_cell = lst[e];
                    nx_m    = mvp[e];
                }
            }
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
                for (int h = 0; h < G; ++h) {
                    const float sc = sScale[w*G + h];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        O[h][k] *= sc;
                    }
                }
            }
            QSA_PRAGMA(unroll QSA_PV_UNROLL)
            for (int p = 0; p < 8; ++p) {
                const uint4 vv = *(const uint4 *) &sKV[p*VP + l*4];
                const h2v v4[4] = {as_h2(vv.x), as_h2(vv.y), as_h2(vv.z), as_h2(vv.w)};
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const h2v pp = *(const h2v *) &sP[(w*G + h)*PP + vs*16 + 2*p];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        O[h][k] = dot2(pp, v4[k], O[h][k]);
                    }
                }
            }
        }
        lds_barrier(); // sP / sKV / chunk cells are rewritten next
    }

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float sum = row16_sum(ls[i]);
        if (cg == 0) {
            sL[rg*3 + i]     = sum;
            sScale[rg*3 + i] = m[i];
        }
    }
    lds_barrier();
    const int t = t0 + w;
    if (t < T) {
        if (nseg == 1) {
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float lsum = sL[w*G + h];
                const float inv  = lsum > 0.0f ? 1.0f/lsum : 0.0f;
                *(float4 *) &dst[((size_t) t*Hq + hk*G + h)*D + 4*l] = make_float4(O[h][0]*inv, O[h][1]*inv, O[h][2]*inv, O[h][3]*inv);
            }
        } else {
            float * pt = part + pm*R*D;
#pragma unroll
            for (int h = 0; h < G; ++h) {
                *(float4 *) &pt[(w*G + h)*D + 4*l] = make_float4(O[h][0], O[h][1], O[h][2], O[h][3]);
            }
            if (l < G) {
                meta[pm*R + w*G + l] = make_float2(sScale[w*G + l], sL[w*G + l]);
            }
        }
    }
#else
    (void) q16; (void) Kc; (void) Vc; (void) k_nb1; (void) k_nb2; (void) v_nb1; (void) v_nb2; (void) list; (void) mval;
    (void) count; (void) cap; (void) T; (void) Hq; (void) n_kvh; (void) nseg; (void) dst; (void) part; (void) meta;
#endif
}

// 3. segment merge: one wave per output row r = t*Hq + hq (4 rows per block), lane = 4 dims (float4). The row index is
// wave-uniform (readfirstlane), so meta comes through scalar loads; the part loads of up to 4 segments are in flight at
// once (a dword per thread and segment with a wait each made it a latency chain: 127 -> 45 us at 24 heads, nseg 3).
// Segments in fixed order (deterministic); l == 0 partials (empty segment, or fully masked for this row) are skipped;
// a row with no visible cell gives 0.
static __global__ void __launch_bounds__(NT) qsa_combine(
        const float * __restrict__ part, const float2 * __restrict__ meta, float * __restrict__ dst,
        const int nrows, const int Hq, const int n_kvh, const int nseg) {
#if QSA_DEVICE
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
    const int32_t * topk; int64_t tk_s1; int width;           // TOP_K result
    const unsigned short * mask; int64_t m_s1;                // KQ mask (f16 bits)
    const float * Q; int64_t q_s1, q_s2; int q_vec4;          // f32, strides in elements; q_vec4: 16-B aligned rows and heads
    const char * K; const char * V;                           // f16, cell/head strides in bytes
    unsigned k_nb1, k_nb2, v_nb1, v_nb2;
    int T, n_kv, Hq, n_kvh; float scale;
    int32_t * list; uint2 * mval; int32_t * count; _Float16 * q16; int cap; // pool
    float * part; float2 * meta; int nseg;                    // pool (nseg > 1)
    float * dst;
};

static int n_groups(const int T) {
    return (T + TQ - 1)/TQ;
}

// segments per (group, KV head), a shape-only rule (HIP graph capture), by the attention blocks per segment
// b1 = groups x KV heads. Tuned on real wikitext top_k dumps (layers 3/27/47 at 8K/32K/64K, the first T tokens for short
// ubatches; GPU 7, engine kernels): mean regret vs the best of 3/4/6/8/12/16 <= 1.1% per (T, KV heads), worst 6.5%
// (the previous clamp(240/b1, 3, 16): up to 8.3% mean, 11.8% worst). Synthetic unions prefer 8 at b1 = 16; real ones
// 16 (by 15-23% at T = 32, 2 KV heads).
//   b1 <= 16 (T <= 64 at 1 KV head, T <= 32 at 2):   16
//   b1 <= 32 (T <= 128 / 64):                         12
//   b1 <= 64 (T <= 256 / 128):                        8
//   T >= 288 (full ubatches):                         3, 6 from 32K cells (6 wins 12 of 12 real dumps at 32K/64K by
//                                                     up to 5%, 3 wins at 8K by up to 2%)
//   else (T 132..287 at 2 KV heads):                  6
static int nseg_auto(const int T, const int n_kv, const int n_kvh) {
    const int b1 = std::max(1, n_groups(T)*n_kvh);
    if (b1 <= 16) {
        return 16;
    }
    if (b1 <= 32) {
        return 12;
    }
    if (b1 <= 64) {
        return 8;
    }
    if (T >= 288) {
        return n_kv >= 32768 ? 6 : 3;
    }
    return 6;
}

static size_t part_elems(const int ng, const int n_kvh, const int nseg) {
    return (size_t) ng*n_kvh*nseg*R*D;
}

// the three launches (the bench times them one by one); all geometry depends on shapes only
static void launch_build(const params & p, hipStream_t st) {
    const dim3 grid(n_groups(p.T), 2);
    if (p.q_vec4) {
        qsa_build<true><<<grid, NT, 0, st>>>(p.topk, p.tk_s1, p.width, p.mask, p.m_s1, p.n_kv, p.T,
            p.Q, p.q_s1, p.q_s2, p.Hq, p.scale, p.list, p.mval, p.count, p.q16, p.cap);
    } else {
        qsa_build<false><<<grid, NT, 0, st>>>(p.topk, p.tk_s1, p.width, p.mask, p.m_s1, p.n_kv, p.T,
            p.Q, p.q_s1, p.q_s2, p.Hq, p.scale, p.list, p.mval, p.count, p.q16, p.cap);
    }
}
static void launch_attn(const params & p, hipStream_t st) {
    qsa_attn<<<dim3(n_groups(p.T), p.n_kvh, p.nseg), NT, 0, st>>>(p.q16, p.K, p.V, p.k_nb1, p.k_nb2, p.v_nb1, p.v_nb2,
        p.list, p.mval, p.count, p.cap, p.T, p.Hq, p.n_kvh, p.nseg, p.dst, p.part, p.meta);
}
static void launch_combine(const params & p, hipStream_t st) {
    if (p.nseg > 1) {
        const int nrows = p.T*p.Hq;
        qsa_combine<<<(nrows + NT/64 - 1)/(NT/64), NT, 0, st>>>(p.part, p.meta, p.dst, nrows, p.Hq, p.n_kvh, p.nseg);
    }
}
static void launch(const params & p, hipStream_t st) {
    launch_build(p, st);
    launch_attn(p, st);
    launch_combine(p, st);
}

} // namespace qsa

#endif // GGML_USE_HIP || QSA_STANDALONE

#ifndef QSA_STANDALONE

#ifdef GGML_USE_HIP

static int qsa_env(const char * name, const int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

// n_kv from which the fused path is used (GGML_CUDA_FA_QSA_MIN_KV > 0 fixes it). 2304 is the smallest n_kv with a chain
// (n_kv is padded to 256; the graph skips the chain up to 2054 cells). Fused (builder + attention + combine) vs dense tile FA
// at 24 heads, width 2051 (MI50): T = 128 603 vs 772 us at 2304 cells; T = 64 330 vs 400; T = 32 187 vs 212 (3072: 251 vs
// 271); T = 16 133 vs 126 at 2304, 133 vs 135 at 2560, 135 vs 158 at 3072 (the dense figures leave out the mask kernel)
static int qsa_min_kv(const int min_kv_env, const int T) {
    if (min_kv_env > 0) {
        return min_kv_env;
    }
    return T >= 32 ? 2304 : 3072;
}

// every condition of the fused path; returns nullptr when supported, else the reason
static const char * qsa_unsupported(const ggml_cgraph * cgraph, const ggml_cuda_qsa_chain & c, const ggml_tensor * fa,
                                    const int min_t, const int min_kv_env) {
    const ggml_tensor * Q = fa->src[0], * K = fa->src[1], * V = fa->src[2];
    // a device with an empty attention slice (the meta backend's tensor split, e.g. 4 GPUs over 2 KV heads) has a zero-head
    // FA with COMPUTE cleared, while the mirrored mask chain is still computed: leave both to the unfused path
    if (!(fa->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return "FLASH_ATTN_EXT not computed here";
    }
    for (int k = 0; k < 4; ++k) {
        if (!(cgraph->nodes[c.idx[k]]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return "mask chain node not computed here";
        }
    }
    if (ggml_is_empty(fa) || ggml_is_empty(Q) || ggml_is_empty(K) || ggml_is_empty(V) || ggml_is_empty(c.ad)) {
        return "empty attention slice";
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));
    if (fa->src[4] != nullptr || max_bias != 0.0f || softcap != 0.0f) {
        return "sinks/alibi/softcap";
    }
    if (ggml_get_op_params_i32(fa, 4) != c.width) {
        return "n_kv_max != width";
    }
    if (ggml_node_get_use_count(cgraph, c.idx[3]) != 1 || (c.ad->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return "mask ADD has other uses or is an output";
    }
    if (fa->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa)) {
        return "dst type/layout";
    }
    if (Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || Q->ne[0] != qsa::D || Q->ne[3] != 1) {
        return "Q type/shape";
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || K->ne[0] != qsa::D || V->ne[0] != qsa::D || K->ne[3] != 1 ||
            V->ne[3] != 1 || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] || Q->ne[2] != qsa::G*K->ne[2]) {
        return "K/V type/shape/GQA";
    }
    if (K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || (K->nb[1] | K->nb[2] | V->nb[1] | V->nb[2]) % 16 != 0 ||
            ((uintptr_t) K->data | (uintptr_t) V->data) % 16 != 0) {
        return "K/V alignment";
    }
    if (Q->ne[1] != c.ad->ne[1] || K->ne[1] != c.ad->ne[0]) {
        return "mask shape";
    }
    if (Q->ne[1] < min_t) {
        return "T < MIN_T";
    }
    if (K->ne[1] < qsa_min_kv(min_kv_env, (int) Q->ne[1])) {
        return "n_kv < MIN_KV";
    }
    // 32-bit offsets and the builder's 28-bit cell field
    const int64_t k_max = (K->ne[1] - 1)*(int64_t) K->nb[1] + (K->ne[2] - 1)*(int64_t) K->nb[2] + qsa::D*(int64_t) sizeof(half);
    const int64_t v_max = (V->ne[1] - 1)*(int64_t) V->nb[1] + (V->ne[2] - 1)*(int64_t) V->nb[2] + qsa::D*(int64_t) sizeof(half);
    if (K->ne[1] >= (1 << 28) || k_max > INT32_MAX || v_max > INT32_MAX) {
        return "KV too large for 32-bit offsets";
    }
    return nullptr;
}

int ggml_cuda_qsa_attn_gcn(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const int mode   = qsa_env("GGML_CUDA_FA_QSA_GCN",    1);
    static const int min_t  = qsa_env("GGML_CUDA_FA_QSA_MIN_T",  16);
    static const int min_kv = qsa_env("GGML_CUDA_FA_QSA_MIN_KV", 0);   // 0: by T (qsa_min_kv)
    static const int nseg_e = qsa_env("GGML_CUDA_FA_QSA_NSEG",   0);
    static const int stats  = qsa_env("GGML_CUDA_FA_QSA_STATS",  0);
    if (!mode || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_VEGA20 || ctx.curr_stream_no != 0) {
        return 0;
    }
    ggml_cuda_qsa_chain c;
    const char * why = nullptr;
    if (!ggml_cuda_hc_qsa_chain_match(cgraph, i, c, &why)) {
        if (stats) {
            fprintf(stderr, "qsa_gcn: dev %d node %d (%s): fallback: %s\n", ctx.device, i, cgraph->nodes[i]->name, why);
        }
        return 0;
    }
    int j = c.idx[3] + 1;
    while (j < cgraph->n_nodes && ggml_cuda_hc_is_view_op(cgraph->nodes[j])) {
        ++j;
    }
    ggml_tensor * fa = j < cgraph->n_nodes ? cgraph->nodes[j] : nullptr;
    if (!fa || fa->op != GGML_OP_FLASH_ATTN_EXT || fa->src[3] != c.ad) {
        why = "FLASH_ATTN_EXT(mask = ADD) is not the next compute node";
    } else {
        why = qsa_unsupported(cgraph, c, fa, min_t, min_kv);
    }
    if (why) {
        if (stats) {
            fprintf(stderr, "qsa_gcn: dev %d node %d (%s): fallback: %s\n", ctx.device, i, cgraph->nodes[i]->name, why);
        }
        return 0;
    }

    const ggml_tensor * Q = fa->src[0], * K = fa->src[1], * V = fa->src[2];
    qsa::params p = {};
    p.topk  = (const int32_t *) c.tk->data;
    p.tk_s1 = c.tk->nb[1]/sizeof(int32_t);
    p.width = c.width;
    p.mask  = (const unsigned short *) c.mask->data;
    p.m_s1  = c.mask->nb[1]/sizeof(half);
    p.Q     = (const float *) Q->data;
    p.q_s1  = Q->nb[1]/sizeof(float);
    p.q_s2  = Q->nb[2]/sizeof(float);
    p.q_vec4 = ((uintptr_t) Q->data % 16 == 0 && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0) ? 1 : 0;
    p.K = (const char *) K->data;
    p.V = (const char *) V->data;
    p.k_nb1 = (unsigned) K->nb[1];
    p.k_nb2 = (unsigned) K->nb[2];
    p.v_nb1 = (unsigned) V->nb[1];
    p.v_nb2 = (unsigned) V->nb[2];
    p.T     = (int) Q->ne[1];
    p.n_kv  = (int) K->ne[1];
    p.Hq    = (int) Q->ne[2];
    p.n_kvh = (int) K->ne[2];
    memcpy(&p.scale, (const float *) fa->op_params + 0, sizeof(float));
    const int ng = qsa::n_groups(p.T);
    p.cap  = (int) std::min<int64_t>(p.n_kv, 4*(int64_t) c.width);
    p.nseg = nseg_e > 0 ? std::min(nseg_e, 64) : qsa::nseg_auto(p.T, p.n_kv, p.n_kvh);
    p.dst  = (float *) fa->data;

    ggml_cuda_pool_alloc<int32_t> list (ctx.pool(), (size_t) ng*p.cap);
    ggml_cuda_pool_alloc<uint2>   mval (ctx.pool(), (size_t) ng*p.cap);
    ggml_cuda_pool_alloc<int32_t> count(ctx.pool(), ng);
    ggml_cuda_pool_alloc<half>    q16  (ctx.pool(), (size_t) ng*qsa::TQ*p.Hq*qsa::D);
    ggml_cuda_pool_alloc<float>   part (ctx.pool());
    ggml_cuda_pool_alloc<float2>  meta (ctx.pool());
    if (p.nseg > 1) {
        part.alloc(qsa::part_elems(ng, p.n_kvh, p.nseg));
        meta.alloc((size_t) ng*p.n_kvh*p.nseg*qsa::R);
    }
    p.list  = list.get();
    p.mval  = mval.get();
    p.count = count.get();
    p.q16   = (_Float16 *) q16.get();
    p.part  = part.ptr;
    p.meta  = meta.ptr;

    cudaStream_t st = ctx.stream();
    qsa::launch(p, st);
    CUDA_CHECK(cudaGetLastError());

    if (stats) {
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        CUDA_CHECK(hipStreamIsCapturing(st, &cs));
        fprintf(stderr, "qsa_gcn: dev %d node %d (%s) -> %s: T %d n_kv %d Hq %d n_kvh %d width %d cap %d nseg %d",
            ctx.device, i, cgraph->nodes[i]->name, fa->name, p.T, p.n_kv, p.Hq, p.n_kvh, p.width, p.cap, p.nseg);
        if (cs == hipStreamCaptureStatusNone) {
            std::vector<int32_t> cnt(ng);
            CUDA_CHECK(cudaMemcpyAsync(cnt.data(), p.count, ng*sizeof(int32_t), cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            double sum = 0.0;
            int mx = 0;
            for (int v : cnt) {
                sum += v;
                mx = std::max(mx, v);
            }
            fprintf(stderr, " union mean %.0f max %d", sum/ng, mx);
        }
        fprintf(stderr, "\n");
    }
    return j - i;
}

#else

int ggml_cuda_qsa_attn_gcn(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(i);
    return 0;
}

#endif // GGML_USE_HIP

#endif // !QSA_STANDALONE
