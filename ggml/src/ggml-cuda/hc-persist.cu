// Persistent decode HC pre (qwen4exp, one token), see hc-persist.cuh. Prototype and measurements:
// mi50-engine/persistent-layer/hcpre (engine chain combine+norm / HC down / HC up+pre ~29 us -> ~18 us standalone).
//
// One 1024-thread block per CU (60). Wave 0 is the control wave: norm partial sums and the grid barrier.
// - HC down is tiled 10 row groups x 6 K groups: each block quantizes its K group's slice of y = x*w_norm (rstd is a
//   positive per-stream constant, so xn = y*rstd has the same int8 values; rstd folds into the reduction), dots it
//   with its 32 rows, and adds per-(row, stream) partials into int64 fixed-point accumulators (2^-32: exact, order
//   independent, deterministic). One grid barrier, then every block has lo and rstd.
// - gfx906 memory-pipe rules (measured): a CU's vector memory ops complete in issue order and __syncthreads() after a
//   global store drains all loads (s_waitcnt vmcnt(0)); so block syncs are LDS-only, the barrier runs on the scalar
//   memory path, and the HC up weights stream across the barrier.
// Weights are read in their raw q8_0 layout as dword-aligned block pairs (68 B): K groups start at even q8 blocks.
#include "hc-persist.cuh"
#include "unary.cuh"
#include "hc-dec.cuh"
#include <map>
#include <string>

namespace {

// HCP_NT: threads per block. 1024 (16 waves per CU, 64 VGPRs) or 512 (8 waves per CU, 128 VGPRs: more loads in flight
// per lane, e.g. both HC weight streams); one block per CU either way
#ifndef HCP_NT
#define HCP_NT 1024
#endif
constexpr int E = 2560, HC = 4, K = HC*E, LR = 320, NT = HCP_NT;
constexpr int QB = K/32;              // q8 blocks of xn (320), 80 per stream
constexpr int PU = LR/64;             // q8_0 block pairs per HC up row (5)
constexpr int NB = 60;
constexpr int KG = 6, RG = NB/KG, RPG = LR/RG; // 10 row groups of 32 rows x 6 K groups
constexpr int RPW2 = RPG*64/NT/2;     // HC down row pairs per wave (1 at 1024 threads, 2 at 512)
constexpr int IU   = (HC*((E + NB - 1)/NB)*(LR/64) + NT - 1)/NT; // HC up (row, pair) items per thread (860 items: 1 or 2)
constexpr int PSTR = IU*NT;           // row stride of part[]: every HC up item
// HCP_EARLY_WU: issue the HC up weight loads at kernel entry, right after the HC down ones (both 3.5 MB streams in flight
// from the start; needs the 512-thread register budget), instead of behind the HC down atomics
#if HCP_NT == 1024
#define HCP_EARLY_WU 0
#define HCP_WU_VMCNT "17"
#elif HCP_NT == 512
#ifndef HCP_EARLY_WU
#define HCP_EARLY_WU 1
#endif
#if HCP_EARLY_WU
#define HCP_WU_VMCNT "0"
#else
#define HCP_WU_VMCNT "34"
#endif
#else
#error "HCP_NT must be 1024 or 512"
#endif
static_assert(RPW2 >= 1 && RPG == 2*RPW2*(NT/64), "HC down tile rows must split evenly over the waves");
struct q81 { half2 ds; int qs[8]; };

constexpr int TMAX = 4;              // tokens per call (decode 1, MTP verify up to 4)
constexpr int HCP_TS_N = 16;         // phase timestamps per block (GGML_CUDA_HCP_TS)
struct acc_t { long long lo[TMAX][HC][LR]; long long ss[TMAX][HC]; };
__device__ acc_t        g_hcp_acc[2];
__device__ unsigned int g_hcp_bar;
// inject logits stash (int64 fixed point, by barrier epoch parity): kernel e accumulates its own sublayer's inject
// into [e&1] (GGML_CUDA_HC_PERSIST=3), the next kernel reads [e&1] before its barrier instead of running the inject
// matmul, and every kernel clears the parity it could have read after its barrier
__device__ long long    g_hcp_inj[2][TMAX][HC];
// kernel counter (its parity picks the stash / accumulator buffers; a kernel runs 1 or 3 grid barriers)
__device__ unsigned int g_hcp_kern;
// FFN extension (GGML_CUDA_HC_PERSIST=4): the FFN-side kernel also computes the shared expert; the next kernel adds it
constexpr int FF = 640;               // shared expert width (max; -sm tensor: this device's shard, 320)
constexpr int NR = 512;               // router experts
__device__ float        g_hcp_h[TMAX][FF];      // silu(gate)*up of the shared expert
__device__ float        g_hcp_sg[TMAX];         // shared expert gate logit
__device__ float        g_hcp_shexp[TMAX][E];   // sigmoid(sg) * down(h): consumed by the next kernel's combine

// early matvecs (GGML_CUDA_HC_PERSIST=4): MUL_MATs of mixed found shortly after the chain (GDN qkv / z, attention
// projections) and the GDN gate group, computed after the second grid barrier straight into their output tensors (the
// host checks that nothing in between touches that memory); the graph then skips those nodes
constexpr int HCP_EARLY_Q8 = 4;
struct hc_early {
    const char * w[HCP_EARLY_Q8];  // q8_0 [rows][2560]
    float      * out[HCP_EARLY_Q8]; // [T][rows]
    int          rows[HCP_EARLY_Q8];
    int          nq, rows_total;
    const float * wa, * wb, * dt, * A;   // GDN gates: gate = softplus(wa.x + dt)*A, beta = sigmoid(wb.x)
    float       * gate, * beta;          // [T][H]
    int           H;
};

// fused tensor-parallel AllReduce (GGML_CUDA_HCP_AR=1, 2 GPUs): the AllReduce of the sublayer output is deferred into
// the persistent kernel that consumes it; block 0 pushes this GPU's partial into the peer's inbox (P2P) and raises the
// peer's flag, waits for its own, and every block reads out = own + inbox (both GPUs add in the same order as the
// AllReduce kernel, so the mirrored results stay bit identical). Device counters replace host tokens (HIP graphs).
// N GPUs (2 or 4): the inbox has one TMAX*E slot per peer in each half ([half][slot][TMAX*E]), a flag per slot; slot of
// source rank q on rank r: q < r ? q : q - 1. 4 GPUs add the partials by rank as (p0 + p1) + (p2 + p3) on every GPU.
constexpr int ARNS = 3;               // max peers
struct hc_ar {
    const float * inbox_mine;  // this GPU's inbox: 2 halves of ns*TMAX*E floats (uncached)
    const int   * flag_mine;   // uncached ints (one per peer slot), raised by the peers' pushes
    int ns;                    // peers: 1 (pair) or 3
    int rank;                  // this GPU's rank in the group
};
struct hc_push {               // this GPU's partial -> each peer's slot for it
    float * inbox[ARNS];       // the peer's inbox + this GPU's slot (half 0)
    int   * flag[ARNS];        // the peer's flag for this GPU's slot
    int n;                     // peers
    int hs;                    // floats per inbox half (ns*TMAX*E)
};
__device__ unsigned int g_hcp_arseq;    // pushes done by this GPU (in lockstep with the peer)
__device__ unsigned int g_hcp_pushed;   // 1: the producer kernel pushed the current exchange itself (its push kernel skips it)
__device__ unsigned int g_hcp_pushcnt;  // producer push: blocks done (the last one raises the peer's flag)

// wait half of the exchange: the peer's push kernel (queued at AllReduce time, so a host sync on this GPU cannot starve
// it) writes its partial into this GPU's inbox half s&1 and then raises flag_mine to s
__device__ __forceinline__ void hcp_ar_wait(const hc_ar & ar, unsigned int s) {
    if (threadIdx.x < ar.ns) {
        while ((int) (((const volatile int *) ar.flag_mine)[threadIdx.x] - (int) s) < 0) { __builtin_amdgcn_s_sleep(1); }
        __threadfence_system();
    }
    __syncthreads();
}

// the reduced value from this GPU's own partial and the inbox half: pairs own + peer (as the AllReduce kernel); 4 GPUs
// (p0 + p1) + (p2 + p3) by rank
template <bool AR4>
__device__ __forceinline__ float hcp_ar_sum(float own, const float * in, int ns, int rank, int idx) {
    if constexpr (!AR4) {
        (void) ns; (void) rank;
        return own + in[idx];
    }
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        v[q] = q == rank ? own : in[(q < rank ? q : q - 1)*(TMAX*E) + idx];
    }
    return (v[0] + v[1]) + (v[2] + v[3]);
}

// routed experts (GGML_CUDA_HC_PERSIST=4, GGML_CUDA_HCP_MOE): after the router logits' barrier every block computes the
// top-k itself (one wave per token, bit-identical to topk_moe_sorted_gcn), then the q4_K gate/up rows of the selected
// experts (16 lanes per row) -> silu(gate)*up; after one more barrier the down rows (q5_1 / q8_0) of all selected experts,
// the routing weights, the shared expert's down rows and the shared gate give the FFN output directly
constexpr int NUMAX = 16;             // max experts per token
constexpr int FFRMAX = 640;           // max expert rows (gate/up) on this device
__device__ float        g_hcp_mh[TMAX*NUMAX][FFRMAX];  // silu(gate)*up of the selected experts, [token*nused + slot][row]
struct hc_moe {
    const char * w_gate, * w_up;   // q4_K [512][ffr][2560]
    const char * w_down;           // q5_1 or q8_0 [512][2560][ffr]
    long long    sg, su, sd, rd;   // bytes per expert (gate, up, down), bytes per down row
    int          ffr;              // rows per expert (gate/up) = down K
    int          dq8;              // down is q8_0 (else q5_1)
    int          nused;            // selected experts per token
    float        clamp;            // weight sum clamp (norm)
    float      * out;              // FFN output [T][2560]
    int          on;
    int          dedup;            // T > 1: share each distinct expert's weight reads between its tokens
    int          gq5;              // gate / up type: 0 q4_K, 1 q5_K, 2 q8_0
    hc_push      push;             // fused AllReduce: the peers' inbox slots for this GPU (push.n == 0: none)
};

struct hc_ffn {
    const float * w_logits;  // router [512][2560] f32
    const char  * w_sgate;   // shared expert gate [640][2560] q8_0
    const char  * w_sup;     // shared expert up   [640][2560] q8_0
    const char  * w_sdown;   // shared expert down [2560][640] q8_0
    const float * w_sg;      // shared expert gate logit [2560] f32
    float       * logits;    // out [T][512]
    int           on;
    int           ff;        // shared expert rows on this device (640, or the -sm tensor shard)
    hc_moe        moe;
};

__device__ __forceinline__ int kslice0(int g) { return g == KG ? QB : 2*((g*QB/KG + 1)/2); } // even: 0 54 106 160 214 266
__device__ __forceinline__ long long to_fx(float v) { return __float2ll_rn(v*4294967296.0f); }
__device__ __forceinline__ float from_fx(long long v) { return (float) v*(1.0f/4294967296.0f); }
__device__ __forceinline__ void acc_add(long long * p, float v) {
    __hip_atomic_fetch_add((unsigned long long *) p, (unsigned long long) to_fx(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void lds_sync() { __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

// full grid barrier (all of this block's global stores done first), counter on the scalar memory path
__device__ __forceinline__ void grid_sync_all() {
    __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    lds_sync();
    if (threadIdx.x < 64) {
        unsigned int old;
        __asm__ volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(old) : "s"(&g_hcp_bar), "0"(1u) : "memory");
        const unsigned int target = (old/NB + 1)*NB;
        while (true) {
            unsigned int cur;
            __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(cur) : "s"(&g_hcp_bar) : "memory");
            if (cur >= target) { break; }
            __builtin_amdgcn_s_sleep(0);
        }
        __asm__ volatile("buffer_wbinvl1_vol" ::: "memory");
    }
    lds_sync();
}

// q8_0 block pair (17 dwords) . two q8_1 blocks
__device__ __forceinline__ float pair_dot(const int w[17], const q81 & y0, const q81 & y1) {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s0 = __builtin_amdgcn_sdot4(__builtin_amdgcn_alignbyte(w[j + 1], w[j], 2), y0.qs[j], s0, false);
        s1 = __builtin_amdgcn_sdot4(w[9 + j], y1.qs[j], s1, false);
    }
    const float d0 = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFF)));
    const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[8] >> 16)));
    return d0*__low2float(y0.ds)*(float) s0 + d1*__low2float(y1.ds)*(float) s1;
}

// q8_0 block pair (17 dwords) . 64 f32 values (two 32-value groups at a0 and a1: padded LDS rows)
__device__ __forceinline__ float pair_dot_f32(const int w[17], const float * a0, const float * a1) {
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q0 = __builtin_amdgcn_alignbyte(w[j + 1], w[j], 2), q1 = w[9 + j];
        s0 += (float) (int8_t) q0*a0[4*j] + (float) (int8_t) (q0 >> 8)*a0[4*j + 1] + (float) (int8_t) (q0 >> 16)*a0[4*j + 2] + (float) (int8_t) (q0 >> 24)*a0[4*j + 3];
        s1 += (float) (int8_t) q1*a1[4*j] + (float) (int8_t) (q1 >> 8)*a1[4*j + 1] + (float) (int8_t) (q1 >> 16)*a1[4*j + 2] + (float) (int8_t) (q1 >> 24)*a1[4*j + 3];
    }
    const float d0 = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFF)));
    const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[8] >> 16)));
    return d0*s0 + d1*s1;
}

// the residual streams: either x (the combine already ran) or, fused (POST), the combine out*pw + res computed on
// the fly (pw = ps2*sigmoid(ps1*raw) per stream and token, as dsv4_hc_post_f32)
struct hc_src {
    const float * x;     // [T][4][2560], or the residual when POST
    const float * out;   // POST: sublayer output [T][2560]
    const float * raw;   // POST: inject logits [T][4] (unless read from the stash)
    const float * w_inj; // INJ: inject weights [4][10240]
    float ps1, ps2;
    int shx;             // POST: out += the shared expert stash (the FFN output is moe_out + shexp)
    const float * ar_in; // POST, fused AllReduce: this exchange's inbox half (the peers' partials of out), else null
    int ar_ns, ar_rank;  // peers, this GPU's rank
};

template <bool POST, bool AR4>
__device__ __forceinline__ float4 load_x4(const hc_src & a, const float * pw, int t, int j4) {
    const float4 r = ((const float4 *) a.x)[t*(K/4) + j4];
    if (!POST) {
        return r;
    }
    float4 o = ((const float4 *) a.out)[t*(E/4) + j4 % (E/4)];
    if (a.ar_in) {
        if constexpr (!AR4) {
            const float4 pr = ((const float4 *) a.ar_in)[t*(E/4) + j4 % (E/4)];
            o = make_float4(o.x + pr.x, o.y + pr.y, o.z + pr.z, o.w + pr.w);
        } else {
            const int b = t*E + 4*(j4 % (E/4));
            o = make_float4(hcp_ar_sum<true>(o.x, a.ar_in, a.ar_ns, a.ar_rank, b), hcp_ar_sum<true>(o.y, a.ar_in, a.ar_ns, a.ar_rank, b + 1),
                            hcp_ar_sum<true>(o.z, a.ar_in, a.ar_ns, a.ar_rank, b + 2), hcp_ar_sum<true>(o.w, a.ar_in, a.ar_ns, a.ar_rank, b + 3));
        }
    }
    if (a.shx) {
        const float4 sh = ((const float4 *) g_hcp_shexp[t])[j4 % (E/4)];
        o = make_float4(o.x + sh.x, o.y + sh.y, o.z + sh.z, o.w + sh.w);
    }
    const float w = pw[t*HC + (4*j4) / E];
    return make_float4(o.x*w + r.x, o.y*w + r.y, o.z*w + r.z, o.w*w + r.w);
}

// q8_0 block pair . T f32 vectors at once (each int8 is converted once; converting per token let the compiler keep
// all 64 converted weights live across the token loop and spill)
template <int T>
__device__ __forceinline__ void pair_dot_f32_t(const int w[17], const float (*lo)[(LR/32)*33], int pb, float out[T]) {
    float s0[T], s1[T];
#pragma unroll
    for (int t = 0; t < T; ++t) { s0[t] = 0.0f; s1[t] = 0.0f; }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q0 = __builtin_amdgcn_alignbyte(w[j + 1], w[j], 2), q1 = w[9 + j];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const float f0 = (float) (int8_t) (q0 >> (8*b)), f1 = (float) (int8_t) (q1 >> (8*b));
#pragma unroll
            for (int t = 0; t < T; ++t) {
                s0[t] += f0*lo[t][33*pb + 4*j + b];
                s1[t] += f1*lo[t][33*(pb + 1) + 4*j + b];
            }
        }
    }
    const float d0 = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFF)));
    const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[8] >> 16)));
#pragma unroll
    for (int t = 0; t < T; ++t) { out[t] = d0*s0[t] + d1*s1[t]; }
}

// ---- routed experts helpers
template <int ctrl, int row_mask = 0xF>
__device__ __forceinline__ uint32_t moe_dpp(const uint32_t v) {
    return (uint32_t) __builtin_amdgcn_update_dpp((int) v, (int) v, ctrl, row_mask, 0xF, false);
}
template <bool MAX>
__device__ __forceinline__ uint32_t moe_wave_reduce(uint32_t v) {
    const auto op = [](uint32_t a, uint32_t b) { return MAX ? max(a, b) : min(a, b); };
    v = op(v, moe_dpp<0xB1>(v));
    v = op(v, moe_dpp<0x4E>(v));
    v = op(v, moe_dpp<0x141>(v));
    v = op(v, moe_dpp<0x140>(v));
    v = op(v, moe_dpp<0x142, 0xA>(v));
    v = op(v, moe_dpp<0x143, 0xC>(v));
    return (uint32_t) __builtin_amdgcn_readlane((int) v, 63);
}

// sums over DPP rows (no LDS): every lane of each 16-lane row / 8-lane half row gets its group's sum
template <int ctrl>
__device__ __forceinline__ float dppf(const float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xF, 0xF, false));
}
__device__ __forceinline__ float row8_sum(float v) {
    v += dppf<0xB1>(v);    // quad_perm [1,0,3,2]
    v += dppf<0x4E>(v);    // quad_perm [2,3,0,1]
    v += dppf<0x141>(v);   // row_half_mirror
    return v;
}
__device__ __forceinline__ float row16_sum(float v) {
    v = row8_sum(v);
    v += dppf<0x140>(v);   // row_mirror
    return v;
}
__device__ __forceinline__ float row8_max(float v) {
    v = fmaxf(v, dppf<0xB1>(v));
    v = fmaxf(v, dppf<0x4E>(v));
    v = fmaxf(v, dppf<0x141>(v));
    return v;
}
// sum over each 32-lane half: the result is valid in the upper row of each half (lanes 16-31 and 48-63)
__device__ __forceinline__ float half32_sum(float v) {
    v = row16_sum(v);
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x142, 0xA, 0xF, false));   // row_bcast15
    return v;
}
// wave sum (any order), result in every lane
__device__ __forceinline__ float wave_sum_f(float v) {
    v = half32_sum(v);
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x143, 0xC, 0xF, false));   // row_bcast31
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 63));
}
// wave maximum (any order), result in every lane
__device__ __forceinline__ float wave_max_f(float v) {
    v = row8_max(v);
    v = fmaxf(v, dppf<0x140>(v));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(v), __float_as_int(v), 0x142, 0xA, 0xF, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(v), __float_as_int(v), 0x143, 0xC, 0xF, false)));
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 63));
}

// softmax + top-k + normalized weights of one token's 512 logits, by one wave, exactly as topk_moe_sorted_gcn (the
// softmax sum in the reference 32-lane order via a shuffle instead of LDS); results in sel_e / sel_w (LDS)
__device__ __forceinline__ void moe_topk_wave(float v[NR/64], const int nused, const float clamp_val, const int lane,
                                              int * sel_e, float * sel_w) {
    constexpr int J = NR/64, NL = 32;
    float m = v[0];
#pragma unroll
    for (int j = 1; j < J; ++j) { m = fmaxf(m, v[j]); }
    m = wave_max_f(m);
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        v[j] = expf(v[j] - m);
        const float hi = __shfl(v[j], (lane + 32) % 64, 64);   // expert lane + 32 + 64 j
        s += v[j];
        s += hi;                                                // lane L < 32: experts L + 32 i in order i
    }
    s = warp_reduce_sum<NL>(s);
    s = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(s), 0));
    const float inv_sum = 1.0f/s;
    // keys (larger float -> larger key, never 0); each round every lane offers its largest untaken key (ties: lower j,
    // i.e. the lower expert, as the sorted lists of the reference), the wave maximum wins (ties: lowest expert)
    uint32_t key[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        float p = v[j]*inv_sum;
        if (__isnanf(p)) { p = -FLT_MAX; }
        const uint32_t u = __float_as_uint(p);
        key[j] = u & 0x80000000u ? ~u : u | 0x80000000u;
    }
    float selp = 0.0f;   // lane r < nused: the r-th pick
    int   sele = 0;
    for (int r = 0; r < nused; ++r) {
        uint32_t hk = key[0];
        int      hj = 0;
#pragma unroll
        for (int j = 1; j < J; ++j) { if (key[j] > hk) { hk = key[j]; hj = j; } }
        const int he = lane + 64*hj;
        const uint32_t mk = moe_wave_reduce<true>(hk);
        const uint64_t tied = __ballot(hk == mk);
        int win;
        if (__popcll(tied) == 1) {
            win = __ffsll((unsigned long long) tied) - 1;
        } else {
            const uint32_t me = moe_wave_reduce<false>(hk == mk ? (uint32_t) he : 0xFFFFFFFFu);
            win = __ffsll((unsigned long long) __ballot(hk == mk && (uint32_t) he == me)) - 1;
        }
        const int we = __builtin_amdgcn_readlane(he, win);
        if (lane == r) { selp = __uint_as_float(mk & 0x80000000u ? mk & 0x7FFFFFFFu : ~mk); sele = we; }
        if (lane == win) {
#pragma unroll
            for (int j = 0; j < J; ++j) { key[j] = j == hj ? 0u : key[j]; }
        }
    }
    // weight sum as the reference: lane L < 32 adds the picks whose expert % 32 == L, in pick order, then a butterfly
    // (the picks go through LDS: broadcast reads instead of a chain of shuffles)
    if (lane < nused) { sel_e[lane] = sele; sel_w[lane] = selp; }
    __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    float ws = 0.0f;
    for (int q = 0; q < nused; ++q) {
        if (sel_e[q] % NL == lane) { ws += sel_w[q]; }
    }
    ws = warp_reduce_sum<NL>(ws);
    ws = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ws), 0));
    const float inv = 1.0f/fmaxf(ws, clamp_val);
    if (lane < nused) { sel_w[lane] = selp*inv; }
}

// scales / mins of q4_K sub-blocks 2p and 2p + 1 (get_scale_min_k4) from the 12 scale bytes (w0 w1 w2)
__device__ __forceinline__ void moe_q4k_sm(const unsigned w0, const unsigned w1, const unsigned w2, const int p,
                                           int & scA, int & mA, int & scB, int & mB) {
    // branchless (p is per lane): sub-blocks 2p, 2p+1 sit at bit shift 16*(p & 1) in both layouts
    const int sh = 16*(p & 1);
    const bool hi = p >= 2;
    const unsigned a0 = w0 >> sh, a1 = w1 >> sh, a2 = w2 >> sh;
    scA = hi ? (int) ((a2 & 0xF) | ((a0 >> 6 & 3) << 4))         : (int) (a0 & 63);
    mA  = hi ? (int) ((a2 >> 4 & 0xF) | ((a1 >> 6 & 3) << 4))    : (int) (a1 & 63);
    scB = hi ? (int) ((a2 >> 8 & 0xF) | ((a0 >> 14 & 3) << 4))   : (int) (a0 >> 8 & 63);
    mB  = hi ? (int) ((a2 >> 12 & 0xF) | ((a1 >> 14 & 3) << 4))  : (int) (a1 >> 8 & 63);
}

// one q4_K item of a row: the lane's 16-byte qs chunk qv (sub-blocks 2p, 2p+1 at positions 16h..16h+15, hd: the
// super-block's d/dmin and scales) . the q8 x ints xa (block 2p) / xb (block 2p+1) with their scales and int sums
// (q5_K: qh holds the 16 qh bytes of the same positions; bit 2p / 2p + 1 is the 5th bit of sub-block 2p / 2p + 1)
template <bool Q5>
__device__ __forceinline__ float moe_q4k_item(const int4 hd, const int4 qv, const int4 qh, const int xa[4], const int xb[4],
                                              const int p, const float d8a, const float d8b, const int sA, const int sB) {
    const int q[4] = { qv.x, qv.y, qv.z, qv.w }, hb[4] = { qh.x, qh.y, qh.z, qh.w };
    int dA = 0, dB = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int lo = q[j] & 0x0F0F0F0F, hi = (q[j] >> 4) & 0x0F0F0F0F;
        if (Q5) {
            lo |= ((hb[j] >> (2*p)) & 0x01010101) << 4;
            hi |= ((hb[j] >> (2*p + 1)) & 0x01010101) << 4;
        }
        dA = __builtin_amdgcn_sdot4(lo, xa[j], dA, false);
        dB = __builtin_amdgcn_sdot4(hi, xb[j], dB, false);
    }
    int scA, mA, scB, mB;
    moe_q4k_sm(hd.y, hd.z, hd.w, p, scA, mA, scB, mB);
    const float2 dm = __half22float2(__builtin_bit_cast(half2, hd.x));
    return dm.x*(d8a*(float) (dA*scA) + d8b*(float) (dB*scB)) - dm.y*(d8a*(float) (sA*mA) + d8b*(float) (sB*mB));
}

// q5_1 block (6 dwords) . q8_1 block (ds = d, sum)
__device__ __forceinline__ float moe_q51_dot(const int * w, const q81 & x) {
    const unsigned qh = w[1];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = w[2 + j];
        const unsigned hb = qh >> (4*j), hb2 = qh >> (4*j + 16);
        const int lo = (q & 0x0F0F0F0F)        | ((hb & 1) << 4)  | ((hb & 2) << 11)  | ((hb & 4) << 18)  | ((hb & 8) << 25);
        const int hi = ((q >> 4) & 0x0F0F0F0F) | ((hb2 & 1) << 4) | ((hb2 & 2) << 11) | ((hb2 & 4) << 18) | ((hb2 & 8) << 25);
        sumi = __builtin_amdgcn_sdot4(lo, x.qs[j], sumi, false);
        sumi = __builtin_amdgcn_sdot4(hi, x.qs[4 + j], sumi, false);
    }
    const float2 dm = __half22float2(__builtin_bit_cast(half2, w[0]));
    const float2 ds = __half22float2(x.ds);
    return dm.x*ds.x*(float) sumi + dm.y*ds.y;
}

// 8 consecutive threads quantize 32 values (4 each) into a q8_1 block with ds = (d, sum), as quantize_q8_1
__device__ __forceinline__ void moe_quant4(const float4 v, q81 & b, const int sub) {
    const float amax = row8_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    const float sum  = row8_sum(v.x + v.y + v.z + v.w);
    const float d = amax/127.0f, dd = amax == 0.0f ? 1.0f : d;
    char4 q;
    const float id = __builtin_amdgcn_rcpf(dd); // one reciprocal instead of 4 IEEE divisions (x/d rounds the same
                                                // except within an ulp of a .5 boundary)
    q.x = roundf(v.x*id); q.y = roundf(v.y*id); q.z = roundf(v.z*id); q.w = roundf(v.w*id);
    b.qs[sub] = *(int *) &q;
    if (sub == 0) { b.ds = make_half2(d, sum); }
}

template <int T, bool POST, bool STASH, bool INJ, bool AR4>
__launch_bounds__(NT)
#if HCP_NT == 512
__attribute__((amdgpu_waves_per_eu(1, 2)))
#endif
__global__ void hc_persist_kernel(const hc_src src_in, const float * __restrict__ w_norm, const char * __restrict__ w_down,
                                  const char * __restrict__ w_up, const float eps, const float lo_scale, const float pre_scale,
                                  float * __restrict__ xpost, float * __restrict__ xn, float * __restrict__ mixed, const hc_ffn ffn,
                                  const hc_early early, const hc_ar ar, unsigned long long * __restrict__ hts) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__) // GCN only (hc_persist_chain): other targets compile an empty kernel
#define HTS(k) if (hts && threadIdx.x == 0) { hts[blockIdx.x*HCP_TS_N + (k)] = __builtin_amdgcn_s_memrealtime(); }
    __shared__ q81   xqk[T][80];         // q8 of y over this K group; FFN phase: q8 of mixed (80 blocks)
    __shared__ float lo2[T][(LR/32)*33];  // silu(lo_scale*lo) in f32 like the engine's fused HC up (padded rows)
    // part[T][NT] (HC up / shared expert down partials), and in the routed expert phase the q8_1 staging of the down
    // inputs (T*(nused*ffr + ff)/32 blocks)
    // (T <= 3: also mixed in f32 for the E phase's f32 matvecs, T*10 KB)
    constexpr int ARENA = T < 4 ? T*E*4 : 4*4800;
    __shared__ __attribute__((aligned(16))) char arena[ARENA > T*PSTR*4 ? ARENA : T*PSTR*4];
    float (*part)[PSTR] = (float (*)[PSTR]) arena;
    __shared__ int   sel_e[T][NUMAX];
    __shared__ int   d_ex[T*NUMAX];       // routed experts: distinct experts of the call (first-appearance order)
    __shared__ int   d_ent[T*NUMAX][T];   // their users: entries t*nused + slot
    __shared__ int   d_tok[T*NUMAX][T];   // their tokens
    __shared__ int   f_ent[T*NUMAX];      // all entries grouped by expert (distinct order, users in entry order)
    __shared__ int   f_tok[T*NUMAX];      // their tokens
    __shared__ int   d_n[T*NUMAX];
    __shared__ int   d_cnt;
    __shared__ float sel_w[T][NUMAX];
    __shared__ float gl[T][HC][64];
    __shared__ float xcw[T][HC][64];
    __shared__ float rstd[T][HC];
    __shared__ float red[2*T][NT/64];
    __shared__ float pw[T*HC];
    __shared__ float injp[HC][TMAX*44];
    __shared__ float4 xsl[TMAX*44];      // this block's slice of the residual streams (read before the barrier)
    __shared__ float4 wil[HC][44];       // INJ: inject weights of the slice
    __shared__ int   epoch_s;
    const int tid = threadIdx.x, bid = blockIdx.x, lane = tid % 64, wave = tid / 64;
    HTS(0)
    const int kg = bid / RG, rg = bid % RG;
    const int kq0 = kslice0(kg), nqk = kslice0(kg + 1) - kq0, clk = kq0 / (QB/HC), npk = nqk/2;
    const int e0 = bid*E/NB, e1 = (bid + 1)*E/NB, n = e1 - e0, niu = HC*n*PU;
    // HC down weights of this block's tile, loaded first: the 3.4 MB stream overlaps phase A (latency bound: epoch, stash,
    // the peer AllReduce wait, the norm slice) instead of starting after it
    int wdn[RPW2][17];
#pragma unroll
    for (int j = 0; j < RPW2; ++j) {
        const int p = min(lane % 32, npk - 1);
        const int row = rg*RPG + 2*(wave*RPW2 + j) + lane / 32;
        const int * wsrc = (const int *) (w_down + (size_t) row*(QB*34) + (size_t) (kq0 + 2*p)*34);
#pragma unroll
        for (int k = 0; k < 17; ++k) { wdn[j][k] = wsrc[k]; }
    }
    int wu[IU][17];
#define HCP_LOAD_WU() do { \
        _Pragma("unroll") \
        for (int u = 0; u < IU; ++u) { \
            const int it = min(tid + NT*u, niu - 1); \
            const int rr = it / PU, c = rr / n, i = rr % n; \
            const int * usrc = (const int *) (w_up + ((size_t) c*E + e0 + i)*(LR/32*34) + (size_t) (it % PU)*68); \
            _Pragma("unroll") \
            for (int k = 0; k < 17; ++k) { wu[u][k] = usrc[k]; } \
        } } while (0)
#if HCP_EARLY_WU
    HCP_LOAD_WU();
#endif

    if (tid == 0) {
        unsigned int b0;
        __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(b0) : "s"(&g_hcp_kern) : "memory");
        epoch_s = b0 & 1;
    }
    lds_sync();
    const int ep = epoch_s;
    // fused AllReduce of out: wait for the peer's partial (s = this GPU's last push, the exchange of out)
    hc_src src = src_in;
    if (POST && ar.inbox_mine) {
        unsigned int s;
        __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(s) : "s"(&g_hcp_arseq) : "memory");
        hcp_ar_wait(ar, s);
        src.ar_in = ar.inbox_mine + (s & 1)*ar.ns*TMAX*E;
        src.ar_ns = ar.ns;
        src.ar_rank = ar.rank;
    }
    HTS(15)
    if (POST) {
        if (tid < T*HC) {
            const float raw = STASH ? from_fx(g_hcp_inj[ep ^ 1][tid / HC][tid % HC]) : src.raw[tid];
            pw[tid] = src.ps2/(1.0f + expf(-src.ps1*raw));
        }
        lds_sync();
    }
    // ---- A (before any weight traffic): K group slice of y = x*w_norm -> q8 in LDS per token, K group sums of
    //      squares (block rg == 0 of each K group adds them), this block's mix columns
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float ssa = 0.0f, ssb = 0.0f;
        if (tid < nqk*8) {
            const int jk = kq0*8 + tid;
            const float4 xk = load_x4<POST, AR4>(src, pw, t, jk), wk = ((const float4 *) w_norm)[jk];
            const float4 yk = make_float4(xk.x*wk.x, xk.y*wk.y, xk.z*wk.z, xk.w*wk.w);
            const float amax = row8_max(fmaxf(fmaxf(fabsf(yk.x), fabsf(yk.y)), fmaxf(fabsf(yk.z), fabsf(yk.w))));
            const float d = amax/127.0f, dd = amax == 0.0f ? 1.0f : d; // as quantize_q8_1: q = round(x/d)
            char4 q;
            const float id = __builtin_amdgcn_rcpf(dd); // one reciprocal instead of 4 IEEE divisions (x/d rounds the same
            // except within an ulp of a .5 boundary)
            q.x = roundf(yk.x*id); q.y = roundf(yk.y*id); q.z = roundf(yk.z*id); q.w = roundf(yk.w*id);
            xqk[t][tid/8].qs[tid % 8] = *(int *) &q;
            if (tid % 8 == 0) { xqk[t][tid/8].ds = make_half2(d, 0.0f); }
            const float s4 = xk.x*xk.x + xk.y*xk.y + xk.z*xk.z + xk.w*xk.w;
            if ((4*jk) / E != clk) { ssb = s4; } else { ssa = s4; }
        }
        if (rg == 0) {
            ssa = wave_sum_f(ssa); ssb = wave_sum_f(ssb);
            if (lane == 0) { red[2*t][wave] = ssa; red[2*t + 1][wave] = ssb; }
        }
    }
    // this block's xn slice of the residual streams, read now: fused with the combine, the residual and sublayer
    // output die at the combine node, so the allocator may hand their memory to xn / mixed, which other blocks write
    // after the barrier
    const int j40 = bid*(K/4)/NB, nj4 = (bid + 1)*(K/4)/NB - j40;
    if (tid < T*nj4) { xsl[tid] = load_x4<POST, AR4>(src, pw, tid / nj4, j40 + tid % nj4); }
    if (INJ && tid >= 256 && tid < 256 + HC*nj4) {
        const int r = (tid - 256) / nj4, k = (tid - 256) % nj4;
        wil[r][k] = ((const float4 *) src.w_inj)[r*(K/4) + j40 + k];
    }
    for (int it = tid; it < T*HC*n; it += NT) {
        const int t = it / (HC*n), c = (it / n) % HC, i = it % n;
        const float r = src.x[t*K + c*E + e0 + i];
        const float o = POST ? (src.ar_in ? hcp_ar_sum<AR4>(src.out[t*E + e0 + i], src.ar_in, src.ar_ns, src.ar_rank, t*E + e0 + i)
                                          : src.out[t*E + e0 + i]) +
                               (src.shx ? g_hcp_shexp[t][e0 + i] : 0.0f) : 0.0f;
        const float xv = POST ? o*pw[t*HC + c] + r : r;
        xcw[t][c][i] = xv*w_norm[c*E + e0 + i];
    }
    lds_sync();
    HTS(1)
    acc_t & A = g_hcp_acc[ep];
    if (rg == 0 && tid < 2*T && clk + tid % 2 < HC) {
        float t = 0.0f;
        for (int w = 0; w < NT/64; ++w) { t += red[tid][w]; }
        acc_add(&A.ss[tid/2][clk + tid % 2], t);
    }

    // ---- HC down: wave w takes row pairs RPW2*w + j (row 2(.) on lanes 0-31, 2(.)+1 on lanes 32-63) of the tile, one
    //      q8 block pair per lane and all T tokens (loads unconditional so every wave issues 17 per pair: spare lanes
    //      reload the last pair)
#pragma unroll
    for (int j = 0; j < RPW2; ++j) {
        const int half_ = lane / 32, p = min(lane % 32, npk - 1);
        const int row = rg*RPG + 2*(wave*RPW2 + j) + half_;
        const int * w = wdn[j];
        const int qs1 = (clk + 1)*(QB/HC) - kq0; // local q from which the K slice is in the next stream (even)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float v = lane % 32 < npk ? pair_dot(w, xqk[t][2*p], xqk[t][2*p + 1]) : 0.0f;
            float p0 = 2*p < qs1 ? v : 0.0f, p1 = 2*p < qs1 ? 0.0f : v;
            p0 = half32_sum(p0); p1 = half32_sum(p1);
            if (lane % 32 == 31) {
                acc_add(&A.lo[t][clk][row], p0);
                if (qs1 < nqk) { acc_add(&A.lo[t][clk + 1][row], p1); }
            }
        }
    }
    // HC up: IU (row, pair) items per thread (item tid + NT*u), issued behind the atomics; they stream across the barrier.
    // vmcnt(17*IU) = this wave's atomics done, its HC up loads may still be in flight.
    __asm__ volatile("" ::: "memory");
#if !HCP_EARLY_WU
    HCP_LOAD_WU();
#endif
    __asm__ volatile("s_waitcnt vmcnt(" HCP_WU_VMCNT ")" ::: "memory");
    lds_sync();
    if (wave == 0) {
        // grid barrier on the scalar memory path (a vector atomic/poll would queue behind the HC up stream)
        unsigned int old;
        __asm__ volatile("s_waitcnt vmcnt(" HCP_WU_VMCNT ") lgkmcnt(0)\n\ts_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(old) : "s"(&g_hcp_bar), "0"(1u) : "memory");
        const unsigned int target = (old/NB + 1)*NB;
        while (true) {
            unsigned int cur;
            __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(cur) : "s"(&g_hcp_bar) : "memory");
            if (cur >= target) { break; }
            __builtin_amdgcn_s_sleep(0);
        }
        __asm__ volatile("buffer_wbinvl1_vol" ::: "memory");
        if (bid == 0 && lane == 0) { atomicAdd(&g_hcp_kern, 1u); } // every block read it before arriving here
        // the T*4 accumulated sums of squares: scalar loads of 8 int64 each
        typedef long long ll8 __attribute__((ext_vector_type(8)));
#pragma unroll
        for (int h = 0; h < (T*HC + 7)/8; ++h) {
            ll8 ss;
            const unsigned long long pa = (unsigned long long) (&A.ss[0][0] + 8*h);
            unsigned int plo, phi;
            __asm__ volatile("v_readfirstlane_b32 %0, %1" : "=s"(plo) : "v"((unsigned int) pa));
            __asm__ volatile("v_readfirstlane_b32 %0, %1" : "=s"(phi) : "v"((unsigned int) (pa >> 32)));
            const long long * ps = (const long long *) (((unsigned long long) phi << 32) | plo);
            __asm__ volatile("s_load_dwordx16 %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(ss) : "s"(ps) : "memory");
            long long v = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) { v = lane == 8*h + k ? ss[k] : v; }
            if (lane >= 8*h && lane < min(8*h + 8, T*HC)) { rstd[lane/HC][lane % HC] = 1.0f/sqrtf(from_fx(v)/E + eps); }
        }
    }
    lds_sync();
    HTS(2)
    // ---- lo = sum_c rstd[c]*lo_c; silu(lo_scale*lo) -> f32 in LDS; xn (and the combined residual) slice
    for (int it = tid; it < T*LR; it += NT) {
        const int t = it / LR, r = it % LR;
        float v = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) { v += rstd[t][c]*from_fx(A.lo[t][c][r]); }
        v *= lo_scale;
        lo2[t][(r/32)*33 + r % 32] = v/(1.0f + expf(-v));
    }
    if (tid < T*nj4) {
        const int t = tid / nj4, j4 = j40 + tid % nj4;
        const float4 xv = xsl[tid], wv = ((const float4 *) w_norm)[j4];
        if (POST) { ((float4 *) xpost)[t*(K/4) + j4] = xv; }
        const float r = rstd[t][(4*j4) / E];
        const float4 xq = make_float4((xv.x*r)*wv.x, (xv.y*r)*wv.y, (xv.z*r)*wv.z, (xv.w*r)*wv.w);
        ((float4 *) xn)[t*(K/4) + j4] = xq;
        if (INJ) {
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const float4 wc = wil[c][tid % nj4];
                injp[c][tid] = wc.x*xq.x + wc.y*xq.y + wc.z*xq.z + wc.w*xq.w;
            }
        }
    }
    if (bid == 0) {
        long long * o = (long long *) &g_hcp_acc[ep ^ 1];
        for (int i = tid; i < (int) (sizeof(acc_t)/8); i += NT) { o[i] = 0; }
        if (tid < TMAX*HC) { (&g_hcp_inj[ep ^ 1][0][0])[tid] = 0; }
    }
    lds_sync();
    if (INJ && tid < T*HC) {
        // this block's inject partial per (token, row), fixed order, into the stash
        const int t = tid / HC, c = tid % HC;
        float sum = 0.0f;
        for (int k = 0; k < nj4; ++k) { sum += injp[c][t*nj4 + k]; }
        acc_add(&g_hcp_inj[ep][t][c], sum);
    }
    // ---- HC up dots (q8_0 x f32 lo, as the engine's fused decode kernel), gate per (stream, column), gated mix
#pragma unroll
    for (int u = 0; u < IU; ++u) {
        const int it = tid + NT*u;
        if (it < niu) {
            float g[T];
            pair_dot_f32_t<T>(wu[u], lo2, 2*(it % PU), g);
#pragma unroll
            for (int t = 0; t < T; ++t) { part[t][it] = g[t]; }
        }
    }
    lds_sync();
    for (int it = tid; it < T*HC*n; it += NT) {
        const int t = it / (HC*n), ci = it % (HC*n);
        const float * pp = part[t] + ci*PU;
        float g = 0.0f;
#pragma unroll
        for (int j = 0; j < PU; ++j) { g += pp[j]; }
        gl[t][ci / n][ci % n] = g;
    }
    lds_sync();
    for (int it = tid; it < T*n; it += NT) {
        const int t = it / n, i = it % n;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) { s += (xcw[t][c][i]*rstd[t][c])/(1.0f + expf(-gl[t][c][i])); }
        mixed[t*E + e0 + i] = s*pre_scale;
    }
    HTS(3)
    if (!ffn.on && early.nq == 0 && early.H == 0) {
        HTS(7)
        return;
    }
    // ---- consumers of mixed: early matvecs, and (FFN extension) router logits and the shared expert
    grid_sync_all();
    HTS(4)
    // mixed -> q8_1 (as the engine's MMVQ) for the shared expert gate / up
    // (T <= 3: the f32 values also go to the arena for the f32 matvecs below)
    constexpr bool MIX_LDS = T < 4;
    float4 * mixl = (float4 *) arena;
    for (int i4 = tid; i4 < T*(E/4); i4 += NT) {
        const int t = i4 / (E/4), j = i4 % (E/4);
        const float4 v = ((const float4 *) mixed)[i4];
        if (MIX_LDS) { mixl[i4] = v; }
        const float amax = row8_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        const float d = amax/127.0f, dd = amax == 0.0f ? 1.0f : d; // as quantize_q8_1: q = round(x/d)
        char4 q;
        const float id = __builtin_amdgcn_rcpf(dd); // one reciprocal instead of 4 IEEE divisions (x/d rounds the same
                                                    // except within an ulp of a .5 boundary)
        q.x = roundf(v.x*id); q.y = roundf(v.y*id); q.z = roundf(v.z*id); q.w = roundf(v.w*id);
        xqk[t][j/8].qs[j % 8] = *(int *) &q;
        if (j % 8 == 0) { xqk[t][j/8].ds = make_half2(d, 0.0f); }
    }
    lds_sync();
    if (early.nq > 0) {
        // early q8_0 matrices: this block's share of all their rows. The first NT/8 rows take 8 lanes each (5 block pairs
        // per lane, one round); rows beyond that take a whole wave each (one pair per lane), so the leftover rows cost one
        // load deep instead of a second 5-deep round
        const int g0 = bid*early.rows_total/NB, g1 = (bid + 1)*early.rows_total/NB;
        const int l8 = lane % 8;
        const int gsplit = min(g1, g0 + NT/8);
        for (int g = g0 + tid/8; g < gsplit; g += NT/8) {
            int m = 0, r = g;
            while (m + 1 < early.nq && r >= early.rows[m]) { r -= early.rows[m]; ++m; }
            const int * wrow = (const int *) (early.w[m] + (size_t) r*(E/32*34));
            float acc[T];
#pragma unroll
            for (int t = 0; t < T; ++t) { acc[t] = 0.0f; }
#pragma unroll 1
            for (int k = 0; k < 5; ++k) {
                const int p = l8 + 8*k;
                int w[17];
#pragma unroll
                for (int q = 0; q < 17; ++q) { w[q] = wrow[17*p + q]; }
#pragma unroll
                for (int t = 0; t < T; ++t) { acc[t] += pair_dot(w, xqk[t][2*p], xqk[t][2*p + 1]); }
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float v = acc[t];
                v = row8_sum(v);
                if (l8 == 0) { early.out[m][t*early.rows[m] + r] = v; }
            }
        }
#pragma unroll 1
        for (int g = gsplit + wave; g < g1; g += NT/64) {
            int m = 0, r = g;
            while (m + 1 < early.nq && r >= early.rows[m]) { r -= early.rows[m]; ++m; }
            const int p = min(lane, E/64 - 1);
            const int * wsrc = (const int *) (early.w[m] + (size_t) r*(E/32*34) + (size_t) p*68);
            int w[17];
#pragma unroll
            for (int q = 0; q < 17; ++q) { w[q] = wsrc[q]; }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float v = lane < E/64 ? pair_dot(w, xqk[t][2*p], xqk[t][2*p + 1]) : 0.0f;
                v = wave_sum_f(v);
                if (lane == 0) { early.out[m][t*early.rows[m] + r] = v; }
            }
        }
    }
    HTS(14)
    if (ffn.on || early.H > 0) {
        const int l0 = bid*NR/NB, nl = ffn.on ? (bid + 1)*NR/NB - l0 : 0;   // router rows
        const int r0 = bid*ffn.ff/NB, ns = ffn.on ? (bid + 1)*ffn.ff/NB - r0 : 0;   // shared expert rows (gate and up)
        const int a0 = bid*2*early.H/NB, na = (bid + 1)*2*early.H/NB - a0;          // GDN gate rows (alpha, then beta)
        const int nsg = ffn.on && bid == NB - 1;
        const int ntask = nl + ns + nsg + na;
#pragma unroll 1
        for (int task = wave; task < ntask; task += NT/64) {
            if (task < nl || task >= nl + ns) {
                // f32 row . f32 mixed: router logits, the shared expert gate logit, or a GDN gate row
                const int ab = task - nl - ns - nsg;   // >= 0: GDN gate row a0 + ab
                const int hr = a0 + ab, is_b = hr >= early.H, h = is_b ? hr - early.H : hr;
                const float4 * w4 = task < nl ? (const float4 *) (ffn.w_logits + (size_t) (l0 + task)*E) :
                                    ab < 0    ? (const float4 *) ffn.w_sg :
                                                (const float4 *) ((is_b ? early.wb : early.wa) + (size_t) h*E);
                float acc[T];
#pragma unroll
                for (int t = 0; t < T; ++t) { acc[t] = 0.0f; }
                // 10 float4 per lane: batches of FB weight loads in flight (one load per iteration left the row 10 memory
                // round trips deep); mixed from LDS (T <= 3) or L2
                constexpr int FB = MIX_LDS ? 5 : 2;
#pragma unroll 1
                for (int k0 = lane; k0 < E/4; k0 += FB*64) {
                    float4 w[FB];
#pragma unroll
                    for (int b = 0; b < FB; ++b) { w[b] = w4[k0 + b*64]; }
#pragma unroll
                    for (int b = 0; b < FB; ++b) {
                        const int k = k0 + b*64;
#pragma unroll
                        for (int t = 0; t < T; ++t) {
                            const float4 m = MIX_LDS ? mixl[t*(E/4) + k] : ((const float4 *) mixed)[t*(E/4) + k];
                            acc[t] += w[b].x*m.x + w[b].y*m.y + w[b].z*m.z + w[b].w*m.w;
                        }
                    }
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float v = wave_sum_f(acc[t]);
                    if (lane == 0) {
                        if (task < nl) {
                            ffn.logits[t*NR + l0 + task] = v;
                        } else if (ab < 0) {
                            g_hcp_sg[t] = v;
                        } else if (is_b) {
                            early.beta[t*early.H + h] = 1.0f/(1.0f + expf(-v));
                        } else {
                            const float u = v + early.dt[h];
                            early.gate[t*early.H + h] = (u > 20.0f ? u : logf(1.0f + expf(u)))*early.A[h]; // as ggml softplus
                        }
                    }
                }
            } else {
                // shared expert gate row, then up row (same registers), r -> silu(gate)*up
                const int r = r0 + task - nl, p = min(lane, 39);
                int a[17];
#pragma unroll 1
                for (int h = 0; h < 2; ++h) {
                    const int * wsrc = (const int *) ((h == 0 ? ffn.w_sgate : ffn.w_sup) + (size_t) r*(E/32*34) + (size_t) p*68);
#pragma unroll
                    for (int k = 0; k < 17; ++k) { a[k] = wsrc[k]; }
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        const float v = wave_sum_f(lane < 40 ? pair_dot(a, xqk[t][2*p], xqk[t][2*p + 1]) : 0.0f);
                        if (lane == 0) {
                            if (h == 0) { gl[t][0][wave] = v; }
                            else { const float g = gl[t][0][wave]; g_hcp_h[t][r] = g/(1.0f + expf(-g))*v; }
                        }
                    }
                }
            }
        }
    }
    HTS(5)
    if (!ffn.on) {
        HTS(7)
        return;
    }
    grid_sync_all();
    HTS(6)
    if (ffn.moe.on) {
        const hc_moe & mo = ffn.moe;
        unsigned int push_s = 0;
        long long pofs = 0;              // this exchange's half in every peer inbox
        if (mo.push.n) {
            unsigned int s0;
            __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(s0) : "s"(&g_hcp_arseq) : "memory");
            push_s = s0 + 1;
            pofs = (long long) (push_s & 1)*mo.push.hs;
        }
        const int NU = mo.nused, ffr = mo.ffr, NBK = ffr/32;
        const int wu_ = __builtin_amdgcn_readfirstlane(wave);   // wave-uniform in SGPRs
        // router: top-k of every token in every block (wave t: token t)
        if (wu_ < T) {
            float lg[NR/64];
#pragma unroll
            for (int j = 0; j < NR/64; ++j) { lg[j] = ffn.logits[wu_*NR + lane + 64*j]; }
            moe_topk_wave(lg, NU, mo.clamp, lane, sel_e[wu_], sel_w[wu_]);
            HTS(11)
        }
        lds_sync();
        // distinct experts over the call's T*NU picks (wave 0, identical in every block): entry i = t*NU + slot;
        // an entry is the first of its expert if no lower entry has it; its users are in entry order
        if (wu_ == 0) {
            const int ne = T*NU;
            const int x = lane < ne ? (&sel_e[0][0])[(lane / NU)*NUMAX + lane % NU] : -1 - lane;
            unsigned long long same = 1ull << lane;
            if (T > 1 && mo.dedup) {
                for (int j = 0; j < ne; ++j) {
                    const int xj = __builtin_amdgcn_readlane(x, j);
                    same |= xj == x ? 1ull << j : 0ull;
                }
            }
            const unsigned long long lt = (1ull << lane) - 1;
            const bool first = lane < ne && (same & lt) == 0;
            const unsigned long long fm = __ballot(first);
            const int fj = __ffsll((unsigned long long) same) - 1;               // first entry of my expert
            const int di = __popcll(fm & ((1ull << fj) - 1));                   // its distinct index
            const int k  = __popcll(same & lt);                                 // my rank among its users
            // position in the expert-grouped entry list: the entries of experts that appear earlier, then my rank
            int before = 0;
            for (int j = 0; j < ne; ++j) { before += __builtin_amdgcn_readlane(fj, j) < fj ? 1 : 0; }
            if (lane < ne) {
                f_ent[before + k] = lane;
                f_tok[before + k] = lane / NU;
                d_ent[di][k] = lane;
                d_tok[di][k] = lane / NU;
                if (first) { d_ex[di] = x; d_n[di] = __popcll(same); }
            }
            if (lane == 0) { d_cnt = __popcll(fm); }
        }
        lds_sync();
        const int ND = d_cnt;
        HTS(8)
        // gate / up: a wave takes 4 consecutive rows of one (token, slot) entry (16 lanes per row, the expert
        // wave-uniform; per entry rather than per distinct expert keeps the waves balanced); xqk holds q8(mixed)
        {
            const int l16 = lane % 16, ii = l16 % 8, sb0 = l16 / 8, p = ii/2, h = ii % 2;
            const int nwi = T*NU*(ffr/4);                    // wave items
            const int gw = bid*(NT/64) + wu_;
            int f = 0, r4 = gw;                              // wave item -> (grouped entry f, first row), no division
            while (r4 >= ffr/4 && f < T*NU) { r4 -= ffr/4; ++f; }
#pragma unroll 1
            for (int it = gw; it < nwi; it += NB*(NT/64)) {
                // entries in expert-grouped order: an expert shared by several tokens runs within one round, so its
                // repeated rows come from L2
                const int slot = __builtin_amdgcn_readfirstlane(f_ent[f]);
                const int t = __builtin_amdgcn_readfirstlane(f_tok[f]), e = slot - t*NU;
                const int ex = __builtin_amdgcn_readfirstlane(sel_e[t][e]);
                const int r = 4*r4 + lane/16;
                const q81 * xt = xqk[t] + 8*sb0 + 2*p;
                float gg = 0.0f, uu = 0.0f;
                // the super-block loop per type (the branch outside the loop keeps each path's registers separate):
                // q4_K 144 B (d/dmin, scales, qs at 16), q5_K 176 B (qh at 16, qs at 48)
                const auto sb_loop = [&](auto q5c) {
                    constexpr bool Q5 = decltype(q5c)::value;
                    constexpr int SB = Q5 ? 176 : 144, QS = Q5 ? 48 : 16;
                    const char * rg_ = mo.w_gate + ex*mo.sg + (long long) r*(10*SB) + sb0*SB;
                    const char * ru  = mo.w_up   + ex*mo.su + (long long) r*(10*SB) + sb0*SB;
#pragma unroll 1
                    for (int k = 0; k < 5; ++k) {
                        // super-block sb0 + 2k of the gate and the up row together: 4 (q5_K 6) x 16 B in flight per lane
                        const char * bg = rg_ + k*2*SB, * bu = ru + k*2*SB;
                        const q81 & xa = xt[16*k], & xb = xt[16*k + 1];
                        if (!Q5) {
                            const int4 hg = *(const int4 *) bg, qg = *(const int4 *) (bg + QS + 16*ii);
                            const int4 hu = *(const int4 *) bu, qu = *(const int4 *) (bu + QS + 16*ii);
                            int xav[4], xbv[4], sA = 0, sB = 0;
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                xav[j] = xa.qs[4*h + j]; xbv[j] = xb.qs[4*h + j];
                                sA = __builtin_amdgcn_sdot4(0x01010101, xav[j], sA, false);
                                sB = __builtin_amdgcn_sdot4(0x01010101, xbv[j], sB, false);
                            }
                            const float d8a = __low2float(xa.ds), d8b = __low2float(xb.ds);
                            const int4 z = make_int4(0, 0, 0, 0);
                            gg += moe_q4k_item<false>(hg, qg, z, xav, xbv, p, d8a, d8b, sA, sB);
                            uu += moe_q4k_item<false>(hu, qu, z, xav, xbv, p, d8a, d8b, sA, sB);
                        } else {
                            // q5_K (one layer of the model): the gate row, then the up row, 3 x 16 B each, to keep the
                            // register pressure of this path below the q4_K one
#pragma unroll 1
                            for (int gu = 0; gu < 2; ++gu) {
                                const char * bb = gu == 0 ? bg : bu;
                                const int4 hd = *(const int4 *) bb, qv = *(const int4 *) (bb + QS + 16*ii);
                                const int4 qh = *(const int4 *) (bb + 16 + 16*h);
                                int xav[4], xbv[4], sA = 0, sB = 0;
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    xav[j] = xa.qs[4*h + j]; xbv[j] = xb.qs[4*h + j];
                                    sA = __builtin_amdgcn_sdot4(0x01010101, xav[j], sA, false);
                                    sB = __builtin_amdgcn_sdot4(0x01010101, xbv[j], sB, false);
                                }
                                const float v = moe_q4k_item<true>(hd, qv, qh, xav, xbv, p, __low2float(xa.ds),
                                                                   __low2float(xb.ds), sA, sB);
                                if (gu == 0) { gg += v; } else { uu += v; }
                            }
                        }
                    }
                };
                if (mo.gq5 == 2) {
                    // q8_0 rows (the MTP draft layer): 40 block pairs, lane pairs l16, l16 + 16, l16 + 32; the gate pair
                    // then the up pair (17 dwords each in flight)
                    const char * rg8 = mo.w_gate + ex*mo.sg + (long long) r*(E/32*34);
                    const char * ru8 = mo.w_up   + ex*mo.su + (long long) r*(E/32*34);
#pragma unroll 1
                    for (int k = 0; k < 3; ++k) {
                        const int pp = l16 + 16*k;
                        if (pp < E/64) {
#pragma unroll 1
                            for (int gu = 0; gu < 2; ++gu) {
                                const int * src = (const int *) ((gu == 0 ? rg8 : ru8) + pp*68);
                                int w8[17];
#pragma unroll
                                for (int q = 0; q < 17; ++q) { w8[q] = src[q]; }
                                const float v = pair_dot(w8, xqk[t][2*pp], xqk[t][2*pp + 1]);
                                if (gu == 0) { gg += v; } else { uu += v; }
                            }
                        }
                    }
                } else if (mo.gq5) {
                    sb_loop(std::true_type{});
                } else {
                    sb_loop(std::false_type{});
                }
                gg = row16_sum(gg); uu = row16_sum(uu);
                if (l16 == 0) { g_hcp_mh[slot][r] = gg/(1.0f + expf(-gg))*uu; }
                r4 += NB*(NT/64);
                while (r4 >= ffr/4) { r4 -= ffr/4; ++f; }
            }
        }
        HTS(9)
        grid_sync_all();
        HTS(10)
        // down inputs -> q8_1 in the arena: [token*NU + slot][NBK] blocks, then the shared expert's [token][ff/32]
        // (up to QU loads per thread in flight: the L2 reads are shared by all blocks)
        q81 * dq = (q81 *) arena;
        {
            const int ffr4 = ffr/4, nm4 = T*NU*ffr4, ff4 = ffn.ff/4, ntot = nm4 + T*ff4;
            constexpr int QU = T < 4 ? 4 : 5;   // one round of loads up to T = 4 (4160 float4 at ffr 384)
#pragma unroll 1
            for (int base = tid; base < ntot; base += QU*NT) {
                float4 v[QU];
#pragma unroll
                for (int u = 0; u < QU; ++u) {
                    const int i4 = min(base + u*NT, ntot - 1);
                    if (i4 < nm4) {
                        v[u] = ((const float4 *) g_hcp_mh[i4 / ffr4])[i4 % ffr4];
                    } else {
                        v[u] = ((const float4 *) g_hcp_h[(i4 - nm4) / ff4])[(i4 - nm4) % ff4];
                    }
                }
#pragma unroll
                for (int u = 0; u < QU; ++u) {
                    const int i4 = base + u*NT;
                    if (i4 < ntot) { moe_quant4(v[u], dq[i4/8], i4 % 8); }
                }
            }
        }
        if (tid < T) { pw[tid] = 1.0f/(1.0f + expf(-g_hcp_sg[tid])); }
        lds_sync();
        HTS(12)
        const q81 * sq = dq + T*NU*NBK;
        const int PD = ffn.ff/64;
        {
            // a wave takes 4 output rows (16 lanes per row: lane b < NBK has K block b, or block pair b < NBK/2 for
            // q8_0) and loops over the distinct experts (wave-uniform) in batches of EB with all loads first; each loaded
            // block serves every token using that expert; the shared expert's PD block pairs of the row on lanes b < PD
            const int l16 = lane % 16;
            const int nbl = mo.dq8 ? NBK/2 : NBK;
            const bool kact = l16 < nbl;
            const int kb = kact ? l16 : 0;
            constexpr int EB = T == 1 ? 4 : 2;
#pragma unroll 1
            for (int jr = 4*wu_; jr < n; jr += 4*(NT/64)) {
                const int  j  = e0 + jr + lane/16;
                const bool ja = jr + lane/16 < n;
                const char * wrow = mo.w_down + (long long) (ja ? j : e0)*mo.rd;
                float acc[T];
#pragma unroll
                for (int t = 0; t < T; ++t) { acc[t] = 0.0f; }
                // the users of distinct expert d (uniform): weight * dot(block, h(entry)) into its token's sum
                const auto users = [&](const int d, const auto & dot) {
                    const int nq = __builtin_amdgcn_readfirstlane(d_n[d]);
#pragma unroll 1
                    for (int q = 0; q < nq; ++q) {
                        const int ent = __builtin_amdgcn_readfirstlane(d_ent[d][q]);
                        const int tq  = __builtin_amdgcn_readfirstlane(d_tok[d][q]);
                        const float v = kact ? sel_w[tq][ent - tq*NU]*dot(ent) : 0.0f;
#pragma unroll
                        for (int t = 0; t < T; ++t) { acc[t] += t == tq ? v : 0.0f; }
                    }
                };
                if (!mo.dq8) {
#pragma unroll 1
                    for (int db = 0; db < ND; db += EB) {
                        int wv[EB][6];
#pragma unroll
                        for (int b = 0; b < EB; ++b) {
                            const int ex = __builtin_amdgcn_readfirstlane(d_ex[min(db + b, ND - 1)]);
                            const int * w = (const int *) (wrow + ex*mo.sd + kb*24);
#pragma unroll
                            for (int q = 0; q < 6; ++q) { wv[b][q] = w[q]; }
                        }
#pragma unroll
                        for (int b = 0; b < EB; ++b) {
                            if (db + b < ND) {
                                users(db + b, [&](const int ent) { return moe_q51_dot(wv[b], dq[ent*NBK + kb]); });
                            }
                        }
                    }
                } else {
#pragma unroll 1
                    for (int d = 0; d < ND; ++d) {
                        const int ex = __builtin_amdgcn_readfirstlane(d_ex[d]);
                        const int * w = (const int *) (wrow + ex*mo.sd + kb*68);
                        int wv[17];
#pragma unroll
                        for (int q = 0; q < 17; ++q) { wv[q] = w[q]; }
                        users(d, [&](const int ent) { return pair_dot(wv, dq[ent*NBK + 2*kb], dq[ent*NBK + 2*kb + 1]); });
                    }
                }
#pragma unroll 1
                for (int t = 0; t < T; ++t) {
                    float ash = 0.0f;
                    if (l16 < PD) {
                        const int * w = (const int *) (ffn.w_sdown + (long long) (ja ? j : e0)*(ffn.ff/32*34) + l16*68);
                        int wv[17];
#pragma unroll
                        for (int q = 0; q < 17; ++q) { wv[q] = w[q]; }
                        ash = pair_dot(wv, sq[t*(ffn.ff/32) + 2*l16], sq[t*(ffn.ff/32) + 2*l16 + 1]);
                    }
                    float a = 0.0f;
#pragma unroll
                    for (int tt = 0; tt < T; ++tt) { a = tt == t ? acc[tt] : a; }
                    a = row16_sum(a); ash = row16_sum(ash);
                    if (l16 == 0 && ja) {
                        const float o = a + ash*pw[t];
                        mo.out[t*E + j] = o;
#pragma unroll
                        for (int k = 0; k < (AR4 ? ARNS : 1); ++k) { if (k < mo.push.n) { mo.push.inbox[k][pofs + t*E + j] = o; } }
                    }
                }
            }
        }
        if (mo.push.n) {
            // fused AllReduce push: this block's rows are in the peers' inboxes; the last block raises the peers' flags
            __threadfence_system();
            __asm__ volatile("s_waitcnt vmcnt(0)" ::: "memory");
            lds_sync();
            if (tid == 0) {
                const unsigned int old = atomicAdd(&g_hcp_pushcnt, 1u);
                if (old % NB == NB - 1) {
                    __threadfence_system();
                    for (int k = 0; k < mo.push.n; ++k) { *(volatile int *) mo.push.flag[k] = (int) push_s; }
                    __threadfence_system();
                    atomicAdd(&g_hcp_arseq, 1u);
                    g_hcp_pushed = 1u;
                }
            }
        }
        HTS(7)
        return;
    }
    // shared expert down: h -> q8_1 in LDS (xqk rows reused), rows [e0, e1) x 10 block pairs, times sigmoid(sg)
    const int ff4 = ffn.ff/4;
    for (int i4 = tid; i4 < T*ff4; i4 += NT) {
        const int t = i4 / ff4, j = i4 % ff4;
        const float4 v = ((const float4 *) g_hcp_h[t])[j];
        const float amax = row8_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        const float d = amax/127.0f, dd = amax == 0.0f ? 1.0f : d; // as quantize_q8_1: q = round(x/d)
        char4 q;
        const float id = __builtin_amdgcn_rcpf(dd); // one reciprocal instead of 4 IEEE divisions (x/d rounds the same
            // except within an ulp of a .5 boundary)
            q.x = roundf(v.x*id); q.y = roundf(v.y*id); q.z = roundf(v.z*id); q.w = roundf(v.w*id);
        xqk[t][j/8].qs[j % 8] = *(int *) &q;
        if (j % 8 == 0) { xqk[t][j/8].ds = make_half2(d, 0.0f); }
    }
    if (tid < T) { pw[tid] = 1.0f/(1.0f + expf(-g_hcp_sg[tid])); }
    lds_sync();
    const int PD = ffn.ff/64;             // block pairs per down row (10, or 5 for a shard)
    if (tid < n*PD) {
        const int i = tid / PD, p = tid % PD;
        const int * wd2 = (const int *) (ffn.w_sdown + (size_t) (e0 + i)*(ffn.ff/32*34) + (size_t) p*68);
        int w[17];
#pragma unroll
        for (int k = 0; k < 17; ++k) { w[k] = wd2[k]; }
#pragma unroll
        for (int t = 0; t < T; ++t) { part[t][tid] = pair_dot(w, xqk[t][2*p], xqk[t][2*p + 1]); }
    }
    lds_sync();
    for (int it = tid; it < T*n; it += NT) {
        const int t = it / n, i = it % n;
        float s2 = 0.0f;
        for (int p = 0; p < PD; ++p) { s2 += part[t][i*PD + p]; }
        g_hcp_shexp[t][e0 + i] = s2*pw[t];
    }
    HTS(7)
#undef HTS
#endif // gfx9 device code
}

} // namespace

int ggml_cuda_hc_persist_mode() {
    static const int mode = [] { const char * e = getenv("GGML_CUDA_HC_PERSIST"); return e ? atoi(e) : 0; }();
    return mode;
}

bool ggml_cuda_hc_persist_enabled() {
    return ggml_cuda_hc_persist_mode() > 0;
}

// the 10-node chain at i (RMS_NORM): returns the token count (1..4), or 0
static int hc_persist_chain(int device, const ggml_cgraph * cgraph, int i) {
    if (!ggml_cuda_hc_persist_enabled() || i + 9 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_cuda_device_info::cuda_device_info & info = ggml_cuda_info().devices[device];
    // needs every CU of the GPU (grid barrier) and owns per-GPU globals: one (virtual) device per physical GPU only
    if (!GGML_CUDA_CC_IS_GCN(info.cc) || info.nsm != NB || info.physical_share_count != 1) {
        return 0;
    }
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * norm = nd[0], * mul = nd[1];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || nd[2]->op != GGML_OP_RESHAPE || nd[3]->op != GGML_OP_RESHAPE ||
            nd[4]->op != GGML_OP_MUL_MAT || nd[5]->op != GGML_OP_SCALE || nd[6]->op != GGML_OP_UNARY ||
            ggml_get_unary_op(nd[6]) != GGML_UNARY_OP_SILU || nd[7]->op != GGML_OP_MUL_MAT || nd[8]->op != GGML_OP_RESHAPE ||
            nd[9]->op != GGML_OP_DSV4_HC_PRE) {
        return 0;
    }
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * wn = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * r2d = nd[2]->ne[0] == K ? nd[2] : nd[3];  // xn as [10240, T] for the down projection
    const ggml_tensor * r3d = nd[2]->ne[0] == K ? nd[3] : nd[2];  // xn as [2560, 4, T] for HC_PRE
    const ggml_tensor * down = nd[4]->src[0], * up = nd[7]->src[0];
    const int64_t T = x->ne[2];
    const bool ok = T >= 1 && T <= TMAX &&
        (mul->src[0] == norm || mul->src[1] == norm) && r2d->src[0] == mul && (r3d->src[0] == mul || r3d->src[0] == r2d) &&
        x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[0] == E && x->ne[1] == HC && x->ne[3] == 1 &&
        wn->type == GGML_TYPE_F32 && ggml_is_contiguous(wn) && ggml_nelements(wn) == K &&
        mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && ggml_are_same_shape(mul, x) &&
        nd[4]->src[1] == r2d && down->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(down) && down->ne[0] == K && down->ne[1] == LR &&
        nd[5]->src[0] == nd[4] && ggml_get_op_params_f32(nd[5], 1) == 0.0f && nd[6]->src[0] == nd[5] &&
        nd[7]->src[1] == nd[6] && up->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(up) && up->ne[0] == LR && up->ne[1] == K &&
        nd[8]->src[0] == nd[7] && nd[9]->src[0] == r3d && nd[9]->src[1] == nd[8] && ggml_get_op_params_i32(nd[9], 1) != 0 &&
        nd[9]->type == GGML_TYPE_F32 && ggml_is_contiguous(nd[9]) && nd[9]->ne[0] == E && ggml_nrows(nd[9]) == T &&
        ((uintptr_t) down->data) % 4 == 0 && ((uintptr_t) up->data) % 4 == 0;
    return ok ? (int) T : 0;
}

bool ggml_cuda_hc_persist_match(int device, const ggml_cgraph * cgraph, int i) {
    return hc_persist_chain(device, cgraph, i) > 0 &&
        ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT,
            GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE }, { i + 1, i + 2, i + 3, i + 9 });
}

static bool hc_persist_inject_on();

bool ggml_cuda_hc_persist_post_match(int device, const ggml_cgraph * cgraph, int i, int inj) {
    if (ggml_cuda_hc_persist_mode() < 2 || i + 13 >= cgraph->n_nodes || (inj >= 0 && !hc_persist_inject_on())) {
        return false;
    }
    ggml_tensor * const * nd = cgraph->nodes + i;
    if (nd[0]->op != GGML_OP_SCALE || nd[1]->op != GGML_OP_UNARY || ggml_get_unary_op(nd[1]) != GGML_UNARY_OP_SIGMOID ||
            nd[2]->op != GGML_OP_SCALE || nd[3]->op != GGML_OP_DSV4_HC_POST || nd[1]->src[0] != nd[0] || nd[2]->src[0] != nd[1] ||
            nd[3]->src[2] != nd[2] || nd[3]->src[3] != nullptr || nd[4]->src[0] != nd[3]) {
        return false;
    }
    const int T = hc_persist_chain(device, cgraph, i + 4);
    const ggml_tensor * raw = nd[0]->src[0], * out = nd[3]->src[0], * res = nd[3]->src[1];
    const bool ok = T > 0 && ggml_get_op_params_f32(nd[0], 1) == 0.0f && ggml_get_op_params_f32(nd[2], 1) == 0.0f &&
        raw->type == GGML_TYPE_F32 && ggml_is_contiguous(raw) && raw->ne[0] == HC && ggml_nrows(raw) == T &&
        out->type == GGML_TYPE_F32 && ggml_is_contiguous(out) && out->ne[0] == E && ggml_nrows(out) == T &&
        res->type == GGML_TYPE_F32 && ggml_is_contiguous(res) && ggml_are_same_shape(res, nd[3]) && ggml_is_contiguous(nd[3]);
    if (!ok) {
        return false;
    }
    if (inj >= 0) {
        // the inject MUL_MAT (its result is in the stash) joins the fused range and is never computed
        return inj == i - 1 && raw == cgraph->nodes[inj] &&
            ggml_can_fuse_subgraph(cgraph, inj, { GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST,
                GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT,
                GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE }, { i + 3, i + 5, i + 6, i + 7, i + 13 });
    }
    return ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST,
            GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT,
            GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE }, { i + 3, i + 5, i + 6, i + 7, i + 13 });
}

// mode >= 3 inject fusion (the stash): opt-in GGML_CUDA_HCP_INJ=1. Off by default since 2026-09-29: 2+2 decode is faster
// without it (64.2 vs 63.0 t/s), and under 4-GPU tensor parallel it is wrong (KLD 0.27; cause not found).
// KLD references with it off: 2+2 t1 0.026132 t4 0.020113 l1 0.023430 l4 0.028213; TP4 t1 0.026311 t4 0.022200
static bool hc_persist_inject_on() {
    static const bool on = [] { const char * v = getenv("GGML_CUDA_HCP_INJ"); return v && atoi(v) != 0; }();
    return on && ggml_cuda_hc_persist_mode() >= 3;
}

// mode 3: the inject MUL_MAT of this sublayer (w_inject x xn, much later in the graph) if its result can be produced here
static int hc_persist_find_inject(const ggml_cgraph * cgraph, int i) {
    if (!hc_persist_inject_on()) {
        return -1;
    }
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * r2d = nd[2]->ne[0] == K ? nd[2] : nd[3];
    const int64_t T = nd[0]->src[0]->ne[2];
    for (int j = i + 10; j < cgraph->n_nodes && j < i + 4000; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_MUL_MAT && t->src[1] == r2d) {
            const ggml_tensor * w = t->src[0];
            const bool ok = w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == K && w->ne[1] == HC &&
                w->ne[2] == 1 && w->ne[3] == 1 && ((uintptr_t) w->data) % 16 == 0 &&
                t->type == GGML_TYPE_F32 && t->ne[0] == HC && t->ne[1] == T;
            return ok ? j : -1;
        }
    }
    return -1;
}

static hc_ar hc_persist_take_ar(ggml_backend_cuda_context & ctx, const ggml_tensor * out);
static bool hcp_ar_group4(int device);
static void hcp_ar_push_target(int device, hc_push * push);

// per-phase timestamps (GGML_CUDA_HCP_TS=1, run with GGML_CUDA_DISABLE_GRAPHS=1): ring of launches per device, summarized
// once after a warm-up by variant (max over blocks at each phase point, minus the earliest block start)
struct hcp_ts_dev {
    unsigned long long * buf = nullptr;
    std::vector<std::string> tags;
    int n = 0, skipped = 0;
    bool done = false;
};
static constexpr int HCP_TS_SLOTS = 1536, HCP_TS_SKIP = 480;
static hcp_ts_dev g_hcp_ts[GGML_CUDA_MAX_DEVICES];

static unsigned long long * hcp_ts_slot(ggml_backend_cuda_context & ctx, const std::string & tag) {
    static const bool en = [] { const char * e = getenv("GGML_CUDA_HCP_TS"); return e && atoi(e) != 0; }();
    hcp_ts_dev & d = g_hcp_ts[ctx.device];
    if (!en || d.done) {
        return nullptr;
    }
    if (d.skipped < HCP_TS_SKIP) {
        d.skipped++;
        return nullptr;
    }
    if (d.buf == nullptr) {
        CUDA_CHECK(cudaMalloc(&d.buf, (size_t) HCP_TS_SLOTS*NB*HCP_TS_N*sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d.buf, 0, (size_t) HCP_TS_SLOTS*NB*HCP_TS_N*sizeof(unsigned long long)));
    }
    if (d.n < HCP_TS_SLOTS) {
        d.tags.push_back(tag);
        return d.buf + (size_t) (d.n++)*NB*HCP_TS_N;
    }
    // full: summarize once
    d.done = true;
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    std::vector<unsigned long long> h((size_t) HCP_TS_SLOTS*NB*HCP_TS_N);
    CUDA_CHECK(cudaMemcpy(h.data(), d.buf, h.size()*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    const int khz = 25000; // s_memrealtime runs at 25 MHz on gfx906
    // phase points in order: A, bar1, mixed, bar2, E, bar3, [top-k, gate/up, bar4,] end
    static const int order[] = { 0, 15, 1, 2, 3, 4, 14, 5, 6, 11, 8, 9, 10, 12, 7 };
    constexpr int NP = sizeof(order)/sizeof(order[0]);
    struct agg { int n = 0; double t[NP] = {}; };
    std::map<std::string, agg> m;
    for (int l = 0; l < d.n; ++l) {
        const unsigned long long * b = h.data() + (size_t) l*NB*HCP_TS_N;
        unsigned long long t0 = ~0ull;
        for (int k = 0; k < NB; ++k) { t0 = std::min(t0, b[k*HCP_TS_N]); }
        agg & a = m[d.tags[l]];
        a.n++;
        double prev = 0;
        for (int q = 0; q < NP; ++q) {
            const int p = order[q];
            unsigned long long mx = 0;
            bool any = false;
            for (int k = 0; k < NB; ++k) { if (b[k*HCP_TS_N + p]) { any = true; mx = std::max(mx, b[k*HCP_TS_N + p] - t0); } }
            const double us = any ? mx*1000.0/khz : prev;
            a.t[q] += us;
            prev = us;
        }
    }
    fprintf(stderr, "hcp phase times dev %d (us from the first block start; max over blocks; mean over launches)\n", ctx.device);
    fprintf(stderr, "  %-52s %5s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s\n", "variant", "n", "start", "arwait", "A", "bar1", "mixed", "bar2", "Eq8", "E", "bar3",
            "rtr", "topk", "gu", "bar4", "dquant", "end");
    for (auto & kv : m) {
        fprintf(stderr, "  %-52s %5d", kv.first.c_str(), kv.second.n);
        for (int q = 0; q < NP; ++q) { fprintf(stderr, " %7.2f", kv.second.t[q]/kv.second.n); }
        fprintf(stderr, "\n");
    }
    return nullptr;
}

template <bool POST, bool STASH, bool INJ>
static void hc_persist_launch_t(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, const hc_src & src, float * xpost,
                                const hc_ffn & ffn, const hc_early & early, const hc_ar & ar) {
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * norm = nd[0], * mul = nd[1];
    const ggml_tensor * wn = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const float lo_scale  = ggml_get_op_params_f32(nd[5], 0);
    const float pre_scale = ggml_get_op_params_f32(nd[9], 0);
    const float * w = (const float *) wn->data;
    const char * wd = (const char *) nd[4]->src[0]->data, * wu = (const char *) nd[7]->src[0]->data;
    float * xn = (float *) mul->data, * mixed = (float *) nd[9]->data;
    cudaStream_t st = ctx.stream();
    char tag[160];
    snprintf(tag, sizeof(tag), "T%d post%d stash%d inj%d ffn%d(ff%d) moe%d early%d/%d ab%d ar%d", (int) norm->src[0]->ne[2], (int) POST,
             (int) STASH, (int) INJ, ffn.on, ffn.on ? ffn.ff : 0, ffn.moe.on ? ffn.moe.ffr : 0, early.nq, early.rows_total, early.H,
             ar.inbox_mine != nullptr);
    unsigned long long * hts = hcp_ts_slot(ctx, tag);
    static const bool dbg = getenv("GGML_CUDA_HC_PERSIST_DEBUG") != nullptr;
    static int ndbg = 0;
    if (dbg && ndbg < 8) { ndbg++; fprintf(stderr, "hc_persist: %s T=%d post=%d stash=%d inj=%d\n", nd[9]->name, (int) norm->src[0]->ne[2], (int) POST, (int) STASH, (int) INJ); }
    const bool ar4 = hcp_ar_group4(ctx.device);
    switch (norm->src[0]->ne[2]) {
#define HCP_K(TT) do { if (ar4) { hc_persist_kernel<TT, POST, STASH, INJ, true><<<NB, NT, 0, st>>>(src, w, wd, wu, eps, lo_scale, pre_scale, xpost, xn, mixed, ffn, early, ar, hts); } \
                        else     { hc_persist_kernel<TT, POST, STASH, INJ, false><<<NB, NT, 0, st>>>(src, w, wd, wu, eps, lo_scale, pre_scale, xpost, xn, mixed, ffn, early, ar, hts); } } while (0)
        case 1: HCP_K(1); break;
        case 2: HCP_K(2); break;
        case 3: HCP_K(3); break;
        case 4: HCP_K(4); break;
#undef HCP_K
        default: GGML_ABORT("hc_persist: token count");
    }
    CUDA_CHECK(cudaGetLastError());
}

// mode 4: the router logits MUL_MAT right after the chain (joins the fused range) and the shared expert run of the same
// FFN (MUL_MAT gate, MUL_MAT up, SWIGLU, MUL_MAT down, MUL_MAT gate logit, SIGMOID, MUL, ADD) further on, computed into
// the stash. Returns the shared expert start index (-1: no extension).
static int hc_persist_ffn(const ggml_cgraph * cgraph, int i, hc_ffn * f) {
    if (ggml_cuda_hc_persist_mode() < 4 || i + 10 >= cgraph->n_nodes) {
        return -1;
    }
    ggml_tensor * const * nd = cgraph->nodes;
    const ggml_tensor * mixed = nd[i + 9], * lg = nd[i + 10];
    const int64_t T = mixed->ne[1];
    if (lg->op != GGML_OP_MUL_MAT || lg->src[1] != mixed || lg->src[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(lg->src[0]) ||
            lg->src[0]->ne[0] != E || lg->src[0]->ne[1] != NR || lg->type != GGML_TYPE_F32 || !ggml_is_contiguous(lg) ||
            lg->ne[0] != NR || lg->ne[1] != T) {
        return -1;
    }
    auto q8w = [](const ggml_tensor * w, int64_t ne0, int64_t ne1) {
        return w->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(w) && w->ne[0] == ne0 && w->ne[1] == ne1 && ((uintptr_t) w->data) % 4 == 0;
    };
    for (int j = i + 11; j + 7 < cgraph->n_nodes && j < i + 400; ++j) {
        const ggml_tensor * g = nd[j];
        const int64_t ff = g->src[0]->ne[1];
        // this device's shared-expert shard: the whole FF, or a 64-row multiple of it (2 devices: 320; 4: 128/192)
        if (g->op != GGML_OP_MUL_MAT || g->src[1] != mixed || ff < 64 || ff > FF || ff % 64 != 0 || !q8w(g->src[0], E, ff)) {
            continue;
        }
        const ggml_tensor * u = nd[j + 1], * sw = nd[j + 2], * dn = nd[j + 3], * sg = nd[j + 4], * sig = nd[j + 5], * mul = nd[j + 6], * add = nd[j + 7];
        const bool ok = u->op == GGML_OP_MUL_MAT && u->src[1] == mixed && q8w(u->src[0], E, ff) &&
            sw->op == GGML_OP_GLU && ggml_get_glu_op(sw) == GGML_GLU_OP_SWIGLU && sw->src[0] == g && sw->src[1] == u &&
            ggml_get_op_params_i32(sw, 1) == 0 && // not swapped
            dn->op == GGML_OP_MUL_MAT && dn->src[1] == sw && q8w(dn->src[0], ff, E) &&
            sg->op == GGML_OP_MUL_MAT && sg->src[1] == mixed && sg->src[0]->type == GGML_TYPE_F32 && ggml_nelements(sg->src[0]) == E &&
            sig->op == GGML_OP_UNARY && ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->src[0] == sg &&
            mul->op == GGML_OP_MUL && mul->src[0] == dn && mul->src[1] == sig &&
            add->op == GGML_OP_ADD && (add->src[0] == mul || add->src[1] == mul) && ggml_is_contiguous(add) &&
            ggml_nrows(add) == T && add->ne[0] == E;
        if (!ok) {
            return -1;
        }
        f->w_logits = (const float *) lg->src[0]->data;
        f->w_sgate  = (const char *) g->src[0]->data;
        f->w_sup    = (const char *) u->src[0]->data;
        f->w_sdown  = (const char *) dn->src[0]->data;
        f->w_sg     = (const float *) sg->src[0]->data;
        f->logits   = (float *) lg->data;
        f->on       = 1;
        f->ff       = (int) ff;
        return j;
    }
    return -1;
}

// routed experts: the nodes between the router logits (i + 10) and the shared expert run at js (softmax top-k group,
// expert gate / up / SwiGLU / down, routing weights, the expert sum) plus the shared expert run up to the FFN output ADD
// at js + 7, all computed by the persistent kernel when no intermediate is read outside that range
static bool hc_persist_moe(const ggml_cgraph * cgraph, int i, int js, int T, int ff, hc_moe * mo) {
    *mo = {};
    static const bool disabled = [] { const char * v = getenv("GGML_CUDA_HCP_MOE"); return v && atoi(v) == 0; }();
    if (disabled || js < 0 || js + 7 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * const * nd = cgraph->nodes;
    const ggml_tensor * mixed = nd[i + 9], * logits = nd[i + 10];
    const int k = i + 11;
    if (k + 10 > js) {
        return false;
    }
    // softmax top-k group (as the topk_moe fusion: SOFT_MAX RESHAPE ARGSORT VIEW GET_ROWS RESHAPE SUM_ROWS CLAMP DIV RESHAPE)
    const ggml_tensor * sm = nd[k], * pr = nd[k + 1], * as = nd[k + 2], * ids = nd[k + 3], * gr = nd[k + 4], * w2 = nd[k + 5],
                      * su = nd[k + 6], * cl = nd[k + 7], * dv = nd[k + 8], * wts = nd[k + 9];
    if (sm->op != GGML_OP_SOFT_MAX || sm->src[0] != logits || sm->src[1] != nullptr || ggml_get_op_params_f32(sm, 0) != 1.0f ||
            ggml_get_op_params_f32(sm, 1) != 0.0f ||
            pr->op != GGML_OP_RESHAPE || pr->src[0] != sm || as->op != GGML_OP_ARGSORT || as->src[0] != sm ||
            ggml_get_op_params_i32(as, 0) != GGML_SORT_ORDER_DESC || ids->op != GGML_OP_VIEW || ids->src[0] != as ||
            gr->op != GGML_OP_GET_ROWS || gr->src[0] != pr || gr->src[1] != ids || w2->op != GGML_OP_RESHAPE || w2->src[0] != gr ||
            su->op != GGML_OP_SUM_ROWS || su->src[0] != w2 || cl->op != GGML_OP_CLAMP || cl->src[0] != su ||
            dv->op != GGML_OP_DIV || dv->src[0] != w2 || dv->src[1] != cl || wts->op != GGML_OP_RESHAPE || wts->src[0] != dv) {
        return false;
    }
    const int nused = (int) ids->ne[0];
    if (nused < 1 || nused > NUMAX || ids->ne[1] != T || ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    // expert nodes
    const ggml_tensor * gate = nullptr, * up = nullptr, * glu = nullptr, * down = nullptr, * mul = nullptr, * last_add = nullptr;
    int n_view = 0, n_add = 0;
    for (int j = k + 10; j < js; ++j) {
        const ggml_tensor * t = nd[j];
        switch (t->op) {
            case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_NONE:
                n_view += t->op == GGML_OP_VIEW;
                break;
            case GGML_OP_MUL_MAT_ID:
                if (t->src[2] != ids) { return false; }
                if (t->src[1]->view_src == mixed || t->src[1] == mixed) {
                    if (gate == nullptr) { gate = t; } else if (up == nullptr) { up = t; } else { return false; }
                } else if (down == nullptr) { down = t; } else { return false; }
                break;
            case GGML_OP_GLU:
                if (glu != nullptr) { return false; }
                glu = t;
                break;
            case GGML_OP_MUL:
                if (mul != nullptr) { return false; }
                mul = t;
                break;
            case GGML_OP_ADD:
                n_add++;
                last_add = t;
                break;
            default:
                return false;
        }
    }
    if (!gate || !up || !glu || !down || !mul || n_add != nused - 1 || n_view != nused) {
        return false;
    }
    if (glu->src[1] == gate && glu->src[0] == up) { std::swap(gate, up); }
    const ggml_tensor * wg = gate->src[0], * wu = up->src[0], * wd = down->src[0];
    static const bool q5k_on = [] { const char * v = getenv("GGML_CUDA_HCP_MOE_Q5K"); return !v || atoi(v) != 0; }();
    const int64_t ffr = wg->ne[1];
    const bool ok = ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && glu->src[0] == gate && glu->src[1] == up &&
        ggml_get_op_params_i32(glu, 1) == 0 && down->src[1] == glu && mul->src[0] == down && mul->src[1] == wts &&
        (wg->type == GGML_TYPE_Q4_K || (wg->type == GGML_TYPE_Q5_K && q5k_on) || wg->type == GGML_TYPE_Q8_0) &&
        wu->type == wg->type && (wg->type != GGML_TYPE_Q8_0 || (((uintptr_t) wg->data) % 4 == 0 && ((uintptr_t) wu->data) % 4 == 0)) &&
        ggml_is_contiguous(wg) && ggml_is_contiguous(wu) &&
        wg->ne[0] == E && wu->ne[0] == E && wu->ne[1] == ffr && wg->ne[2] == NR && wu->ne[2] == NR &&
        ffr % 64 == 0 && ffr <= FFRMAX && ffr/32 <= 16 && ff/64 <= 16 && ((uintptr_t) wg->data) % 16 == 0 && ((uintptr_t) wu->data) % 16 == 0 &&
        (wd->type == GGML_TYPE_Q5_1 || wd->type == GGML_TYPE_Q8_0) && ggml_is_contiguous(wd) && wd->ne[0] == ffr &&
        wd->ne[1] == E && wd->ne[2] == NR && ((uintptr_t) wd->data) % 4 == 0 &&
        gate->src[1]->ne[0] == E && ggml_nelements(gate->src[1]) == (int64_t) E*T &&
        // the FFN output: moe_out (the last expert ADD) + the gated shared expert
        (nd[js + 7]->src[0] == last_add || nd[js + 7]->src[1] == last_add) && nd[js + 7]->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(nd[js + 7]);
    if (!ok) {
        return false;
    }
    // the down inputs staged in the kernel's arena (T*(nused*ffr + ff)/32 q8_1 blocks)
    const int arena = T < 4 ? T*E*4 : 4*4800;   // as the kernel's ARENA
    if ((int) (T*(nused*ffr + ff)/32*sizeof(q81)) > std::max(arena, T*NT*4)) {
        return false;
    }
    // no intermediate may be read outside [k, js + 7] or be a graph output
    for (int j = k; j < js + 7; ++j) {
        const ggml_tensor * t = nd[j];
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int inside = 0;
        for (int q = k; q <= js + 7; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) { inside += nd[q]->src[s] == t; }
        }
        if (ggml_node_get_use_count(cgraph, j) != inside) {
            return false;
        }
    }
    mo->gq5 = wg->type == GGML_TYPE_Q5_K ? 1 : wg->type == GGML_TYPE_Q8_0 ? 2 : 0;
    mo->w_gate = (const char *) wg->data;
    mo->w_up   = (const char *) wu->data;
    mo->w_down = (const char *) wd->data;
    mo->sg = (long long) wg->nb[2];
    mo->su = (long long) wu->nb[2];
    mo->sd = (long long) wd->nb[2];
    mo->rd = (long long) wd->nb[1];
    mo->ffr = (int) ffr;
    mo->dq8 = wd->type == GGML_TYPE_Q8_0;
    mo->nused = nused;
    mo->clamp = ggml_get_op_params_f32(cl, 0);
    mo->out = (float *) nd[js + 7]->data;
    mo->on = 1;
    static const bool dedup = [] { const char * v = getenv("GGML_CUDA_HCP_MOE_DEDUP"); return !v || atoi(v) != 0; }();
    mo->dedup = dedup;
    return true;
}

int ggml_cuda_hc_persist_ffn_ext(const ggml_cgraph * cgraph, int i) {
    hc_ffn f = {};
    return hc_persist_ffn(cgraph, i, &f) >= 0 ? 1 : 0;
}

bool ggml_cuda_hc_persist_shexp_match(int device, const ggml_cgraph * cgraph, int i) {
    // i: the shared expert gate MUL_MAT; i+8: the inject MUL_MAT (stash); i+9: the combine SCALE
    if (ggml_cuda_hc_persist_mode() < 4 || i + 22 >= cgraph->n_nodes || cgraph->nodes[i + 7]->op != GGML_OP_ADD ||
            cgraph->nodes[i + 9]->src[0] != cgraph->nodes[i + 8] || cgraph->nodes[i + 12]->src[0] != cgraph->nodes[i + 7] ||
            !ggml_cuda_hc_persist_post_match(device, cgraph, i + 9, i + 8)) {
        return false;
    }
    return ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT,
            GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD,
            GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST,
            GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT,
            GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE }, { i + 12, i + 14, i + 15, i + 16, i + 22 });
}

// the 8 shared expert nodes of this split replaced by ffn_out = moe_out + stash (the combine is in the next split)
static __global__ void hc_persist_shexp_add(const float * __restrict__ moe, float * __restrict__ out, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = moe[i] + g_hcp_shexp[i / E][i % E];
    }
}

bool ggml_cuda_hc_persist_shexp_local_match(const ggml_cgraph * cgraph, int i) {
    if (ggml_cuda_hc_persist_mode() < 4 || i + 7 >= cgraph->n_nodes || cgraph->nodes[i + 7]->op != GGML_OP_ADD) {
        return false;
    }
    return ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT,
            GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD }, { i + 7 });
}

void ggml_cuda_hc_persist_shexp_local(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * add = nd[7];
    const ggml_tensor * moe_out = add->src[0] == nd[6] ? add->src[1] : add->src[0];
    const int n = (int) ggml_nelements(add);
    hc_persist_shexp_add<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const float *) moe_out->data, (float *) add->data, n);
    CUDA_CHECK(cudaGetLastError());
}

// mode 4: MUL_MATs of mixed (q8_0, K = 2560) and the GDN gate group within a short window after the fused range at
// [i, i_end], computed by the persistent kernel when their outputs do not overlap anything the nodes in between touch
static bool hc_persist_is_view_or_noop(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE;
}

static void hc_persist_early(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int i_end, hc_early * e,
                             const std::vector<const ggml_tensor *> & own, int shexp_start) {
    *e = {};
    if (ggml_cuda_hc_persist_mode() < 4) {
        return;
    }
    static const bool disabled = [] { const char * v = getenv("GGML_CUDA_HC_PERSIST_EARLY"); return v && atoi(v) == 0; }();
    if (disabled) {
        return;
    }
    ggml_tensor * const * nd = cgraph->nodes;
    const ggml_tensor * mixed = nd[i + 9];
    const int64_t T = mixed->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr || ggml_is_empty(a) || ggml_is_empty(b)) {
            return false;
        }
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    std::vector<const ggml_tensor *> outs = own;          // memory the kernel reads or writes
    std::vector<std::pair<int, int>> skip;                // accepted (start, length)
    const int end = std::min(cgraph->n_nodes, i_end + 80);
    for (int j = i_end + 1; j < end; ++j) {
        if (shexp_start >= 0 && j >= shexp_start && j < shexp_start + 8) {
            continue; // the shared expert run belongs to the FFN extension
        }
        const ggml_tensor * t = nd[j];
        int len = 0;
        std::vector<const ggml_tensor *> produced;
        if (t->op == GGML_OP_MUL_MAT && t->src[1] == mixed && t->src[0]->type == GGML_TYPE_Q8_0 && e->nq < HCP_EARLY_Q8) {
            const ggml_tensor * w = t->src[0];
            // big matrices stay with MMVQ (tuned for GCN, faster than this 8-lane path); small ones save a launch
            static const int max_rows = [] { const char * v = getenv("GGML_CUDA_HC_PERSIST_EARLY_MAXROWS"); return v ? atoi(v) : 1 << 30; }();
            if (w->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(w) && w->ne[0] == E && w->ne[2] == 1 && ((uintptr_t) w->data) % 4 == 0 &&
                    w->ne[1] <= max_rows &&
                    t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == w->ne[1] && t->ne[1] == T) {
                len = 1;
                produced = { t };
            }
        } else if (t->op == GGML_OP_MUL_MAT && t->src[1] == mixed && e->H == 0 && j + 8 < cgraph->n_nodes &&
                ggml_can_fuse_subgraph(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
                                       GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY }, { j + 5, j + 8 }) &&
                ggml_cuda_gdn_ab_dec_supported(cc, t, nd[j + 2], nd[j + 3], nd[j + 4], nd[j + 6], nd[j + 8])) {
            len = 9;
            produced = { nd[j + 4], nd[j + 8] };
        }
        if (len > 0) {
            // outputs may not overlap the kernel's memory, other early outputs, or anything run in between
            bool ok = true;
            for (const ggml_tensor * o : produced) {
                for (const ggml_tensor * u : outs) { ok = ok && !overlaps(o, u); }
                for (int k = i_end + 1; k < j && ok; ++k) {
                    bool skipped = false;
                    for (auto & sk : skip) { skipped = skipped || (k >= sk.first && k < sk.first + sk.second); }
                    const ggml_tensor * u = nd[k];
                    if (skipped || ggml_is_empty(u) || hc_persist_is_view_or_noop(u) || (u->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                        continue;
                    }
                    ok = !overlaps(o, u);
                    for (int q = 0; q < GGML_MAX_SRC && ok; ++q) { ok = !overlaps(o, u->src[q]); }
                }
            }
            if (ok) {
                if (len == 1) {
                    e->w[e->nq] = (const char *) t->src[0]->data;
                    e->out[e->nq] = (float *) t->data;
                    e->rows[e->nq] = (int) t->src[0]->ne[1];
                    e->rows_total += e->rows[e->nq];
                    e->nq++;
                } else {
                    const ggml_tensor * mul = nd[j + 4], * sp = mul->src[0]->op == GGML_OP_UNARY ? mul->src[0] : mul->src[1];
                    e->wa = (const float *) t->src[0]->data;
                    e->wb = (const float *) nd[j + 6]->src[0]->data;
                    e->dt = (const float *) nd[j + 2]->src[1]->data;
                    e->A  = (const float *) (mul->src[0] == sp ? mul->src[1] : mul->src[0])->data;
                    e->gate = (float *) mul->data;
                    e->beta = (float *) nd[j + 8]->data;
                    e->H = (int) t->src[0]->ne[1];
                }
                skip.push_back({ j, len });
                for (const ggml_tensor * o : produced) { outs.push_back(o); }
                ctx.hcp_done.push_back({ t, len });
                j += len - 1;
            }
        }
    }
}

template <bool POST, bool STASH>
static void hc_persist_launch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, hc_src src, float * xpost) {
    const int j = hc_persist_find_inject(cgraph, i);
    ctx.hcp_inject_node = j >= 0 ? cgraph->nodes[j] : nullptr;
    ctx.hcp_seq++;
    const ggml_tensor * r2d = cgraph->nodes[i + 2]->ne[0] == K ? cgraph->nodes[i + 2] : cgraph->nodes[i + 3];
    const void * w_down = cgraph->nodes[i + 4]->src[0]->data;
    const void * w_other = nullptr; // -sm tensor: the inject weight of this producer, recorded by its consumer in a later split
    if (j < 0 && hc_persist_inject_on()) {
        const auto it = ctx.hcp_inj_map.find(w_down);
        if (it != ctx.hcp_inj_map.end() && ((uintptr_t) it->second) % 16 == 0) {
            w_other = it->second;
        }
    }
    ctx.hcp_last_down = w_down;
    ctx.hcp_last_xn   = r2d->data;
    ctx.hcp_last_injw = w_other;
    ctx.hcp_last_seq  = ctx.hcp_seq;
    hc_ffn ffn = {};
    const int js = hc_persist_ffn(cgraph, i, &ffn);
    ctx.hcp_shexp_node = js >= 0 ? cgraph->nodes[js] : nullptr;
    if (js >= 0 && hc_persist_moe(cgraph, i, js, (int) cgraph->nodes[i + 9]->ne[1], ffn.ff, &ffn.moe)) {
        // the routed experts and the shared expert run through the FFN output: skip them all
        ctx.hcp_done.push_back({ cgraph->nodes[i + 11], js + 7 - (i + 11) + 1 });
        ctx.hcp_shexp_node = nullptr;
        // -sm tensor: the FFN output ends this graph split and is all-reduced next; the kernel pushes it to the peer
        if (js + 7 == cgraph->n_nodes - 1) {
            hcp_ar_push_target(ctx.device, &ffn.moe.push);
        }
    }
    // memory the kernel itself touches, then the early matvecs after the fused range (chain [+ logits])
    std::vector<const ggml_tensor *> own = { cgraph->nodes[i]->src[0], cgraph->nodes[i + 1], cgraph->nodes[i + 9] };
    if (ffn.on) { own.push_back(cgraph->nodes[i + 10]); }
    if (xpost) {
        own.push_back(cgraph->nodes[i - 1]); // the combine's inputs are only read before the first barrier, long before any early write
    }
    hc_early early;
    hc_persist_early(ctx, cgraph, i, i + 9 + (ffn.on ? 1 : 0), &early, own, js);
    // fused AllReduce: the consumed sublayer output (the combine's src0) was deferred by the comm backend
    const hc_ar ar = POST ? hc_persist_take_ar(ctx, cgraph->nodes[i - 1]->src[0]) : hc_ar{};
    if (j >= 0 || w_other != nullptr) {
        src.w_inj = j >= 0 ? (const float *) cgraph->nodes[j]->src[0]->data : (const float *) w_other;
        hc_persist_launch_t<POST, STASH, true>(ctx, cgraph, i, src, xpost, ffn, early, ar);
    } else {
        hc_persist_launch_t<POST, STASH, false>(ctx, cgraph, i, src, xpost, ffn, early, ar);
    }
}

void ggml_cuda_hc_persist(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const hc_src src = { (const float *) cgraph->nodes[i]->src[0]->data, nullptr, nullptr, nullptr, 1.0f, 1.0f, 0, nullptr };
    hc_persist_launch<false, false>(ctx, cgraph, i, src, nullptr);
}

void ggml_cuda_hc_persist_post(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, bool raw_from_stash) {
    ggml_tensor * const * nd = cgraph->nodes + i;
    const hc_src src = { (const float *) nd[3]->src[1]->data, (const float *) nd[3]->src[0]->data, (const float *) nd[0]->src[0]->data,
                         nullptr, ggml_get_op_params_f32(nd[0], 0), ggml_get_op_params_f32(nd[2], 0), 0, nullptr };
    if (raw_from_stash) {
        hc_persist_launch<true, true>(ctx, cgraph, i + 4, src, (float *) nd[3]->data);
    } else {
        hc_persist_launch<true, false>(ctx, cgraph, i + 4, src, (float *) nd[3]->data);
    }
}

void ggml_cuda_hc_persist_shexp(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    // the FFN output is moe_out + the shared expert stash: read moe_out (the ADD's other operand) with shx
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * add = nd[7];
    const ggml_tensor * moe_out = add->src[0] == nd[6] ? add->src[1] : add->src[0];
    const int s = i + 9;
    const hc_src src = { (const float *) cgraph->nodes[s + 3]->src[1]->data, (const float *) moe_out->data, nullptr,
                         nullptr, ggml_get_op_params_f32(cgraph->nodes[s], 0), ggml_get_op_params_f32(cgraph->nodes[s + 2], 0), 1, nullptr };
    hc_persist_launch<true, true>(ctx, cgraph, s + 4, src, (float *) cgraph->nodes[s + 3]->data);
}

bool ggml_cuda_hc_persist_inject_other(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    // i: a MUL_MAT that may be an inject whose xn was produced in an earlier graph split (-sm tensor)
    if (!hc_persist_inject_on()) {
        return false;
    }
    const ggml_tensor * t = cgraph->nodes[i], * w = t->src[0], * xn = t->src[1];
    if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != K || w->ne[1] != HC || w->ne[2] != 1 ||
            xn->type != GGML_TYPE_F32 || xn->ne[0] != K || t->ne[0] != HC) {
        return false;
    }
    if (xn->data != ctx.hcp_last_xn || ctx.hcp_last_seq != ctx.hcp_seq) {
        return false; // not the xn of the last persistent kernel, or another one ran since
    }
    if (ctx.hcp_last_injw == w->data && ggml_cuda_hc_persist_post_match(ctx.device, cgraph, i + 1, i)) {
        return true;
    }
    // not produced: remember the pairing for that producer's next launch (graphs warm up live before capture)
    ctx.hcp_inj_map[ctx.hcp_last_down] = w->data;
    return false;
}

int ggml_cuda_hc_persist_done(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    for (const auto & d : ctx.hcp_done) {
        if (d.first == node) {
            return d.second;
        }
    }
    return 0;
}

// GDN gated norm at decode: RMS_NORM(o) -> MUL(w) , RESHAPE(z) -> SIGMOID -> MUL -> [RESHAPE ->] q8_0 MUL_MAT: one block per
// (head, token) writes o*rsqrt(mean(o^2)+eps)*w*sigmoid(z) and its q8_1 copy for the output projection (its quantize launch
// is skipped through the q8 cache)
static __global__ void __launch_bounds__(128) hc_gdn_gated_norm(const float * __restrict__ x, const float * __restrict__ w,
        const float * __restrict__ z, float * __restrict__ y, block_q8_1 * __restrict__ yq, const int H, const int64_t sx1,
        const int64_t sx2, const int64_t sz2, const float eps) {
    const int h = blockIdx.x % H, t = blockIdx.x / H, k = threadIdx.x;
    const float v = x[t*sx2 + h*sx1 + k];
    constexpr int NW = 128/WARP_SIZE;    // WARP_SIZE is the ggml logical warp (32), not the GCN wave
    __shared__ float red[NW];
    float ss = warp_reduce_sum(v*v);
    if (k % WARP_SIZE == 0) { red[k / WARP_SIZE] = ss; }
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int q = 0; q < NW; ++q) { ss += red[q]; }
    const float zz = z[t*sz2 + h*128 + k];
    const float o = (v*rsqrtf(ss/128.0f + eps))*w[k]*(1.0f/(1.0f + expf(-zz)));
    y[(int64_t) t*H*128 + h*128 + k] = o;
    if (yq) {
        const float amax = warp_reduce_max<QK8_1>(fabsf(o));
        const float sum  = warp_reduce_sum<QK8_1>(o);
        const float d = amax/127.0f;
        block_q8_1 & b = yq[((int64_t) t*H*128 + h*128 + k)/QK8_1];
        b.qs[k % QK8_1] = amax == 0.0f ? 0 : (int8_t) roundf(o/d);
        if (k % QK8_1 == 0) { b.ds = make_half2(d, sum); }
    }
}

int ggml_cuda_hc_gdn_gated_norm_match(const ggml_cgraph * cgraph, int i, int * mm_idx, int * z_mm) {
    // nodes: RMS_NORM, MUL(w), RESHAPE(z), SIGMOID, MUL  [, RESHAPE, MUL_MAT (q8 cache)]; returns the node count (0: no).
    // Or RMS_NORM, MUL(w), MUL_MAT(z), RESHAPE(z), ... with the gate's up projection between the norm and the gate
    // (GLM-5.3-Flash: ssm_g_b): it does not read the norm, so the caller computes it first (*z_mm = its index, else -1).
    static const bool disabled = [] { const char * v = getenv("GGML_CUDA_GDN_GNORM"); return v && atoi(v) == 0; }();
    static const bool zmm_ok   = [] { const char * v = getenv("GGML_CUDA_GDN_GNORM_ZMM"); return !v || atoi(v) != 0; }();
    *mm_idx = -1;
    *z_mm   = -1;
    if (disabled || i + 4 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * const * nd = cgraph->nodes + i;
    // o = 1: the z matmul at nd[2], everything after it one node later
    const int o = zmm_ok && i + 5 < cgraph->n_nodes && nd[2]->op == GGML_OP_MUL_MAT && nd[3]->op == GGML_OP_RESHAPE &&
        nd[3]->src[0] == nd[2] && nd[2]->src[0] != nd[0] && nd[2]->src[1] != nd[0] && nd[2]->src[0] != nd[1] &&
        nd[2]->src[1] != nd[1] && (nd[2]->flags & GGML_TENSOR_FLAG_COMPUTE) && !(nd[2]->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        ggml_node_get_use_count(cgraph, i + 2) == 1 ? 1 : 0;
    const ggml_tensor * rn = nd[0], * mw = nd[1], * zr = nd[2 + o], * sg = nd[3 + o], * mg = nd[4 + o];
    if (rn->op != GGML_OP_RMS_NORM || mw->op != GGML_OP_MUL || zr->op != GGML_OP_RESHAPE || sg->op != GGML_OP_UNARY ||
            ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID || mg->op != GGML_OP_MUL) {
        return 0;
    }
    const ggml_tensor * x = rn->src[0], * w = mw->src[0] == rn ? mw->src[1] : mw->src[0], * z = zr->src[0];
    const int64_t T = x->ne[2];
    const bool ok = x->type == GGML_TYPE_F32 && x->ne[0] == 128 && x->nb[0] == sizeof(float) && x->ne[3] == 1 && T >= 1 && T <= 8 &&
        (mw->src[0] == rn || mw->src[1] == rn) && w->type == GGML_TYPE_F32 && ggml_nelements(w) == 128 && ggml_is_contiguous(w) &&
        sg->src[0] == zr && z->type == GGML_TYPE_F32 && ggml_is_contiguous(z) && ggml_nelements(z) == ggml_nelements(x) &&
        ((mg->src[0] == mw && mg->src[1] == sg) || (mg->src[0] == sg && mg->src[1] == mw)) &&
        mg->type == GGML_TYPE_F32 && ggml_is_contiguous(mg) && ggml_are_same_shape(mg, x) &&
        // (not ggml_can_fuse_subgraph: the z reshape views a non-weight tensor from outside, which the kernel reads directly)
        ggml_node_get_use_count(cgraph, i) == 1 && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
        ggml_node_get_use_count(cgraph, i + 2 + o) == 1 && ggml_node_get_use_count(cgraph, i + 3 + o) == 1 &&
        !((rn->flags | mw->flags | zr->flags | sg->flags) & GGML_TENSOR_FLAG_OUTPUT) &&
        (rn->flags & mw->flags & sg->flags & mg->flags & GGML_TENSOR_FLAG_COMPUTE);
    if (!ok) {
        return 0;
    }
    // the output projection reads the q8_1 copy through the q8 cache: q8_0, or any quantized type when the cache serves
    // all types (GGML_CUDA_Q8_CACHE_ANYTYPE, the default; UD-Q2_K_XL: q5_K)
    static const bool any_type = [] { const char * e = getenv("GGML_CUDA_Q8_CACHE_ANYTYPE"); return !e || atoi(e) != 0; }();
    if (i + 6 + o < cgraph->n_nodes && nd[5 + o]->op == GGML_OP_RESHAPE && nd[5 + o]->src[0] == mg &&
            nd[6 + o]->op == GGML_OP_MUL_MAT && nd[6 + o]->src[1] == nd[5 + o] &&
            (nd[6 + o]->src[0]->type == GGML_TYPE_Q8_0 || (any_type && ggml_is_quantized(nd[6 + o]->src[0]->type))) &&
            ggml_is_contiguous(nd[5 + o]) && nd[5 + o]->ne[0] % 512 == 0 && nd[5 + o]->ne[2] == 1) {
        *mm_idx = i + 6 + o;
    }
    *z_mm = o ? i + 2 : -1;
    return 5 + o;
}

void ggml_cuda_hc_gdn_gated_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int mm_idx, int o) {
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * rn = nd[0], * mw = nd[1], * z = nd[2 + o]->src[0];
    const ggml_tensor * x = rn->src[0], * w = mw->src[0] == rn ? mw->src[1] : mw->src[0];
    float eps;
    memcpy(&eps, rn->op_params, sizeof(float));
    const int H = (int) x->ne[1], T = (int) x->ne[2];
    block_q8_1 * yq = nullptr;
    static const bool no_q8 = [] { const char * v = getenv("GGML_CUDA_GDN_GNORM_Q8"); return v && atoi(v) == 0; }();
    if (mm_idx >= 0 && !no_q8) {
        const ggml_tensor * y = cgraph->nodes[mm_idx]->src[1];
        const int64_t K = y->ne[0];
        const size_t nbytes = (size_t) y->ne[1]*K/QK8_1*sizeof(block_q8_1);
        ctx.q8_cache.push_back({ y, (int) GGML_TYPE_Q8_0, { K, y->ne[1], 1, 1 },
                                 std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes) });
        yq = (block_q8_1 *) ctx.q8_cache.back().buf->get();
    }
    hc_gdn_gated_norm<<<H*T, 128, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (const float *) z->data,
        (float *) nd[4 + o]->data, yq, H, x->nb[1]/sizeof(float), x->nb[2]/sizeof(float), (int64_t) H*128, eps);
    CUDA_CHECK(cudaGetLastError());
}

// attention output gate at decode: y = attn * sigmoid(gate), gate read through its strided view of the q projection, and
// the q8_1 copy of y for the output projection (its quantize launch is skipped through the q8 cache)
static __global__ void __launch_bounds__(256) hc_attn_gate(const float * __restrict__ attn, const float * __restrict__ g,
        float * __restrict__ y, block_q8_1 * __restrict__ yq, const int D, const int H, const int64_t g_s1, const int64_t g_s2) {
    const int nb = (D*H)/256, t = blockIdx.x / nb, c = (blockIdx.x % nb)*256 + threadIdx.x;
    const int h = c / D, d = c % D;
    const float gv = g[t*g_s2 + h*g_s1 + d];
    const float o = attn[(int64_t) t*D*H + c]*(1.0f/(1.0f + expf(-gv)));
    y[(int64_t) t*D*H + c] = o;
    if (yq) {
        const float amax = warp_reduce_max<QK8_1>(fabsf(o));
        const float sum  = warp_reduce_sum<QK8_1>(o);
        const float dd = amax/127.0f;
        block_q8_1 & b = yq[((int64_t) t*D*H + c)/QK8_1];
        b.qs[c % QK8_1] = amax == 0.0f ? 0 : (int8_t) roundf(o/dd);
        if (c % QK8_1 == 0) { b.ds = make_half2(dd, sum); }
    }
}

int ggml_cuda_hc_attn_gate_match(const ggml_cgraph * cgraph, int i, int * mm_idx) {
    static const bool disabled = [] { const char * v = getenv("GGML_CUDA_ATTN_GATE"); return v && atoi(v) == 0; }();
    *mm_idx = -1;
    if (disabled || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * ct = nd[0], * sg = nd[1], * ml = nd[2];
    if (ct->op != GGML_OP_CONT || sg->op != GGML_OP_UNARY || ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID ||
            ml->op != GGML_OP_MUL || sg->src[0] != ct) {
        return 0;
    }
    const ggml_tensor * gv = ct->src[0];
    const ggml_tensor * attn = ml->src[0] == sg ? ml->src[1] : ml->src[0];
    const int64_t D = gv->ne[0], H = gv->ne[1], T = gv->ne[2];
    const bool ok = (ml->src[0] == sg || ml->src[1] == sg) && gv->type == GGML_TYPE_F32 && gv->nb[0] == sizeof(float) &&
        gv->ne[3] == 1 && T >= 1 && T <= 8 && (D*H) % 256 == 0 && ct->type == GGML_TYPE_F32 && ggml_nelements(ct) == D*H*T &&
        attn->type == GGML_TYPE_F32 && ggml_is_contiguous(attn) && ggml_nelements(attn) == D*H*T &&
        ml->type == GGML_TYPE_F32 && ggml_is_contiguous(ml) && ggml_nelements(ml) == D*H*T &&
        ggml_node_get_use_count(cgraph, i) == 1 && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
        !((ct->flags | sg->flags) & GGML_TENSOR_FLAG_OUTPUT) && (ct->flags & sg->flags & ml->flags & GGML_TENSOR_FLAG_COMPUTE);
    if (!ok) {
        return 0;
    }
    // with the CONT skipped the gate's source can be freed early: the output must not have been placed over it
    const char * g0 = (const char *) gv->data, * y0 = (const char *) ml->data;
    if (g0 < y0 + ggml_nbytes(ml) && y0 < g0 + ggml_nbytes(gv)) {
        return 0;
    }
    if (i + 3 < cgraph->n_nodes && nd[3]->op == GGML_OP_MUL_MAT && nd[3]->src[1] == ml && nd[3]->src[0]->type == GGML_TYPE_Q8_0 &&
            ml->ne[0] % 512 == 0 && ml->ne[2] == 1) {
        *mm_idx = i + 3;
    }
    return 3;
}

void ggml_cuda_hc_attn_gate(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int mm_idx) {
    ggml_tensor * const * nd = cgraph->nodes + i;
    const ggml_tensor * gv = nd[0]->src[0], * sg = nd[1], * ml = nd[2];
    const ggml_tensor * attn = ml->src[0] == sg ? ml->src[1] : ml->src[0];
    const int D = (int) gv->ne[0], H = (int) gv->ne[1], T = (int) gv->ne[2];
    block_q8_1 * yq = nullptr;
    if (mm_idx >= 0) {
        const ggml_tensor * y = cgraph->nodes[mm_idx]->src[1];
        const int64_t K = y->ne[0];
        const size_t nbytes = (size_t) y->ne[1]*K/QK8_1*sizeof(block_q8_1);
        ctx.q8_cache.push_back({ y, (int) GGML_TYPE_Q8_0, { K, y->ne[1], 1, 1 },
                                 std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes) });
        yq = (block_q8_1 *) ctx.q8_cache.back().buf->get();
    }
    hc_attn_gate<<<T*(D*H/256), 256, 0, ctx.stream()>>>((const float *) attn->data, (const float *) gv->data, (float *) ml->data,
        yq, D, H, gv->nb[1]/sizeof(float), gv->nb[2]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

// sparse attention mask from the indexer's top-k (long contexts): CONT(top_k) -> FILL(mask, -inf) -> SET_ROWS(zeros at the
// selected cells) -> ADD(mask) is out[t][j] = mask[t][j] for the selected cells, -inf elsewhere (0 + x = x in f16, so
// bit-identical); one block per query row
static __global__ void hc_qsa_mask(const int32_t * __restrict__ top_k, const half * __restrict__ mask, half * __restrict__ out,
        const int n_kv, const int width, const int64_t tk_s1, const int64_t m_s1, const int64_t o_s1) {
    const int t = blockIdx.x;
    half * o = out + t*o_s1;
    const half * m = mask + t*m_s1;
    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) {
        o[j] = __float2half(-INFINITY);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
        const int j = top_k[t*tk_s1 + k];
        if (j >= 0 && j < n_kv) {
            o[j] = m[j];
        }
    }
}

static bool hc_is_view_op(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE;
}
static const ggml_tensor * hc_view_base(const ggml_tensor * t) {
    while (t && hc_is_view_op(t) && t->src[0]) { t = t->src[0]; }
    return t;
}

bool ggml_cuda_hc_is_view_op(const ggml_tensor * t) {
    return hc_is_view_op(t);
}

bool ggml_cuda_hc_qsa_chain_match(const ggml_cgraph * cgraph, int i, ggml_cuda_qsa_chain & c, const char ** why) {
    auto refuse = [why](const char * r) {
        if (why) {
            *why = r;
        }
        return false;
    };
    // the next compute nodes (views skipped): CONT(top_k) FILL SET_ROWS ADD
    int idx[4], n = 0;
    for (int j = i; j < cgraph->n_nodes && n < 4; ++j) {
        if (!hc_is_view_op(cgraph->nodes[j])) {
            idx[n++] = j;
        }
    }
    if (n < 4) {
        return refuse("chain: fewer than 4 compute nodes follow");
    }
    ggml_tensor * const * nd = cgraph->nodes;
    const ggml_tensor * ct = nd[idx[0]], * fl = nd[idx[1]], * sr = nd[idx[2]];
    ggml_tensor * ad = nd[idx[3]];
    if (ct->op != GGML_OP_CONT || fl->op != GGML_OP_FILL || sr->op != GGML_OP_SET_ROWS || ad->op != GGML_OP_ADD ||
            ct->src[0]->op != GGML_OP_TOP_K) {
        return refuse("chain: not CONT(TOP_K) FILL SET_ROWS ADD");
    }
    const ggml_tensor * mask = fl->src[0];
    const ggml_tensor * ad_other = hc_view_base(ad->src[0]) == sr ? ad->src[1] : hc_view_base(ad->src[1]) == sr ? ad->src[0] : nullptr;
    const float fill_v = ggml_get_op_params_f32(fl, 0);
    const int64_t n_kv = mask->ne[0], T = mask->ne[1];
    const bool ok = ad_other == mask && ct->type == GGML_TYPE_I32 && ggml_is_contiguous(ct) && fill_v == -INFINITY &&
        mask->type == GGML_TYPE_F16 && fl->type == GGML_TYPE_F16 && ad->type == GGML_TYPE_F16 && ggml_is_contiguous(ad) &&
        mask->nb[0] == sizeof(half) && mask->ne[2] == 1 && mask->ne[3] == 1 && ad->ne[0] == n_kv && ad->ne[1] == T &&
        sr->view_src != nullptr && hc_view_base(sr->view_src) == fl && hc_view_base(sr->src[1]) == ct && sr->src[0]->type == GGML_TYPE_F32 &&
        sr->src[0]->op == GGML_OP_FILL && ggml_get_op_params_f32(sr->src[0], 0) == 0.0f &&
        ct->ne[1] == T && ct->ne[2] == 1 && ct->ne[3] == 1 &&
        ggml_node_get_use_count(cgraph, idx[0]) == 1 && ggml_node_get_use_count(cgraph, idx[2]) == 1 &&
        !((ct->flags | fl->flags | sr->flags) & GGML_TENSOR_FLAG_OUTPUT);
    if (!ok) {
        return refuse("chain: types/shapes/uses/outputs");
    }
    // the filled mask is read only through the SET_ROWS; the skipped CONT and SET_ROWS results have no reader through views
    // either, other than the SET_ROWS and the ADD (use counts only see direct readers: a view of the CONT read by another
    // layer's attention, IndexShare-style, would read a CONT that is never computed)
    for (int u = idx[0] + 1; u < cgraph->n_nodes; ++u) {
        const ggml_tensor * t = nd[u];
        if (hc_is_view_op(t)) {
            continue;
        }
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            if (!t->src[q]) {
                continue;
            }
            const ggml_tensor * b = hc_view_base(t->src[q]);
            if ((u > idx[1] && t != sr && t != ad && b == fl) || (t != sr && b == ct) || (t != ad && b == sr)) {
                return refuse("chain: a skipped CONT/FILL/SET_ROWS result has another reader");
            }
        }
    }
    // the CONT is skipped too: read the TOP_K result itself
    const ggml_tensor * tk = ct->src[0];
    if (tk->type != GGML_TYPE_I32 || tk->nb[0] != sizeof(int32_t) || tk->ne[0] != ct->ne[0] || tk->ne[1] != T || ggml_nrows(tk) != T) {
        return refuse("chain: TOP_K layout");
    }
    for (int k = 0; k < 4; ++k) {
        c.idx[k] = idx[k];
    }
    c.tk    = tk;
    c.mask  = mask;
    c.ad    = ad;
    c.width = (int) ct->ne[0];
    return true;
}

int ggml_cuda_hc_qsa_mask(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = [] { const char * v = getenv("GGML_CUDA_QSA_MASK_FUSE"); return v && atoi(v) == 0; }();
    if (disabled) {
        return 0;
    }
    ggml_cuda_qsa_chain c;
    if (!ggml_cuda_hc_qsa_chain_match(cgraph, i, c)) {
        return 0;
    }
    const ggml_tensor * tk = c.tk, * mask = c.mask, * ad = c.ad;
    const int64_t n_kv = mask->ne[0], T = mask->ne[1];
    const int width = c.width;
    // the allocator may place the FILL/ADD result over the TOP_K result (its only consumer, the CONT, is skipped): then
    // stage the indices in pool memory first
    const int32_t * tkd = (const int32_t *) tk->data;
    int64_t tk_s1 = tk->nb[1]/sizeof(int32_t);
    ggml_cuda_pool_alloc<int32_t> tk_copy(ctx.pool());
    const char * tk_lo = (const char *) tk->data, * tk_hi = tk_lo + (T - 1)*tk->nb[1] + width*sizeof(int32_t);
    const char * ad_lo = (const char *) ad->data, * ad_hi = ad_lo + ggml_nbytes(ad);
    // the kernel fills a row with -inf before reading the selected mask cells: the result may not lie over the mask
    // (the allocator reuses the KQ mask's memory at its last consumer, the last attention layer: fully masked rows
    // there made every MTP draft fail once the context exceeded the top-k width)
    const char * m_lo = (const char *) mask->data, * m_hi = m_lo + ggml_nbytes(mask);
    if (m_lo < ad_hi && ad_lo < m_hi) {
        return 0;
    }
    if (tk_lo < ad_hi && ad_lo < tk_hi) {
        tk_copy.alloc(T*width);
        CUDA_CHECK(cudaMemcpy2DAsync(tk_copy.get(), width*sizeof(int32_t), tk->data, tk->nb[1], width*sizeof(int32_t), T,
            cudaMemcpyDeviceToDevice, ctx.stream()));
        tkd = tk_copy.get();
        tk_s1 = width;
    }
    hc_qsa_mask<<<(unsigned) T, 256, 0, ctx.stream()>>>(tkd, (const half *) mask->data, (half *) ad->data,
        (int) n_kv, width, tk_s1, mask->nb[1]/sizeof(half), ad->nb[1]/sizeof(half));
    CUDA_CHECK(cudaGetLastError());
    return c.idx[3] - i;
}

// PLE conv tail: CONT(permuted conv input) -> K x (CONT(weight column) MUL [ADD]) -> SILU -> ADD(gated) -> ADD(hidden) as one
// kernel, one thread per (channel, token). Multiplies and adds stay separate roundings in the graph's order, but the SiLU
// comes out 1 ulp off the unary kernel's on ~1/4 of the values (KLD unchanged), and pp4 gained only ~0.07 ms of 20.8:
// opt-in (GGML_CUDA_PLE_CONV_FUSE=1).
#define HC_PLE_MAXK 8
struct hc_ple_args {
    const char * x;                  // the permuted conv input: element (c, p) at c*x_nb0 + p*x_nb1
    int64_t x_nb0, x_nb1;
    const char * w[HC_PLE_MAXK];     // weight column k: element c at c*w_nb[k]
    int64_t w_nb[HC_PLE_MAXK];
    int start[HC_PLE_MAXK];          // tap k reads position start[k] + t
    const float * gated, * hidden;
    float * out;
    int C, T, K;
};

static __global__ void hc_ple_conv(const hc_ple_args a) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= a.C*a.T) {
        return;
    }
    const int c = i % a.C, t = i / a.C;
    const char * xc = a.x + c*a.x_nb0;
    float acc = 0.0f;
    for (int k = 0; k < a.K; ++k) {
        const float term = __fmul_rn(*(const float *) (xc + (a.start[k] + t)*a.x_nb1), *(const float *) (a.w[k] + c*a.w_nb[k]));
        acc = k == 0 ? term : __fadd_rn(acc, term);
    }
    const float sv = ggml_cuda_op_silu_single(acc);
    a.out[i] = __fadd_rn(a.hidden[i], __fadd_rn(a.gated[i], sv));
}

int ggml_cuda_hc_ple_conv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = [] { const char * v = getenv("GGML_CUDA_PLE_CONV_FUSE"); return v && atoi(v) != 0; }();
    if (!enabled) {
        return 0;
    }
    ggml_tensor * const * nd = cgraph->nodes;
    const ggml_tensor * pt = nd[i];
    const ggml_tensor * perm = pt->src[0];
    if (pt->op != GGML_OP_CONT || perm->op != GGML_OP_PERMUTE || pt->type != GGML_TYPE_F32 || perm->type != GGML_TYPE_F32 ||
            pt->ne[2] != 1 || pt->ne[3] != 1 || ggml_node_get_use_count(cgraph, i) < 1) {
        return 0;
    }
    const int64_t C = pt->ne[0];
    hc_ple_args a = {};
    a.x = (const char *) perm->data; a.x_nb0 = perm->nb[0]; a.x_nb1 = perm->nb[1];
    a.C = (int) C;
    // walk the compute nodes after the CONT (views skipped)
    std::vector<int> used;                 // every fused node but the last must be used only inside
    const ggml_tensor * acc = nullptr;
    int j = i + 1, K = 0;
    const auto next = [&]() -> const ggml_tensor * {
        while (j < cgraph->n_nodes && hc_is_view_op(nd[j])) { ++j; }
        return j < cgraph->n_nodes ? nd[j++] : nullptr;
    };
    const ggml_tensor * t = next();
    while (t && t->op == GGML_OP_CONT) {
        // tap K: CONT(weight column view) MUL(shifted view of pt, weight) [ADD(acc, term)]
        const ggml_tensor * cw = t, * wv = cw->src[0];
        const int icw = j - 1;
        const ggml_tensor * m = next();
        const int im = j - 1;
        if (K >= HC_PLE_MAXK || cw->type != GGML_TYPE_F32 || wv->type != GGML_TYPE_F32 || ggml_nelements(cw) != C || wv->ne[0] != 1 ||
                !m || m->op != GGML_OP_MUL || hc_view_base(m->src[1]) != cw || m->src[0]->op != GGML_OP_VIEW ||
                m->src[0]->view_src != pt || m->src[0]->nb[1] != pt->nb[1] || m->src[0]->ne[0] != C || m->src[0]->ne[2] != 1 ||
                m->src[0]->view_offs % pt->nb[1] != 0 || m->type != GGML_TYPE_F32 || !ggml_is_contiguous(m)) {
            return 0;
        }
        a.w[K] = (const char *) wv->data; a.w_nb[K] = wv->nb[1];
        a.start[K] = (int) (m->src[0]->view_offs / pt->nb[1]);
        if (K == 0) {
            a.T = (int) m->ne[1];
            acc = m;
        } else {
            const ggml_tensor * ad = next();
            if (!ad || ad->op != GGML_OP_ADD || ad->src[0] != acc || ad->src[1] != m || !ggml_is_contiguous(ad)) {
                return 0;
            }
            used.push_back(j - 1);
            acc = ad;
        }
        used.push_back(icw); used.push_back(im);
        ++K;
        t = next();
    }
    const int T = a.T;
    if (K < 1 || !t || t->op != GGML_OP_UNARY || ggml_get_unary_op(t) != GGML_UNARY_OP_SILU || t->src[0] != acc || T < 1 ||
            a.start[K - 1] + T > pt->ne[1]) {
        return 0;
    }
    const ggml_tensor * sl = t;
    used.push_back(j - 1);
    const ggml_tensor * in = next();
    const int iin = j - 1;
    const ggml_tensor * outr = next();
    if (!in || !outr || in->op != GGML_OP_ADD || outr->op != GGML_OP_ADD) {
        return 0;
    }
    const ggml_tensor * gated = hc_view_base(in->src[1]) == sl ? in->src[0] : hc_view_base(in->src[0]) == sl ? in->src[1] : nullptr;
    const ggml_tensor * hidden = outr->src[0] == in ? outr->src[1] : outr->src[1] == in ? outr->src[0] : nullptr;
    used.push_back(iin);
    const int64_t n = C*T;
    if (!gated || !hidden || gated->type != GGML_TYPE_F32 || hidden->type != GGML_TYPE_F32 || outr->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(gated) || !ggml_is_contiguous(hidden) || !ggml_is_contiguous(outr) || !ggml_is_contiguous(in) ||
            ggml_nelements(gated) != n || ggml_nelements(hidden) != n || ggml_nelements(outr) != n) {
        return 0;
    }
    for (int u : used) {
        if (ggml_node_get_use_count(cgraph, u) != 1 || (nd[u]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }
    if (ggml_node_get_use_count(cgraph, i) != K || (pt->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const auto overlap = [](const void * p0, size_t n0, const void * p1, size_t n1) {
        const char * a0 = (const char *) p0, * b0 = (const char *) p1;
        return a0 < b0 + n1 && b0 < a0 + n0;
    };
    // an in-place residual add is element-wise and safe; any other overlap with the inputs is not
    const size_t nb_out = ggml_nbytes(outr);
    if ((outr->data != gated->data && overlap(outr->data, nb_out, gated->data, ggml_nbytes(gated))) ||
            (outr->data != hidden->data && overlap(outr->data, nb_out, hidden->data, ggml_nbytes(hidden)))) {
        return 0;
    }
    // the conv input dies at the skipped CONT, so the output may sit on it: stage it then
    ggml_cuda_pool_alloc<char> x_copy(ctx.pool());
    const size_t nb_x = ggml_nbytes(perm);
    if (overlap(outr->data, nb_out, perm->data, nb_x)) {
        x_copy.alloc(nb_x);
        CUDA_CHECK(cudaMemcpyAsync(x_copy.get(), perm->data, nb_x, cudaMemcpyDeviceToDevice, ctx.stream()));
        a.x = x_copy.get();
    }
    a.gated = (const float *) gated->data; a.hidden = (const float *) hidden->data; a.out = (float *) outr->data; a.K = K;
    hc_ple_conv<<<(unsigned) ((n + 255)/256), 256, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    return (j - 1) - i;
}

// ---- fused tensor-parallel AllReduce: host side ----------------------------------------------------------------------
struct hcp_ar_dev {
    float * inbox = nullptr;             // 2 halves of ns*TMAX*E floats ([half][slot][TMAX*E]), uncached
    int   * flag  = nullptr;             // ns uncached ints (one per peer slot)
    int     peer  = -1;                  // a peer (set once the group is initialized)
    int     n     = 0;                   // group size (2 or 4)
    int     rank  = -1;
    int     devs[ARNS + 1] = { -1, -1, -1, -1 }; // the group's devices by rank
    const ggml_tensor * deferred = nullptr; // this call's deferred AllReduce tensor
};
static hcp_ar_dev g_hcp_ar[GGML_CUDA_MAX_DEVICES];
// the persistent kernel's 4-GPU exchange code (sum of 3 peers, push to 3) is a separate instantiation: the pair path
// keeps the register budget it had (the runtime-generic version cost ~3% of 2+2 decode)
static bool hcp_ar_group4(int device) { return g_hcp_ar[device].n == 4; }

static bool hcp_ar_enabled() {
    static const bool en = [] { const char * e = getenv("GGML_CUDA_HCP_AR"); return e && atoi(e) != 0; }();
    return en && ggml_cuda_hc_persist_mode() >= 2;
}

static __global__ void hcp_ar_push(const hc_push push, const float * data, int n);

// the peers' inbox slots and flags for device rank r's pushes
static hc_push hcp_ar_push_of(const hcp_ar_dev & d) {
    hc_push p = {};
    const int ns = d.n - 1;
    p.hs = ns*TMAX*E;
    for (int q = 0; q < d.n; ++q) {
        if (q == d.rank) {
            continue;
        }
        const hcp_ar_dev & pd = g_hcp_ar[d.devs[q]];
        const int slot = d.rank < q ? d.rank : d.rank - 1; // this GPU's slot on rank q
        p.inbox[p.n] = pd.inbox + (size_t) slot*TMAX*E;
        p.flag[p.n]  = pd.flag + slot;
        p.n++;
    }
    return p;
}

bool ggml_cuda_hcp_ar_defer(const int * dev_ids, int n, ggml_tensor ** tensors, void * const * streams) {
    if (!hcp_ar_enabled() || (n != 2 && n != 4)) {
        return false;
    }
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = tensors[j];
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != E || ggml_nrows(t) < 1 || ggml_nrows(t) > TMAX ||
                ggml_cuda_info().devices[dev_ids[j]].physical_share_count != 1) {
            return false;
        }
    }
    for (int j = 0; j < n; ++j) {
        hcp_ar_dev & d = g_hcp_ar[dev_ids[j]];
        if (d.inbox != nullptr && (d.n != n || d.rank != j)) {
            return false; // a device in two different groups: not supported
        }
        if (d.inbox == nullptr) {
            ggml_cuda_set_device(dev_ids[j]);
            for (int q = 0; q < n; ++q) {
                if (q == j) {
                    continue;
                }
                const cudaError_t rc = cudaDeviceEnablePeerAccess(ggml_cuda_info().devices[dev_ids[q]].physical_device, 0);
                if (rc != cudaSuccess) { (void) cudaGetLastError(); } // already enabled
            }
            if (hipExtMallocWithFlags((void **) &d.inbox, 2*(size_t) (n - 1)*TMAX*E*sizeof(float), hipDeviceMallocUncached) != hipSuccess ||
                    hipExtMallocWithFlags((void **) &d.flag, 64, hipDeviceMallocUncached) != hipSuccess ||
                    cudaMemset(d.flag, 0, 64) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                (void) cudaGetLastError();
                d.inbox = nullptr;
                return false;
            }
            d.n = n;
            d.rank = j;
            for (int q = 0; q < n; ++q) { d.devs[q] = dev_ids[q]; }
            d.peer = dev_ids[j == 0 ? 1 : 0];
        }
    }
    for (int j = 0; j < n; ++j) {
        if (g_hcp_ar[dev_ids[j]].inbox == nullptr) {
            return false;
        }
    }
    // a device whose slice of this sublayer is empty (e.g. 4 GPUs over 2 KV heads) did not compute its node: the tensor
    // holds stale data, and the reduction counts it as zero (as the meta backend's own AllReduce does)
    for (int j = 0; j < n; ++j) {
        if (!(tensors[j]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            ggml_cuda_set_device(dev_ids[j]);
            CUDA_CHECK(cudaMemsetAsync(tensors[j]->data, 0, ggml_nbytes(tensors[j]), (cudaStream_t) streams[j]));
        }
    }
    for (int j = 0; j < n; ++j) {
        ggml_cuda_set_device(dev_ids[j]);
        hcp_ar_push<<<1, 1024, 0, (cudaStream_t) streams[j]>>>(hcp_ar_push_of(g_hcp_ar[dev_ids[j]]),
            (const float *) tensors[j]->data, (int) ggml_nelements(tensors[j]));
        CUDA_CHECK(cudaGetLastError());
        g_hcp_ar[dev_ids[j]].deferred = tensors[j];
    }
    return true;
}

// the peers' inbox slots and flags for a producer kernel's own push (none until the exchange buffers exist)
static void hcp_ar_push_target(int device, hc_push * push) {
    static const bool off = [] { const char * e = getenv("GGML_CUDA_HCP_AR_PUSH"); return e && atoi(e) == 0; }();
    *push = {};
    const hcp_ar_dev & d = g_hcp_ar[device];
    if (off || !hcp_ar_enabled() || d.peer < 0 || d.inbox == nullptr) {
        return;
    }
    for (int q = 0; q < d.n; ++q) {
        if (g_hcp_ar[d.devs[q]].inbox == nullptr) {
            return;
        }
    }
    *push = hcp_ar_push_of(d);
}

static hc_ar hc_persist_take_ar(ggml_backend_cuda_context & ctx, const ggml_tensor * out) {
    hcp_ar_dev & d = g_hcp_ar[ctx.device];
    if (ctx.hcp_ar_absorb == nullptr || (ctx.hcp_ar_absorb != out && ctx.hcp_ar_absorb != out->view_src) || d.peer < 0) {
        return hc_ar{};
    }
    ctx.hcp_ar_absorb = nullptr;
    const hcp_ar_dev & pd = g_hcp_ar[d.peer];
    static const bool dbg = getenv("GGML_CUDA_HC_PERSIST_DEBUG") != nullptr;
    static int nd = 0;
    if (dbg && nd < 40) { nd++; fprintf(stderr, "hcp ar absorbed dev %d '%s'\n", ctx.device, out->name); }
    (void) pd;
    return hc_ar{ d.inbox, d.flag, d.n - 1, d.rank };
}

// push half, launched on each GPU's stream at AllReduce time: this GPU's partial into the peer's inbox, then the peer's flag
static __global__ void hcp_ar_push(const hc_push push, const float * data, int n) {
    unsigned int s0, sp;
    __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(sp) : "s"(&g_hcp_pushed) : "memory");
    if (sp) {
        // the producer kernel right before this one pushed the exchange itself (and advanced the sequence)
        __syncthreads();
        if (threadIdx.x == 0) { g_hcp_pushed = 0u; }
        return;
    }
    __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(s0) : "s"(&g_hcp_arseq) : "memory");
    const unsigned int s = s0 + 1;
    for (int i = threadIdx.x; i < n/4; i += blockDim.x) {
        const float4 v = ((const float4 *) data)[i];
#pragma unroll
        for (int k = 0; k < ARNS; ++k) {
            if (k < push.n) { ((float4 *) (push.inbox[k] + (size_t) (s & 1)*push.hs))[i] = v; }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 0; k < push.n; ++k) { *(volatile int *) push.flag[k] = (int) s; }
        __threadfence_system();
        atomicAdd(&g_hcp_arseq, 1u);
    }
}

// the wait + in-place add, when no persistent kernel consumes the deferred tensor
static __global__ void __launch_bounds__(1024) hcp_ar_fallback(const hc_ar ar, float * data, int n) {
    unsigned int s;
    __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(s) : "s"(&g_hcp_arseq) : "memory");
    hcp_ar_wait(ar, s);
    const float * in = ar.inbox_mine + (s & 1)*ar.ns*TMAX*E;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        data[i] = ar.ns == 1 ? hcp_ar_sum<false>(data[i], in, ar.ns, ar.rank, i) : hcp_ar_sum<true>(data[i], in, ar.ns, ar.rank, i);
    }
}

void ggml_cuda_hcp_ar_graph_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    hcp_ar_dev & d = g_hcp_ar[ctx.device];
    ctx.hcp_ar_absorb = nullptr;
    const ggml_tensor * t = d.deferred;
    if (t == nullptr) {
        return;
    }
    d.deferred = nullptr;
    // absorbed iff its only reader in this graph is the combine of a persistent HC kernel (graph structure only, so a
    // replayed HIP graph made the same choice when it was captured)
    int readers = 0, post = -1;
    for (int k = 0; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * u = cgraph->nodes[k];
        if (hc_persist_is_view_or_noop(u)) {
            continue; // views do not read the data
        }
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            if (u->src[q] && (u->src[q] == t || u->src[q]->view_src == t)) {
                readers++;
                if (u->op == GGML_OP_DSV4_HC_POST && q == 0) { post = k; }
            }
        }
    }
    // GGML_CUDA_HCP_AR_ABSORB=0 (diagnostic): always exchange through the fallback kernel
    static const bool absorb_off = [] { const char * v = getenv("GGML_CUDA_HCP_AR_ABSORB"); return v && atoi(v) == 0; }();
    const bool absorb = !absorb_off && readers == 1 && post >= 3 && ggml_cuda_hc_persist_post_match(ctx.device, cgraph, post - 3, -1);
    static const bool dbg = getenv("GGML_CUDA_HC_PERSIST_DEBUG") != nullptr;
    static int nd = 0;
    if (dbg && nd < 40) { nd++; fprintf(stderr, "hcp ar dev %d '%s' readers %d post %d absorb %d\n", ctx.device, t->name, readers, post, (int) absorb); }
    if (absorb) {
        ctx.hcp_ar_absorb = t;
        return;
    }
    hcp_ar_fallback<<<1, 1024, 0, ctx.stream()>>>(hc_ar{ d.inbox, d.flag, d.n - 1, d.rank }, (float *) t->data, (int) ggml_nelements(t));
    CUDA_CHECK(cudaGetLastError());
}

