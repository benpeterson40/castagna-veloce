#pragma once
// In-kernel phase timing for latency-chain analysis (debug, compiled in only with -DGGML_CUDA_KTRACE).
// A traced kernel stamps s_memrealtime (25 MHz on gfx906, i.e. 40 ns ticks) at its phase boundaries in workgroup 0
// (or in whichever workgroup calls KT_COMMIT) and accumulates the per-phase deltas on the device (sum/count/min/max,
// graph-safe). The launching TU calls KT_HOST_TICK(name) after each launch; every GGML_CUDA_KTRACE launches (env, 0 =
// off) it synchronizes, prints the per-phase statistics in microseconds and resets them. Device accumulators are
// per translation unit (no relocatable device code), so each TU that traces includes this header.
//
//   kernel:  KT_DECL; KT_STAMP(0); ...phase 1...; KT_STAMP(1); ...; KT_COMMIT(kid, n_stamps);
//   host:    KT_HOST_TICK(kid, "name", n_stamps, stream);

#ifdef GGML_CUDA_KTRACE

#include <cstdio>
#include <cstdlib>

#define KT_NK 16
#define KT_NP 10

struct kt_rec {
    unsigned long long sum[KT_NP];
    unsigned int       cnt[KT_NP];
    unsigned int       mn[KT_NP];
    unsigned int       mx[KT_NP];
};

static __device__ kt_rec g_kt[KT_NK];

#define KT_DECL unsigned long long kt_ts[KT_NP]
#define KT_STAMP(i) do { kt_ts[(i)] = __builtin_amdgcn_s_memrealtime(); } while (0)
// workgroup-0 thread 0 commits (call after the last stamp; the caller decides which workgroup/thread stamps)
#define KT_COMMIT(kid, n) do {                                                             \
        for (int kt_i = 1; kt_i < (n); ++kt_i) {                                           \
            const unsigned int kt_d = (unsigned int) (kt_ts[kt_i] - kt_ts[kt_i - 1]);      \
            atomicAdd(&g_kt[(kid)].sum[kt_i], (unsigned long long) kt_d);                  \
            atomicAdd(&g_kt[(kid)].cnt[kt_i], 1u);                                         \
            atomicMin(&g_kt[(kid)].mn[kt_i], kt_d);                                        \
            atomicMax(&g_kt[(kid)].mx[kt_i], kt_d);                                        \
        }                                                                                  \
    } while (0)

static int kt_every() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_KTRACE"); return e ? atoi(e) : 0; }();
    return v;
}

static void kt_reset_dev() {
    static kt_rec z[KT_NK];
    for (int k = 0; k < KT_NK; ++k) {
        for (int p = 0; p < KT_NP; ++p) {
            z[k].sum[p] = 0; z[k].cnt[p] = 0; z[k].mn[p] = 0xFFFFFFFFu; z[k].mx[p] = 0;
        }
    }
    (void) hipMemcpyToSymbol(HIP_SYMBOL(g_kt), z, sizeof(z));
}

static void kt_host_tick(const int kid, const char * name, const int n, cudaStream_t stream) {
    const int every = kt_every();
    if (every <= 0) {
        return;
    }
    int dev = 0;
    (void) hipGetDevice(&dev);
    static int  count[16][KT_NK] = {{0}};
    static bool init[16] = {false};
    if (dev >= 16) {
        return;
    }
    if (!init[dev]) {
        init[dev] = true;
        (void) cudaStreamSynchronize(stream);
        kt_reset_dev();
    }
    if (++count[dev][kid] % every != 0) {
        return;
    }
    (void) cudaStreamSynchronize(stream);
    static kt_rec h[KT_NK];
    (void) hipMemcpyFromSymbol(h, HIP_SYMBOL(g_kt), sizeof(h));
    fprintf(stderr, "ktrace dev%d %-22s", dev, name);
    for (int p = 1; p < n; ++p) {
        if (h[kid].cnt[p]) {
            fprintf(stderr, " | p%d %.2f [%.2f %.2f]", p, 0.04*h[kid].sum[p]/h[kid].cnt[p], 0.04*h[kid].mn[p], 0.04*h[kid].mx[p]);
        }
    }
    fprintf(stderr, " us (n=%u)\n", h[kid].cnt[1]);
    // reset this kernel's record only
    kt_rec z;
    for (int p = 0; p < KT_NP; ++p) {
        z.sum[p] = 0; z.cnt[p] = 0; z.mn[p] = 0xFFFFFFFFu; z.mx[p] = 0;
    }
    (void) hipMemcpyToSymbol(HIP_SYMBOL(g_kt), &z, sizeof(z), kid*sizeof(kt_rec));
}

#define KT_HOST_TICK(kid, name, n, stream) kt_host_tick((kid), (name), (n), (stream))

#else

#define KT_DECL
#define KT_STAMP(i) do { } while (0)
#define KT_COMMIT(kid, n) do { } while (0)
#define KT_HOST_TICK(kid, name, n, stream) do { } while (0)

#endif
