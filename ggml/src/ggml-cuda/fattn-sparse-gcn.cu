#include "fattn-sparse-gcn.cuh"
#include <vector>
#include <algorithm>
#include "fattn-tile.cuh"

thread_local const int * ggml_cuda_fattn_kv_max_override = nullptr;

// Sparse decode attention for GCN (no matrix cores), up to 8 query tokens per sequence (decode, MTP verify), F16 K/V.
//
// A sparse-attention indexer (qwen4exp) leaves ~2K of the context visible in the mask, but the attention kernels still
// stream all n_kv K/V rows: on gfx906 the tile kernel (grouped query heads in power-of-two groups, so 3 passes for 12
// query heads per KV head) reads ~200 MB per layer at 32K context for ~2K rows that count.
//
//   1. compact: the columns visible to any query row -> an index list in column order (64-lane ballots into a bitmap,
//               the last block expands it; an atomic slot per wave made the order, and so the tile kernel's
//               accumulation order and the decode logits, change from run to run)
//   2. gather : those K/V rows and every query row's mask values into a dense buffer of
//               n_pad = GGML_PAD(min(n_q*n_kv_max, n_kv), 256) rows (padding rows repeat column 0 under a -inf mask)
//   3. attend : the tile kernel on the gathered K/V
// Used when the graph gives the selection width (the flash attention op's n_kv_max hint) and the gathered rows are at
// most half of the KV length. gfx906, 2 x 12 heads of 256, 33K cells, 2051 selected: decode 222 -> 72 us, 4-token MTP
// verify 453 -> 210 us; model decode at 32K context 33.2 -> 35.6 t/s. A hand-written attend kernel (lane per key / 4 lanes per key, partials merged in LDS) was tried first
// and was slower than the tile kernel on the gathered rows.

// grid (n_kv/256 blocks, n_seq): each wave stores the ballot of its 64 columns (finite mask value for any query row) in
// bits[]; the last block of the sequence (ticket in count[n_seq + seq], zeroed by the host) writes the index list in
// column order and count[seq]. Wave-level scans of the word popcounts, one block barrier per 256 words.
static __global__ void __launch_bounds__(256) spg_compact(
        const half * __restrict__ mask, int32_t * __restrict__ idx, int32_t * __restrict__ count,
        unsigned long long * __restrict__ bits, const int n_kv, const int cap, const int n_q, const int64_t s_mask_row,
        const int64_t s_mask_seq) {
    const int seq    = blockIdx.y;
    const int lane   = threadIdx.x % 64;
    const int wave   = threadIdx.x / 64;
    const int i      = blockIdx.x*256 + threadIdx.x;
    const int nwords = gridDim.x*4;
    unsigned long long * sbits = bits + (int64_t) seq*nwords;
    bool sel = false; // visible to any of the n_q query rows
    for (int q = 0; q < n_q && i < n_kv && !sel; ++q) {
        sel = __half2float(mask[seq*s_mask_seq + q*s_mask_row + i]) > -INFINITY;
    }
    const uint64_t b = __ballot(sel);
    if (lane == 0) {
        sbits[blockIdx.x*4 + wave] = b;
    }
    __shared__ int s_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        s_last = atomicAdd(&count[gridDim.y + seq], 1) == (int) gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();
    __shared__ int s_wsum[4];
    int base = 0; // block-uniform: set bits in the words before this round
    for (int w0 = 0; w0 < nwords; w0 += 256) {
        const int w = w0 + threadIdx.x;
        const uint64_t word = w < nwords ? __builtin_nontemporal_load(&sbits[w]) : 0ull;
        const int c = __popcll(word);
        int incl = c; // inclusive scan within the wave
#pragma unroll
        for (int off = 1; off < 64; off *= 2) {
            const int v = __shfl_up(incl, off, 64);
            incl += lane >= off ? v : 0;
        }
        if (lane == 63) {
            s_wsum[wave] = incl;
        }
        __syncthreads();
        int pos = base + incl - c;
        int total = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            pos   += j < wave ? s_wsum[j] : 0;
            total += s_wsum[j];
        }
        for (uint64_t m = word; m; m &= m - 1) {
            if (pos < cap) {
                idx[(int64_t) seq*cap + pos] = w*64 + __builtin_ctzll(m);
            }
            ++pos;
        }
        base += total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        count[seq] = base;
    }
}

// grid (n_pad, n_seq), 64 threads: row p of the gathered K, V (all KV heads of the cell) and mask
static __global__ void __launch_bounds__(64) spg_gather(
        const char * __restrict__ K, const char * __restrict__ V, const half * __restrict__ mask,
        const int32_t * __restrict__ idx, const int32_t * __restrict__ count,
        char * __restrict__ Kg, char * __restrict__ Vg, half * __restrict__ mg,
        const int cap, const int n_pad, const int n_kvh, const int row_bytes, // bytes of one head's row
        const int64_t k_nb1, const int64_t k_nb2, const int64_t k_nb3,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t v_nb3, const int n_q, const int64_t s_mask_row,
        const int64_t s_mask_seq, int * __restrict__ kv_max) {
    const int p   = blockIdx.x;
    const int seq = blockIdx.y;
    const int n   = min(count[seq], cap);
    // with kv_max the tile kernel stops at the padded union: rows past it are never read (without it, write them all)
    const int n_used = kv_max ? min(n_pad, ((n + 255)/256)*256) : n_pad;
    if (p >= n_used && p > 0) {
        return;
    }
    if (p == 0 && threadIdx.x < 8 && kv_max) {
        kv_max[seq*8 + threadIdx.x] = max(n_used, 256);
    }
    const bool valid = p < n;
    const int col = valid ? idx[(int64_t) seq*cap + p] : 0;
    if (threadIdx.x < n_q) { // each query row keeps its own mask value
        mg[((int64_t) seq*n_q + threadIdx.x)*n_pad + p] =
            valid ? mask[seq*s_mask_seq + threadIdx.x*s_mask_row + col] : __float2half(-INFINITY);
    }
    const int words = row_bytes / 4;
    for (int h = 0; h < n_kvh; ++h) {
        const int * ks = (const int *) (K + (int64_t) col*k_nb1 + (int64_t) h*k_nb2 + (int64_t) seq*k_nb3);
        const int * vs = (const int *) (V + (int64_t) col*v_nb1 + (int64_t) h*v_nb2 + (int64_t) seq*v_nb3);
        int * kd = (int *) (Kg + (((int64_t) seq*n_pad + p)*n_kvh + h)*row_bytes);
        int * vd = (int *) (Vg + (((int64_t) seq*n_pad + p)*n_kvh + h)*row_bytes);
        for (int w = threadIdx.x; w < words; w += 64) {
            kd[w] = ks[w];
            vd[w] = vs[w];
        }
    }
}

static int spg_env() {
    static const int env = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_GCN"); return e ? atoi(e) : 1; }();
    return env;
}

bool ggml_cuda_flash_attn_sparse_gcn_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // attention sinks are per-head logits independent of the KV rows: the gathered attention keeps them
    // (GGML_CUDA_FA_SPARSE_SINKS=0: dense path for models with sinks, e.g. DeepSeek V4.1)
    static const bool sinks_ok = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_SINKS"); return !e || atoi(e) != 0; }();
    if (!spg_env() || !GGML_CUDA_CC_IS_GCN(cc) || !mask || (dst->src[4] && !sinks_ok)) {
        return false;
    }
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    // the query rows (decode, MTP verify) may each select different columns: the union is gathered
    const int64_t n_pad = GGML_PAD(std::min<int64_t>((int64_t) n_kv_max*Q->ne[1], K->ne[1]), 256);
    // several query rows (MTP verify) at >= 4K cells: the union is usually far smaller than n_q*n_kv_max, and the tile
    // kernel only walks the gathered rows (kv_max), so the path pays even when the buffer bound is the whole cache
    static const int verify_min = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_VERIFY_MIN_KV"); return e ? atoi(e) : 4096; }();
    const bool verify = verify_min > 0 && Q->ne[1] >= 2 && K->ne[1] >= verify_min;
    static const bool always = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_ALWAYS"); return e && atoi(e) != 0; }();
    return n_kv_max > 0 && (2*n_pad <= K->ne[1] || verify || always) && Q->ne[1] <= 8 && mask->ne[1] >= Q->ne[1] &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 &&
        K->ne[3] == Q->ne[3] && V->ne[3] == Q->ne[3] && (mask->ne[3] == Q->ne[3] || mask->ne[3] == 1) &&
        mask->ne[0] >= K->ne[1] && K->nb[0] == sizeof(half) && V->nb[0] == sizeof(half) &&
        (K->ne[0]*sizeof(half)) % 4 == 0 && K->ne[0] == V->ne[0] &&
        K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 && V->nb[1] % 4 == 0 && V->nb[2] % 4 == 0 &&
        ((uintptr_t) K->data) % 4 == 0 && ((uintptr_t) V->data) % 4 == 0;
}

void ggml_cuda_flash_attn_sparse_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int n_kv  = (int) K->ne[1];
    const int n_kvh = (int) K->ne[2];
    const int n_seq = (int) Q->ne[3];
    const int n_q   = (int) Q->ne[1];
    const int cap   = (int) std::min<int64_t>((int64_t) ggml_get_op_params_i32(dst, 4)*n_q, n_kv);
    const int n_pad = (int) GGML_PAD(cap, 256);
    const int64_t s_mask_row = mask->nb[1] / sizeof(half);
    const int row_bytes = (int) (K->ne[0]*sizeof(half));
    const int64_t s_mask_seq = mask->ne[3] == 1 ? 0 : mask->nb[3] / sizeof(half);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int32_t> idx(ctx.pool(), (size_t) n_seq*cap);
    ggml_cuda_pool_alloc<int32_t> cnt(ctx.pool(), 2*n_seq); // [count per sequence, finished-block ticket per sequence]
    const int n_blk = (n_kv + 255)/256;
    ggml_cuda_pool_alloc<unsigned long long> bits(ctx.pool(), (size_t) n_seq*n_blk*4);
    ggml_cuda_pool_alloc<char>    kg(ctx.pool(), (size_t) n_seq*n_pad*n_kvh*row_bytes);
    ggml_cuda_pool_alloc<char>    vg(ctx.pool(), (size_t) n_seq*n_pad*n_kvh*row_bytes);
    ggml_cuda_pool_alloc<half>    mg(ctx.pool(), (size_t) n_seq*n_q*n_pad);
    ggml_cuda_pool_alloc<int>     kvm(ctx.pool(), (size_t) n_seq*8);
    static const bool use_kvmax = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_KVMAX"); return !e || atoi(e) != 0; }();
    int * kv_max = use_kvmax && n_seq == 1 ? kvm.get() : nullptr; // entries are [query tile] of sequence 0

    CUDA_CHECK(cudaMemsetAsync(cnt.get(), 0, 2*n_seq*sizeof(int32_t), stream));
    spg_compact<<<dim3(n_blk, n_seq), 256, 0, stream>>>((const half *) mask->data, idx.get(), cnt.get(), bits.get(), n_kv, cap,
        n_q, s_mask_row, s_mask_seq);
    spg_gather<<<dim3(n_pad, n_seq), 64, 0, stream>>>((const char *) K->data, (const char *) V->data, (const half *) mask->data,
        idx.get(), cnt.get(), kg.get(), vg.get(), mg.get(), cap, n_pad, n_kvh, row_bytes,
        K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3], n_q, s_mask_row, s_mask_seq, kv_max);
    CUDA_CHECK(cudaGetLastError());
    static int dbg_left = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_DBG"); return e ? atoi(e) : 0; }();
    if (dbg_left > 0) {
        dbg_left--;
        int c = 0;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&c, cnt.get(), sizeof(int), cudaMemcpyDeviceToHost));
        fprintf(stderr, "spg: n_q %d n_kv %d cap %d union %d\n", n_q, n_kv, cap, c);
    }

    // the gathered tensors: cell-major like the cache ([D, n_pad, n_kvh, n_seq], heads of a cell adjacent)
    ggml_tensor K2 = *K, V2 = *V, M2 = *mask, dst2 = *dst;
    K2.data = kg.get();
    V2.data = vg.get();
    for (ggml_tensor * t : {&K2, &V2}) {
        t->ne[1] = n_pad;
        t->nb[1] = (size_t) n_kvh*row_bytes;
        t->nb[2] = row_bytes;
        t->nb[3] = (size_t) n_pad*n_kvh*row_bytes;
        t->view_src = nullptr;
        t->view_offs = 0;
    }
    M2.data  = mg.get();
    M2.ne[0] = n_pad;
    M2.ne[1] = n_q;
    M2.ne[2] = 1;
    M2.ne[3] = n_seq;
    M2.nb[1] = (size_t) n_pad*sizeof(half);
    M2.nb[2] = M2.nb[1]*n_q;
    M2.nb[3] = M2.nb[2];
    M2.view_src = nullptr;
    M2.view_offs = 0;
    dst2.src[1] = &K2;
    dst2.src[2] = &V2;
    dst2.src[3] = &M2;
    ggml_set_op_params_i32(&dst2, 4, 0);

    ggml_cuda_fattn_kv_max_override = kv_max;
    ggml_cuda_flash_attn_ext_tile(ctx, &dst2);
    ggml_cuda_fattn_kv_max_override = nullptr;
}

// ---- prefill ----
// DeepSeek V4.1 top-k attention: every query row sees <= n_kv_max cells (128 SWA + 512 top-k) of an n_kv that grows with
// the context (16K at the PR's cap), and nearly every 64-cell block holds some row's selection, so the dense tile kernel
// walks all of n_kv for every row. Here the T rows are cut into tiles of TQ consecutive rows (4 by default; unions of
// 4 rows measured ~900-1050 cells at n_kv ~7K vs 640 per row); spg_compact lists each tile's union, spgp_gather copies
// those K/V rows (once when K and V are the same tensor, as DeepSeek's shared latent) and the rows' mask values into a
// per-tile buffer padded to 256 (mask -inf beyond the union, K/V rows past the padded union are never read), and the
// tile kernel runs with the tiles as sequences: its mask scan (KV_max per sequence) stops each tile at its union.
// GGML_CUDA_FA_SPARSE_PREFILL=0 off, _TQ (2), _MIN_RATIO (0: n_kv >= ratio * TQ * n_kv_max).

// grid (n_pad, n_tiles), 64 threads
static __global__ void __launch_bounds__(64) spgp_gather(
        const char * __restrict__ K, const char * __restrict__ V, const half * __restrict__ mask,
        const int32_t * __restrict__ idx, const int32_t * __restrict__ count,
        char * __restrict__ Kg, char * __restrict__ Vg, half * __restrict__ mg,
        const int cap, const int n_pad, const int n_kvh, const int row_bytes,
        const int64_t k_nb1, const int64_t k_nb2, const int64_t v_nb1, const int64_t v_nb2,
        const int tq, const int64_t s_mask_row) {
    const int p    = blockIdx.x;
    const int tile = blockIdx.y;
    const int n    = min(count[tile], cap);
    const int n_used = min(n_pad, max(256, ((n + 255)/256)*256));
    const bool valid = p < n;
    const int col = valid ? idx[(int64_t) tile*cap + p] : 0;
    if (threadIdx.x < tq) {
        mg[((int64_t) tile*tq + threadIdx.x)*n_pad + p] = valid ?
            mask[((int64_t) tile*tq + threadIdx.x)*s_mask_row + col] : __float2half(-INFINITY);
    }
    if (p >= n_used) {
        return;
    }
    const int words = row_bytes / 4;
    for (int h = 0; h < n_kvh; ++h) {
        const int * ks = (const int *) (K + (int64_t) col*k_nb1 + (int64_t) h*k_nb2);
        int * kd = (int *) (Kg + (((int64_t) tile*n_pad + p)*n_kvh + h)*row_bytes);
        for (int w = threadIdx.x; w < words; w += 64) {
            kd[w] = ks[w];
        }
        if (Vg != Kg) {
            const int * vs = (const int *) (V + (int64_t) col*v_nb1 + (int64_t) h*v_nb2);
            int * vd = (int *) (Vg + (((int64_t) tile*n_pad + p)*n_kvh + h)*row_bytes);
            for (int w = threadIdx.x; w < words; w += 64) {
                vd[w] = vs[w];
            }
        }
    }
}

// default 2: the tile kernel then runs 16 columns (2 rows x 8 heads: 256 threads, occupancy 2) over smaller unions; V4.1
// TP4 pp4096 ub1024: d0 1258 -> 1333 t/s, d16K 1021 -> 1088 (tq 1: 1305 at d0)
// index mode (GGML_CUDA_FA_SPARSE_PREFILL_IDX=0: copy mode): per tile the tile kernel's K/V row list (the union, padded
// with its first row), the gathered mask rows (-inf past the union) and the tile's row count rounded up to the kernel's
// KV step. The copy mode moved every union row (1 KB) through a per-tile buffer: ~1 ms per layer at 1024 tokens, and
// ~400 MB of pool memory.
static __global__ void __launch_bounds__(256) spgp_rows_mask(
        const half * __restrict__ mask, const int32_t * __restrict__ idx, const int32_t * __restrict__ count,
        int32_t * __restrict__ kvbuf, half * __restrict__ mg, const int cap, const int n_pad, const int n_tiles, const int tq,
        const int64_t s_mask_row, const int kv_step) {
    const int tile = blockIdx.x;
    const int n    = min(count[tile], cap);
    const int32_t * ti = idx + (int64_t) tile*cap;
    int32_t * rows = kvbuf + n_tiles + (int64_t) tile*n_pad;
    const int r0 = n > 0 ? ti[0] : 0;
    for (int p = threadIdx.x; p < n_pad; p += blockDim.x) {
        const int r = p < n ? ti[p] : r0;
        rows[p] = r;
        for (int t = 0; t < tq; ++t) {
            mg[((int64_t) tile*tq + t)*n_pad + p] = p < n ? mask[((int64_t) tile*tq + t)*s_mask_row + r] : __float2half(-INFINITY);
        }
    }
    if (threadIdx.x == 0) {
        kvbuf[tile] = min(n_pad, ((n + kv_step - 1)/kv_step)*kv_step);
    }
}

static int spgp_tq() {
    static const int tq = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_PREFILL_TQ"); return e ? std::max(1, std::min(64, atoi(e))) : 2; }();
    return tq;
}

bool ggml_cuda_flash_attn_sparse_prefill_gcn_supported(const ggml_tensor * dst) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_PREFILL"); return !e || atoi(e) != 0; }();
    // default 0 (always): V4.1 TP4 pp4096 ub1024 d0 1208 -> 1252 t/s with the encoder's small caches included (the
    // dense tile kernel walks every window cell of the ubatch for each row), pp512 798 -> 813, d16K unchanged
    static const float min_ratio = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_PREFILL_MIN_RATIO"); return e ? (float) atof(e) : 0.0f; }();
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!env || !spg_env() || !GGML_CUDA_CC_IS_GCN(cc) || !mask) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    const int tq = spgp_tq();
    const int64_t T = Q->ne[1];
    const int64_t cap = std::min<int64_t>((int64_t) tq*n_kv_max, K->ne[1]);
    // >= 2 tiles: the tile kernel then scans the gathered mask for each sequence's KV_max and never reads the K/V rows
    // past a tile's padded union (left uninitialized)
    return n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f && T > 8 && T % tq == 0 && T/tq >= 2 && Q->ne[3] == 1 &&
        K->ne[3] == 1 && V->ne[3] == 1 && mask->ne[2] == 1 && mask->ne[3] == 1 && mask->ne[1] >= T &&
        (float) K->ne[1] >= min_ratio*(float) cap &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 && Q->type == GGML_TYPE_F32 &&
        mask->ne[0] >= K->ne[1] && K->nb[0] == sizeof(half) && V->nb[0] == sizeof(half) &&
        (K->ne[0]*sizeof(half)) % 4 == 0 && K->ne[0] == V->ne[0] && K->ne[2] == V->ne[2] &&
        K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 && V->nb[1] % 4 == 0 && V->nb[2] % 4 == 0 &&
        ((uintptr_t) K->data) % 4 == 0 && ((uintptr_t) V->data) % 4 == 0 &&
        dst->ne[2] == T && dst->ne[3] == 1 && ggml_is_contiguous(dst);
}

void ggml_cuda_flash_attn_sparse_prefill_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int tq      = spgp_tq();
    const int T       = (int) Q->ne[1];
    const int n_tiles = T / tq;
    const int n_kv    = (int) K->ne[1];
    const int n_kvh   = (int) K->ne[2];
    const int cap     = (int) std::min<int64_t>((int64_t) ggml_get_op_params_i32(dst, 4)*tq, n_kv);
    const int n_pad   = (int) GGML_PAD(cap, 256);
    const int64_t s_mask_row = mask->nb[1] / sizeof(half);
    const int row_bytes = (int) (K->ne[0]*sizeof(half));
    const bool same_kv = K->data == V->data && K->nb[1] == V->nb[1] && K->nb[2] == V->nb[2];

    // index mode: the tile kernel's 16-column D 512 instance (2 query rows x 8 heads) reads the union rows in place
    static const bool idx_env = [] { const char * e = getenv("GGML_CUDA_FA_SPARSE_PREFILL_IDX"); return !e || atoi(e) != 0; }();
    const bool use_idx = idx_env && tq == 2 && Q->ne[0] == 512 && K->ne[0] == 512 && V->ne[0] == 512 &&
        (Q->ne[2] / K->ne[2]) % 8 == 0 && n_pad % 256 == 0;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int32_t> idx(ctx.pool(), (size_t) n_tiles*cap);
    ggml_cuda_pool_alloc<int32_t> cnt(ctx.pool(), 2*n_tiles);
    const int n_blk = (n_kv + 255)/256;
    ggml_cuda_pool_alloc<unsigned long long> bits(ctx.pool(), (size_t) n_tiles*n_blk*4);
    ggml_cuda_pool_alloc<char>    kg(ctx.pool());
    ggml_cuda_pool_alloc<char>    vg(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> kvbuf(ctx.pool());
    if (use_idx) {
        kvbuf.alloc((size_t) n_tiles + (size_t) n_tiles*n_pad);
    } else {
        kg.alloc((size_t) n_tiles*n_pad*n_kvh*row_bytes);
        if (!same_kv) {
            vg.alloc((size_t) n_tiles*n_pad*n_kvh*row_bytes);
        }
    }
    ggml_cuda_pool_alloc<half>    mg(ctx.pool(), (size_t) n_tiles*tq*n_pad);

    CUDA_CHECK(cudaMemsetAsync(cnt.get(), 0, 2*n_tiles*sizeof(int32_t), stream));
    spg_compact<<<dim3(n_blk, n_tiles), 256, 0, stream>>>((const half *) mask->data, idx.get(), cnt.get(), bits.get(), n_kv, cap,
        tq, s_mask_row, (int64_t) tq*s_mask_row);
    char * vgp = same_kv ? kg.get() : vg.get();
    if (use_idx) {
        spgp_rows_mask<<<n_tiles, 256, 0, stream>>>((const half *) mask->data, idx.get(), cnt.get(), kvbuf.get(), mg.get(), cap,
            n_pad, n_tiles, tq, s_mask_row, 64);
    } else {
        spgp_gather<<<dim3(n_pad, n_tiles), 64, 0, stream>>>((const char *) K->data, (const char *) V->data, (const half *) mask->data,
            idx.get(), cnt.get(), kg.get(), vgp, mg.get(), cap, n_pad, n_kvh, row_bytes, K->nb[1], K->nb[2], V->nb[1], V->nb[2],
            tq, s_mask_row);
    }
    CUDA_CHECK(cudaGetLastError());

    // tiles as sequences: Q [D, tq, H, n_tiles], K/V [D, n_pad, n_kvh, n_tiles], mask [n_pad, tq, 1, n_tiles],
    // dst [D, H, tq, n_tiles]
    ggml_tensor Q2 = *Q, K2 = *K, V2 = *V, M2 = *mask, dst2 = *dst;
    Q2.ne[1] = tq;
    Q2.ne[3] = n_tiles;
    Q2.nb[3] = (size_t) tq*Q->nb[1];
    if (use_idx) {
        // the original rows: row p of tile s is K/V row rows[s][p] (the tile kernel adds no per-tile offset: nb[3] = 0)
        for (ggml_tensor * t : {&K2, &V2}) {
            t->ne[1] = n_pad;
            t->ne[3] = n_tiles;
            t->nb[3] = 0;
            t->view_src = nullptr;
            t->view_offs = 0;
        }
    } else {
        K2.data = kg.get();
        V2.data = vgp;
        for (ggml_tensor * t : {&K2, &V2}) {
            t->ne[1] = n_pad;
            t->ne[3] = n_tiles;
            t->nb[1] = (size_t) n_kvh*row_bytes;
            t->nb[2] = row_bytes;
            t->nb[3] = (size_t) n_pad*n_kvh*row_bytes;
            t->view_src = nullptr;
            t->view_offs = 0;
        }
    }
    M2.data  = mg.get();
    M2.ne[0] = n_pad;
    M2.ne[1] = tq;
    M2.ne[2] = 1;
    M2.ne[3] = n_tiles;
    M2.nb[1] = (size_t) n_pad*sizeof(half);
    M2.nb[2] = M2.nb[1]*tq;
    M2.nb[3] = M2.nb[2];
    M2.view_src = nullptr;
    M2.view_offs = 0;
    dst2.ne[2] = tq;
    dst2.ne[3] = n_tiles;
    dst2.nb[3] = (size_t) tq*dst->nb[2];
    dst2.src[0] = &Q2;
    dst2.src[1] = &K2;
    dst2.src[2] = &V2;
    dst2.src[3] = &M2;
    ggml_set_op_params_i32(&dst2, 4, 0);
    if (use_idx) {
        ggml_cuda_fattn_kv_max_override = kvbuf.get();
        ggml_cuda_fattn_tile_kv_idx      = true;
        ggml_cuda_fattn_tile_kv_idx_used = false;
        ggml_cuda_flash_attn_ext_tile(ctx, &dst2);
        ggml_cuda_fattn_tile_kv_idx      = false;
        ggml_cuda_fattn_kv_max_override  = nullptr;
        GGML_ASSERT(ggml_cuda_fattn_tile_kv_idx_used && "sparse prefill index mode: the tile launcher picked another instance");
    } else {
        ggml_cuda_flash_attn_ext_tile(ctx, &dst2);
    }
}

// ---- DeepSeek V4(.1) sparse decode attention (GGML_OP_DSV4_SPARSE_ATTN) ----
// Row p of the gathered cells: p < n_raw the window cell p (mask_raw per query row), then each query row's n_sel top-k
// rows of the compressed cache (visible to that row only, where mask_comp allows), padding to n_pad under -inf. K == V
// (the shared latent), so one buffer serves both. Replaces the k_all concat (the whole compressed cache per layer), the
// top-k mask build, the mask scan and the second gather copy of the dense sparse path.
static __global__ void __launch_bounds__(64) dsv4sa_gather(
        const char * __restrict__ kr, const char * __restrict__ kc, const half * __restrict__ mr,
        const half * __restrict__ mc, const int32_t * __restrict__ tk, char * __restrict__ kg, half * __restrict__ mg,
        const int n_raw, const int n_comp, const int n_sel, const int nt, const int n_pad, const int row_bytes,
        const int64_t kr_nb2, const int64_t kc_nb2, const int64_t s_mr, const int64_t s_mc, const int64_t s_tk,
        int * __restrict__ kv_max) {
    const int p      = blockIdx.x;
    const int n_rows = n_raw + nt*n_sel;
    if (p == 0 && threadIdx.x < 8 && kv_max) {
        kv_max[threadIdx.x] = n_pad;
    }
    const char * src = kr;
    if (p < n_raw) {
        src = kr + (int64_t) p*kr_nb2;
        if (threadIdx.x < nt) {
            mg[(int64_t) threadIdx.x*n_pad + p] = mr[(int64_t) threadIdx.x*s_mr + p];
        }
    } else if (p < n_rows) {
        const int  j  = p - n_raw;
        const int  tj = j / n_sel;
        const int  c  = tk[(int64_t) tj*s_tk + (j % n_sel)];
        const bool ok = c >= 0 && c < n_comp;
        src = kc + (int64_t) (ok ? c : 0)*kc_nb2;
        if (threadIdx.x < nt) {
            mg[(int64_t) threadIdx.x*n_pad + p] = threadIdx.x == tj && ok ? mc[(int64_t) tj*s_mc + c] : __float2half(-INFINITY);
        }
    } else if (threadIdx.x < nt) {
        mg[(int64_t) threadIdx.x*n_pad + p] = __float2half(-INFINITY);
    }
    const int * s4 = (const int *) src;
    int * d4 = (int *) (kg + (int64_t) p*row_bytes);
    for (int w = threadIdx.x; w < row_bytes/4; w += 64) {
        d4[w] = s4[w];
    }
}

// Direct kernel (D = 512, one KV head shared by all query heads): workgroup = (group of HG = 4 heads, chunk of NW*CPW of
// the token's cells, token), NW waves of CPW cells. The rows are read once per workgroup and used by its 4 heads (16 heads
// re-reading every row was L2-bound). Lane l owns dims 8l..8l+7 of every row, so a row is one coalesced 1 KB load per
// wave; lanes 0..CPW-1 resolve the wave's cells (window cell or top-k row, mask) and the row indices are broadcast
// through SGPRs, so all loads of a wave are in flight at once, and masked cells (top-k slots past the valid compressed
// rows, window cells past the position) load and compute nothing.
//   QK: v_dot2_f32_f16 on the lane's 8 dims (q*scale in f16), quad sums by DPP (xor 1, 2), the 16 quad sums of each
//       (head, cell) summed by lane (head, cell) through LDS.
//   softmax over the chunk per head; p in f16, packed in cell pairs.
//   PV: v_dot2_f32_f16 of the p pair with the (cell, cell + 1) value pairs of the rows in registers (K == V, f32
//       accumulation); the waves' outputs are summed through LDS.
// The last chunk of a (token, head group) to finish (arrival counter) merges the chunks' (max, sum, output) with the
// sinks in one round of loads. Replaces the gather, the tile kernel and its combine pass.
static constexpr int DSA_D       = 512;
static constexpr int DSA_PS      = DSA_D + 4;   // partial: output, max, sum (16-B aligned rows)
static constexpr int DSA_HG      = 4;           // heads per workgroup
static constexpr int DSA_CSTRIDE = 64;          // arrival counters 256 B apart

typedef _Float16 dsa_h2 __attribute__((ext_vector_type(2)));

#define DSA_TRACE 0 // timestamps: s_memrealtime, 25 MHz on gfx906 (0.04 us per tick; the 0.01 used until 2026-10-04 read 4x short)
#define DSA_WMAX  1024
#ifndef DSA_QLO
#define DSA_QLO 0 // 1: also the f16 residual of q*scale in QK (DS4 PPL ub6 14.427 -> 14.426, tg -0.3%; GLM MQA512 path 13.108 -> 13.077)
#endif // window slots the decode kernel compacts (larger SWA views keep every slot as a cell)
#if DSA_TRACE
static __device__ unsigned long long dsa_trace[4096][8];
#define DSA_T(ph) do { if (threadIdx.x == 0) { dsa_trace[blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z)][ph] = __builtin_amdgcn_s_memrealtime(); } } while (0)
#else
#define DSA_T(ph)
#endif

template <int ctrl> static __device__ __forceinline__ float dsa_dpp(const float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), ctrl, 0xF, 0xF, true));
}

static __device__ __forceinline__ float dsa_dot8(const uint4 w, const dsa_h2 * qh) {
    float s = __builtin_amdgcn_fdot2(__builtin_bit_cast(dsa_h2, w.x), qh[0], 0.0f, false);
    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(dsa_h2, w.y), qh[1], s, false);
    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(dsa_h2, w.z), qh[2], s, false);
    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(dsa_h2, w.w), qh[3], s, false);
    return s;
}

// optional NORM rope of the query (dims [offs, offs + dims), before attention) and its inverse on the output (the math of
// rope.cu's rope_norm + rope_yarn: ggml_dsv4_sparse_attn_set_rope)
struct dsa_rope {
    int   dims, offs;
    float theta_scale, freq_scale, ext_factor, attn_factor, corr0, corr1;
};

static __device__ __forceinline__ float2 dsa_rope_cs(const dsa_rope & rp, const int pos, const int pair) {
    const int   iw           = 2*pair;
    const float theta_extrap = pos*powf(rp.theta_scale, iw/2.0f);
    float theta  = rp.freq_scale*theta_extrap;
    float mscale = rp.attn_factor;
    if (rp.ext_factor != 0.0f) {
        const float y = (iw/2 - rp.corr0) / max(0.001f, rp.corr1 - rp.corr0);
        const float ramp_mix = (1.0f - min(1.0f, max(0.0f, y)))*rp.ext_factor;
        theta = theta*(1 - ramp_mix) + theta_extrap*ramp_mix;
        mscale *= 1.0f + 0.1f*logf(1.0f/rp.freq_scale);
    }
    float sn, cs;
    sincosf(theta, &sn, &cs); // one range reduction for both (the separate rope kernel calls cosf, sinf)
    return make_float2(cs*mscale, sn*mscale);
}

template <int NW, int CPW>
static __global__ void __launch_bounds__(64*NW) dsv4sa_attn(
        const float * __restrict__ q, const char * __restrict__ kr, const char * __restrict__ kc,
        const half * __restrict__ mr, const half * __restrict__ mc, const int32_t * __restrict__ tk,
        const float * __restrict__ sinks, float * __restrict__ dst, float * __restrict__ part, int * __restrict__ counters,
        const int n_raw, const int n_comp, const int n_sel, const int n_head,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t kr_nb2, const int64_t kc_nb2,
        const int64_t s_mr, const int64_t s_mc, const int64_t s_tk, const int64_t d_nb1, const int64_t d_nb2, const float scale,
        const int32_t * __restrict__ rpos, const dsa_rope rp, const bool spec, const bool wcompact) {
    constexpr int HG  = DSA_HG;
    constexpr int NT  = 64*NW;
    constexpr int CPB = NW*CPW;              // cells per workgroup
    constexpr int NV  = HG*CPW;              // (head, cell) values per wave
    constexpr int RH  = 8/NW;                // heads per round of the cross-wave sum (16 KB of LDS)
    constexpr int DPT = RH*DSA_D/NT;         // dims per thread in a round
    static_assert(NV <= 64 && (CPW == 8 || CPW == 16) && (NW == 4 || NW == 8), "one lane per (head, cell) of a wave");
    const int hg    = blockIdx.x;
    const int chunk = blockIdx.y;
    const int nch   = gridDim.y;
    const int t     = blockIdx.z;
    const int tid   = threadIdx.x;
    const int lane  = tid & 63;
    const int wv    = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int h0    = hg*HG;
    DSA_T(0);

    // qs (QK quad sums, 16 quads x NV per wave) and red (PV, RH heads per round) share the buffer
    constexpr int BIG = NW*16*NV > NW*RH*DSA_D ? NW*16*NV : NW*RH*DSA_D;
    __shared__ float    big[BIG];
    __shared__ float    cmask[NW][CPW];      // [wave][cell] mask (-inf: no cell)
    __shared__ float    sc[HG][CPB];         // masked scores
    __shared__ uint32_t pp[NW][HG][CPW/2];   // [wave][head][cell pair] p in f16 pairs
    __shared__ float    wl[NW][HG];          // per-wave sums of p
    __shared__ float    s_mh[HG];            // per-head max over the chunk
    __shared__ int      s_last;

    // the query of the group's heads first (dims 8*lane .. 8*lane + 7): it is converted while the rows are in flight
    float4 qa[HG], qb[HG];
#pragma unroll
    for (int hh = 0; hh < HG; ++hh) {
        const float4 * qp = (const float4 *) ((const char *) q + (h0 + hh)*q_nb1 + t*q_nb2) + 2*lane;
        qa[hh] = qp[0];
        qb[hh] = qp[1];
    }

    // lanes 0..CPW-1: cell c0 + lane -> source (0 none, 1 window, 2 compressed), row, mask
    const int c0 = chunk*CPB + wv*CPW;
    const int cl = c0 + lane;
    half mraw = __float2half(-INFINITY);
    int  idx  = -1;
    int  n_win = n_raw;                      // window cells taking part
    int  wslot = cl;                         // the window cell's cache slot
    bool raw_c = false;
    if (wcompact) {
        // the top-k picks first (their indices go out in the first round), then only the valid window cells: the SWA cache
        // view covers its whole size (512 slots at 16K context, ~128 valid in decode), and its masked slots would otherwise
        // take chunks of their own (64 workgroups at one per CU: a second round)
        __shared__ int   s_wslot[DSA_WMAX];
        __shared__ float s_wmask[DSA_WMAX];
        __shared__ int   s_nwin;
        if (lane < CPW && cl < n_sel) {
            idx = tk[t*s_tk + cl];
        }
        if (wv == 0) {
            // wave 0: 8 cells per lane per 512, a prefix count over the lanes, valid cells written at their offsets
            const half * mrow = mr + t*s_mr;
            int base = 0;
            for (int blk = 0; blk < n_raw; blk += 512) {
                const int c8 = blk + 8*lane;
                float    mv[8];
                unsigned bits = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    mv[i] = __half2float(mrow[min(c8 + i, n_raw - 1)]);
                    bits |= (c8 + i < n_raw && mv[i] != -INFINITY ? 1u : 0u) << i;
                }
                const int cnt = __popc(bits);
                int incl = cnt;
#pragma unroll
                for (int off = 1; off < 64; off <<= 1) {
                    const int v = __shfl_up(incl, off, 64);
                    incl += lane >= off ? v : 0;
                }
                int w = base + incl - cnt;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if ((bits >> i) & 1) {
                        s_wslot[w] = c8 + i;
                        s_wmask[w] = mv[i];
                        ++w;
                    }
                }
                base += __shfl(incl, 63, 64);
            }
            if (lane == 0) {
                s_nwin = base;
            }
        }
        __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory"); // LDS only: the index loads stay in flight
        n_win = s_nwin;
        raw_c = lane < CPW && cl >= n_sel && cl < n_sel + n_win;
        if (raw_c) {
            wslot = s_wslot[cl - n_sel];
            mraw  = __float2half(s_wmask[cl - n_sel]);
        }
    } else {
        // first round: the window mask or the top-k index (issued before the rope table so its math overlaps them)
        const bool lv0 = lane < CPW && cl < n_raw + n_sel;
        raw_c = lv0 && cl < n_raw;
        if (raw_c) {
            mraw = mr[t*s_mr + cl];
        } else if (lv0) {
            idx = tk[t*s_tk + (cl - n_raw)];
        }
    }
    const int  n_cells = n_win + n_sel;
    const bool lv      = lane < CPW && cl < n_cells;

    // rope: every wave tabulates the token's (cos, sin) per pair in its own LDS slot (no workgroup barrier)
    __shared__ float2 rcs[NW][32];
    if (rpos) {
        const int pos = rpos[t];
        if (lane < rp.dims/2) {
            rcs[wv][lane] = dsa_rope_cs(rp, pos, lane);
        }
        __builtin_amdgcn_wave_barrier();
    }

    // which rows to load: a window cell's mask comes with the first round; a compressed cell's mask is a second, dependent
    // round trip, so with spec (the host saw at least top-k + 256 compressed rows: every pick is a real row) the row goes
    // out with its mask instead of after it. Masked cells are dropped before QK/PV either way.
    int sld  = 0;
    int rowi = 0;
    half mcv = __float2half(-INFINITY);
    if (raw_c) {
        sld  = __half2float(mraw) != -INFINITY ? 1 : 0;
        rowi = wslot;
    } else if (lv && idx >= 0 && idx < n_comp) {
        mcv  = mc[t*s_mc + idx];
        rowi = idx;
        if (spec) {
            sld = 2;
        } else {
            sld = __half2float(mcv) != -INFINITY ? 2 : 0;
        }
    }
#if DSA_TRACE
    __builtin_amdgcn_s_waitcnt(0);
#endif
    DSA_T(1);

    uint4 w[CPW];
    int   sjv[CPW];
#pragma unroll
    for (int j = 0; j < CPW; ++j) {
        const int sj = __builtin_amdgcn_readlane(sld, j);
        const int rj = __builtin_amdgcn_readlane(rowi, j);
        w[j] = make_uint4(0, 0, 0, 0);
        if (sj == 1) {
            w[j] = *((const uint4 *) (kr + (int64_t) rj*kr_nb2) + lane);
        } else if (sj == 2) {
            w[j] = *((const uint4 *) (kc + (int64_t) rj*kc_nb2) + lane);
        }
    }

    // q*scale (f16), rotated in the lanes whose 8 dims are in the rope range
    dsa_h2 qh[HG][4];
    dsa_h2 ql[HG][4]; // DSA_QLO: the f16 rounding residual of q*scale (a second fdot2 per pair: ~22-bit query)
    const bool rlane = rpos && 8*lane >= rp.offs && 8*lane < rp.offs + rp.dims;   // the lane's 8 dims are rotated
    float2 cs[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        cs[k] = rlane ? rcs[wv][(8*lane - rp.offs)/2 + k] : make_float2(1.0f, 0.0f);
    }
#pragma unroll
    for (int hh = 0; hh < HG; ++hh) {
        float4 a = qa[hh];
        float4 b = qb[hh];
        if (rlane) {
            float x0, x1;
            x0 = a.x; x1 = a.y; a.x = x0*cs[0].x - x1*cs[0].y; a.y = x0*cs[0].y + x1*cs[0].x;
            x0 = a.z; x1 = a.w; a.z = x0*cs[1].x - x1*cs[1].y; a.w = x0*cs[1].y + x1*cs[1].x;
            x0 = b.x; x1 = b.y; b.x = x0*cs[2].x - x1*cs[2].y; b.y = x0*cs[2].y + x1*cs[2].x;
            x0 = b.z; x1 = b.w; b.z = x0*cs[3].x - x1*cs[3].y; b.w = x0*cs[3].y + x1*cs[3].x;
        }
        const float qs[8] = { a.x*scale, a.y*scale, a.z*scale, a.w*scale, b.x*scale, b.y*scale, b.z*scale, b.w*scale };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const _Float16 h0 = (_Float16) qs[2*k], h1 = (_Float16) qs[2*k + 1];
            qh[hh][k] = dsa_h2{h0, h1};
            ql[hh][k] = DSA_QLO ? dsa_h2{(_Float16) (qs[2*k] - (float) h0), (_Float16) (qs[2*k + 1] - (float) h1)} : dsa_h2{0, 0};
        }
    }

    // the masks (back before the rows: vector memory returns in issue order); a masked cell takes no part
    const float mval = raw_c ? __half2float(mraw) : (sld == 2 ? __half2float(mcv) : -INFINITY);
    const int   src  = sld != 0 && mval != -INFINITY ? sld : 0;
    if (lane < CPW) {
        cmask[wv][lane] = src ? mval : -INFINITY;
    }
#pragma unroll
    for (int j = 0; j < CPW; ++j) {
        sjv[j] = __builtin_amdgcn_readlane(src, j);
    }

    // QK in batches of 4 cells: partial dots, quad sums (DPP), stored per quad: qs[wave][quad][head*CPW + cell]
    float * qs = big + wv*16*NV;
#pragma unroll
    for (int b4 = 0; b4 < CPW/4; ++b4) {
        float v[HG][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = 4*b4 + i;
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                v[hh][i] = sjv[j] ? (DSA_QLO ? dsa_dot8(w[j], ql[hh]) : 0.0f) + dsa_dot8(w[j], qh[hh]) : 0.0f;
            }
        }
#pragma unroll
        for (int hh = 0; hh < HG; ++hh) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[hh][i] += dsa_dpp<0xB1>(v[hh][i]);
            }
        }
#pragma unroll
        for (int hh = 0; hh < HG; ++hh) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[hh][i] += dsa_dpp<0x4E>(v[hh][i]);
            }
        }
#pragma unroll
        for (int hh = 0; hh < HG; ++hh) {
            *(float4 *) &qs[(lane >> 2)*NV + hh*CPW + 4*b4] = make_float4(v[hh][0], v[hh][1], v[hh][2], v[hh][3]);
        }
    }
    DSA_T(2);
    __builtin_amdgcn_wave_barrier();
    // lane (head, cell): the score
    if (lane < NV) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            s += qs[k*NV + lane];
        }
        const int   hh = lane / CPW;
        const int   c  = lane % CPW;
        const float m  = cmask[wv][c];
        sc[hh][wv*CPW + c] = m == -INFINITY ? -INFINITY : s + m;
    }
    __syncthreads();
    DSA_T(3);

    // softmax over the chunk, per head: lane (head, cell) of the wave's cells; p in f16 pairs
    if (lane < NV) {
        const int hh = lane / CPW;
        const int c  = lane % CPW;
        float mh = -INFINITY;
#pragma unroll
        for (int k = 0; k < CPB; k += 4) {
            const float4 x = *(const float4 *) &sc[hh][k];
            mh = fmaxf(mh, fmaxf(fmaxf(x.x, x.y), fmaxf(x.z, x.w)));
        }
        const float sv = sc[hh][wv*CPW + c];
        const float p  = sv == -INFINITY ? 0.0f : __expf(sv - mh);
        const float pn = dsa_dpp<0xB1>(p);   // the pair's other cell
        if ((c & 1) == 0) {
            pp[wv][hh][c/2] = __builtin_bit_cast(uint32_t, dsa_h2{(_Float16) p, (_Float16) pn});
        }
        float ps = p;
        ps += dsa_dpp<0xB1>(ps);
        ps += dsa_dpp<0x4E>(ps);
        ps += dsa_dpp<0x141>(ps);            // row_half_mirror: 8 lanes
        if (CPW == 16) {
            ps += dsa_dpp<0x140>(ps);        // row_mirror: 16 lanes
        }
        if (c == 0) {
            wl[wv][hh] = ps;
            if (wv == 0) {
                s_mh[hh] = mh;
            }
        }
    }
    __builtin_amdgcn_wave_barrier();

    // PV: (cell, cell + 1) value pairs of the rows in registers against the p pairs
    float o[HG][8];
#pragma unroll
    for (int hh = 0; hh < HG; ++hh) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            o[hh][k] = 0.0f;
        }
    }
#pragma unroll
    for (int j = 0; j < CPW; j += 2) {
        if (sjv[j] | sjv[j + 1]) {
            const uint32_t * a = (const uint32_t *) &w[j];
            const uint32_t * b = (const uint32_t *) &w[j + 1];
            dsa_h2 lo[4], hi[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                lo[m] = __builtin_bit_cast(dsa_h2, __builtin_amdgcn_perm(b[m], a[m], 0x05040100));
                hi[m] = __builtin_bit_cast(dsa_h2, __builtin_amdgcn_perm(b[m], a[m], 0x07060302));
            }
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                const dsa_h2 p2 = __builtin_bit_cast(dsa_h2, pp[wv][hh][j/2]);
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    o[hh][2*m]     = __builtin_amdgcn_fdot2(p2, lo[m], o[hh][2*m],     false);
                    o[hh][2*m + 1] = __builtin_amdgcn_fdot2(p2, hi[m], o[hh][2*m + 1], false);
                }
            }
        }
    }

    DSA_T(4);
    // the waves' outputs, RH heads per round, then the partials: (max, sum, output) per head
#pragma unroll
    for (int r = 0; r < HG/RH; ++r) {
        __syncthreads();
#pragma unroll
        for (int k = 0; k < RH; ++k) {
            float * rp = big + (wv*RH + k)*DSA_D + 8*lane;
            *(float4 *) rp       = make_float4(o[RH*r + k][0], o[RH*r + k][1], o[RH*r + k][2], o[RH*r + k][3]);
            *(float4 *) (rp + 4) = make_float4(o[RH*r + k][4], o[RH*r + k][5], o[RH*r + k][6], o[RH*r + k][7]);
        }
        __syncthreads();
        const int k  = (tid*DPT)/DSA_D;      // head of the round
        const int d0 = (tid*DPT)%DSA_D;
        float x[DPT];
#pragma unroll
        for (int i = 0; i < DPT; ++i) {
            x[i] = 0.0f;
        }
#pragma unroll
        for (int ww = 0; ww < NW; ++ww) {
#pragma unroll
            for (int i = 0; i < DPT; ++i) {
                x[i] += big[(ww*RH + k)*DSA_D + d0 + i];
            }
        }
        const int hh = RH*r + k;
        float * pt = part + (((int64_t) t*n_head + h0 + hh)*nch + chunk)*DSA_PS;
#pragma unroll
        for (int i = 0; i < DPT; ++i) {
            pt[d0 + i] = x[i];
        }
        if (d0 == 0) {
            float l = 0.0f;
#pragma unroll
            for (int ww = 0; ww < NW; ++ww) {
                l += wl[ww][hh];
            }
            pt[DSA_D]     = s_mh[hh];
            pt[DSA_D + 1] = l;
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(counters + DSA_CSTRIDE*(t*gridDim.x + hg), 1) == nch - 1;
    }
    __syncthreads();
    DSA_T(5);
    if (!s_last) {
        return;
    }
    __threadfence();

    // merge: NW/4 waves per head; the wave's (max, sum) of chunk `lane` and its outputs of all chunks (NCHM per
    // round) are loaded together; the chunk scales go through SGPRs (readlane)
    {
        constexpr int WPH = NW/4;            // waves per head
        constexpr int MD  = DSA_D/(64*WPH);  // dims per thread: 8 (NW = 4) or 4 (NW = 8)
        constexpr int NCHM = 64/MD;          // chunks per round of loads (16 float4 in flight)
        const int     hh  = wv/WPH;
        const int     d0  = ((wv % WPH)*64 + lane)*MD;
        const float * pb  = part + ((int64_t) t*n_head + h0 + hh)*nch*DSA_PS;
        float a[MD];
#pragma unroll
        for (int i = 0; i < MD; ++i) {
            a[i] = 0.0f;
        }
        float mk = -INFINITY, lk = 0.0f;
        if (lane < nch) {
            mk = pb[lane*DSA_PS + DSA_D];
            lk = pb[lane*DSA_PS + DSA_D + 1];
        }
        float4 ov[NCHM][MD/4];
#pragma unroll
        for (int k = 0; k < NCHM; ++k) {
#pragma unroll
            for (int i = 0; i < MD/4; ++i) {
                ov[k][i] = k < nch ? *(const float4 *) (pb + k*DSA_PS + d0 + 4*i) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        const float sk = sinks ? sinks[h0 + hh] : -INFINITY;
        float M = mk;
#pragma unroll
        for (int off = 32; off >= 1; off >>= 1) {
            M = fmaxf(M, __shfl_xor(M, off, 64));
        }
        M = fmaxf(M, sk);
        const float r = mk == -INFINITY ? 0.0f : __expf(mk - M);
        float L = r*lk;
#pragma unroll
        for (int off = 32; off >= 1; off >>= 1) {
            L += __shfl_xor(L, off, 64);
        }
        L += sk == -INFINITY ? 0.0f : __expf(sk - M);
#pragma unroll
        for (int k = 0; k < NCHM; ++k) {
            const float rk = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(r), k));
#pragma unroll
            for (int i = 0; i < MD/4; ++i) {
                a[4*i + 0] = fmaf(rk, ov[k][i].x, a[4*i + 0]);
                a[4*i + 1] = fmaf(rk, ov[k][i].y, a[4*i + 1]);
                a[4*i + 2] = fmaf(rk, ov[k][i].z, a[4*i + 2]);
                a[4*i + 3] = fmaf(rk, ov[k][i].w, a[4*i + 3]);
            }
        }
        for (int k0 = NCHM; k0 < nch; k0 += NCHM) {   // more chunks than one round (not hit in practice)
            for (int k = k0; k < min(nch, k0 + NCHM); ++k) {
                const float rk = __shfl(r, k, 64);
#pragma unroll
                for (int i = 0; i < MD; ++i) {
                    a[i] = fmaf(rk, pb[k*DSA_PS + d0 + i], a[i]);
                }
            }
        }
        const float inv = L > 0.0f ? 1.0f/L : 0.0f;
#pragma unroll
        for (int i = 0; i < MD; ++i) {
            a[i] *= inv;
        }
        if (rpos && d0 >= rp.offs && d0 < rp.offs + rp.dims) {   // inverse rotation (sin negated)
#pragma unroll
            for (int i = 0; i < MD; i += 2) {
                const float2 c  = rcs[wv][(d0 + i - rp.offs)/2];
                const float  sn = -c.y;
                const float  x0 = a[i], x1 = a[i + 1];
                a[i]     = x0*c.x - x1*sn;
                a[i + 1] = x0*sn  + x1*c.x;
            }
        }
        float * out = dst + (h0 + hh)*d_nb1 + t*d_nb2 + d0;
#pragma unroll
        for (int i = 0; i < MD; ++i) {
            out[i] = a[i];
        }
    }
#if DSA_TRACE
    __builtin_amdgcn_s_waitcnt(0);
#endif
    DSA_T(6);
    if (tid == 0) {
        counters[DSA_CSTRIDE*(t*gridDim.x + hg)] = 0;
    }
}

// MQA on a 512-dim latent row that serves as both K and V (GLM-5-Next's absorbed MLA: one KV head, 16 query heads per
// GPU) for decode / verify (<= 8 query rows, <= 4096 cells): the direct kernel above with every cell a window cell under
// its mask row and no top-k part. The tile kernel (no vector kernel for D = 512 on GCN) took ~60 us per layer for a
// 2-row MTP verify at 256 cells. Only while the launch is one round of workgroups (<= 64: one 512-thread workgroup per CU
// fits): 16 heads, 1 row: 256-768 cells 16-18 us vs 25-27 (tile), 1024 even, 2048 44 vs 38; 2 rows: 256 cells 16 vs 32,
// 512 28 vs 32, 1024 40 vs 36. Opt-in (GGML_CUDA_FA_MQA512=1): GLM-5.3 Q2 pp2 109.0 -> 111.2, tg 67.8 -> 68.8 at short
// context, but its f16 query / probabilities moved GLM's PPL (ub2 12.985 -> 13.108; 13.077 with the query residual)
bool ggml_cuda_flash_attn_mqa512_gcn_supported(const ggml_tensor * dst) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_FA_MQA512"); return e && atoi(e) != 0; }();
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!env || !GGML_CUDA_CC_IS_GCN(cc) || !mask) {
        return false;
    }
    const float max_bias      = ggml_get_op_params_f32(dst, 1);
    const float logit_softcap = ggml_get_op_params_f32(dst, 2);
    const int64_t nt     = Q->ne[1];
    const int64_t n_head = Q->ne[2];
    const int64_t n_kv   = K->ne[1];
    return Q->type == GGML_TYPE_F32 && Q->ne[0] == DSA_D && Q->ne[3] == 1 && Q->nb[0] == sizeof(float) &&
        Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && ((uintptr_t) Q->data) % 16 == 0 &&
        nt >= 1 && nt <= 8 && n_head % DSA_HG == 0 && (int64_t) DSA_CSTRIDE*nt*(n_head/DSA_HG) <= 65535 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 && K->data == V->data && K->ne[0] == DSA_D &&
        V->ne[0] == DSA_D && K->nb[1] == V->nb[1] && K->ne[2] == 1 && K->ne[3] == 1 && V->ne[2] == 1 && V->ne[3] == 1 &&
        K->nb[0] == sizeof(half) && K->nb[1] % 16 == 0 && ((uintptr_t) K->data) % 16 == 0 &&
        n_kv >= 1 && (n_head/DSA_HG)*((n_kv + 63)/64)*nt <= 64 &&
        mask->type == GGML_TYPE_F16 && mask->nb[0] == sizeof(half) && mask->ne[0] >= n_kv && mask->ne[1] >= nt &&
        mask->ne[2] == 1 && mask->ne[3] == 1 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (sinks == nullptr || (sinks->type == GGML_TYPE_F32 && sinks->ne[0] >= n_head)) &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && dst->ne[0] == DSA_D && dst->ne[1] == n_head && dst->ne[2] == nt;
}

void ggml_cuda_flash_attn_mqa512_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const int nt     = (int) Q->ne[1];
    const int n_head = (int) Q->ne[2];
    const int n_kv   = (int) K->ne[1];
    constexpr int NW = 8, CPW = 8;
    const int nch = (n_kv + NW*CPW - 1)/(NW*CPW);
    cudaStream_t stream = ctx.stream();
    if (ctx.hc_mix_counters == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &ctx.hc_mix_counters, 65535*sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ctx.hc_mix_counters, 0, 65535*sizeof(int), stream));
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) nt*n_head*nch*DSA_PS);
    const dsa_rope rp = {0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int64_t s_m = mask->nb[1]/sizeof(half);
    // Q [D, nt, n_head]: head stride nb[2], token stride nb[1]; dst [D, n_head, nt]
    dsv4sa_attn<NW, CPW><<<dim3(n_head/DSA_HG, nch, nt), 64*NW, 0, stream>>>((const float *) Q->data,
        (const char *) K->data, (const char *) K->data, (const half *) mask->data, (const half *) mask->data, nullptr,
        sinks ? (const float *) sinks->data : nullptr, (float *) dst->data, part.get(), ctx.hc_mix_counters,
        n_kv, 0, 0, n_head, Q->nb[2], Q->nb[1], K->nb[1], K->nb[1], s_m, s_m, 0,
        dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), ggml_get_op_params_f32(dst, 0), nullptr, rp, false, false);
    CUDA_CHECK(cudaGetLastError());
}

// the rope is implemented by the direct kernel only: D = 512, whole lanes (8 dims) rotated, <= 64 pairs
static bool dsa_rope_ok(const ggml_tensor * dst) {
    static const bool direct = [] { const char * e = getenv("GGML_CUDA_DSA_DIRECT"); return !e || atoi(e) != 0; }();
    const int dims = ggml_get_op_params_i32(dst, 1);
    const int offs = ggml_get_op_params_i32(dst, 2);
    return direct && dst->src[0]->ne[0] == DSA_D && dst->src[0]->ne[1] % DSA_HG == 0 && dst->src[7]->type == GGML_TYPE_I32 &&
        dims % 8 == 0 && offs % 8 == 0 && dims <= 64 && offs + dims <= DSA_D;
}

bool ggml_cuda_dsv4_sparse_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * kr = dst->src[1];
    const ggml_tensor * mr = dst->src[2];
    const ggml_tensor * kc = dst->src[3];
    const ggml_tensor * mc = dst->src[4];
    const ggml_tensor * tk = dst->src[5];
    const int64_t D = q->ne[0];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_GCN(cc) && (D == 512 || D == 256 || D == 128) && q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        q->ne[2] <= 8 && q->nb[0] == sizeof(float) && ggml_is_contiguous(dst) &&
        kr->type == GGML_TYPE_F16 && kc->type == GGML_TYPE_F16 && kr->nb[0] == sizeof(half) && kc->nb[0] == sizeof(half) &&
        kr->nb[2] % 4 == 0 && kc->nb[2] % 4 == 0 && ((uintptr_t) kr->data) % 4 == 0 && ((uintptr_t) kc->data) % 4 == 0 &&
        mr->type == GGML_TYPE_F16 && mc->type == GGML_TYPE_F16 && mr->nb[0] == sizeof(half) && mc->nb[0] == sizeof(half) &&
        tk->type == GGML_TYPE_I32 && tk->nb[0] == sizeof(int32_t) &&
        (dst->src[6] == nullptr || dst->src[6]->type == GGML_TYPE_F32) &&
        (dst->src[7] == nullptr || dsa_rope_ok(dst));
}

void ggml_cuda_dsv4_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * kr = dst->src[1];
    const ggml_tensor * mr = dst->src[2];
    const ggml_tensor * kc = dst->src[3];
    const ggml_tensor * mc = dst->src[4];
    const ggml_tensor * tk = dst->src[5];

    const int D      = (int) q->ne[0];
    const int n_head = (int) q->ne[1];
    const int nt     = (int) q->ne[2];
    const int n_raw  = (int) kr->ne[2];
    const int n_comp = (int) kc->ne[2];
    const int n_sel  = (int) tk->ne[0];
    const int n_pad  = (int) GGML_PAD(n_raw + nt*n_sel, 256);
    const int row_bytes = D*(int) sizeof(half);

    cudaStream_t stream = ctx.stream();

    // optional rope (ggml_dsv4_sparse_attn_set_rope): only the direct kernel implements it (supports_op checks)
    const ggml_tensor * rpos = dst->src[7];
    dsa_rope rp = {0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (rpos) {
        rp.dims = ggml_get_op_params_i32(dst, 1);
        rp.offs = ggml_get_op_params_i32(dst, 2);
        const int   n_ctx_orig = ggml_get_op_params_i32(dst, 3);
        const float freq_base  = ggml_get_op_params_f32(dst, 4);
        rp.freq_scale  = ggml_get_op_params_f32(dst, 5);
        rp.ext_factor  = ggml_get_op_params_f32(dst, 6);
        rp.attn_factor = ggml_get_op_params_f32(dst, 7);
        float corr[2];
        ggml_rope_yarn_corr_dims(rp.dims, n_ctx_orig, freq_base, ggml_get_op_params_f32(dst, 8), ggml_get_op_params_f32(dst, 9), corr);
        rp.corr0 = corr[0];
        rp.corr1 = corr[1];
        rp.theta_scale = powf(freq_base, -2.0f/rp.dims);
    }

    static const bool direct = [] { const char * e = getenv("GGML_CUDA_DSA_DIRECT"); return !e || atoi(e) != 0; }();
    // GGML_CUDA_DSA_CFG: waves per workgroup, cells per wave: 48, 416, 88 (default), 816. GGML_CUDA_DSA_CFG_VERIFY: the
    // same for 2+ query rows only. At 82 VGPRs one 8-wave workgroup fits a CU, so a 6-token DSpark verify (240 workgroups)
    // runs in ~4 rounds; 416's 4-wave workgroups fit three: DeepSeek V4 Flash pp6 171.5 -> 174.5, but the changed wave
    // sums moved greedy DSpark answers (2 of 4 prompts diverged early; sky 68 -> 49 t/s on a harder text), so opt-in
    static const int cfg_env = [] { const char * e = getenv("GGML_CUDA_DSA_CFG"); return e ? atoi(e) : 88; }();
    static const int cfg_verify = [] { const char * e = getenv("GGML_CUDA_DSA_CFG_VERIFY"); return e ? atoi(e) : 0; }();
    const int cfg = nt >= 2 && cfg_verify ? cfg_verify : cfg_env;
    const int nw  = cfg/100 ? cfg/100 : cfg/10;
    const int cpw = cfg/100 ? cfg%100 : cfg%10;
    const int cpb = nw*cpw;
    const int nch = (n_raw + n_sel + cpb - 1)/cpb;
    if (direct && D == DSA_D && n_head % DSA_HG == 0 && nch <= 64 && (int64_t) DSA_CSTRIDE*nt*(n_head/DSA_HG) <= 65535) {
        if (ctx.hc_mix_counters == nullptr) {
            CUDA_CHECK(cudaMalloc((void **) &ctx.hc_mix_counters, 65535*sizeof(int)));
            CUDA_CHECK(cudaMemsetAsync(ctx.hc_mix_counters, 0, 65535*sizeof(int), stream));
        }
        ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) nt*n_head*nch*DSA_PS);
        const float * sinks = dst->src[6] ? (const float *) dst->src[6]->data : nullptr;
#define DSA_LAUNCH(NW, CPW) \
        dsv4sa_attn<NW, CPW><<<dim3(n_head/DSA_HG, nch, nt), 64*NW, 0, stream>>>((const float *) q->data, \
            (const char *) kr->data, (const char *) kc->data, (const half *) mr->data, (const half *) mc->data, \
            (const int32_t *) tk->data, sinks, (float *) dst->data, part.get(), ctx.hc_mix_counters, n_raw, n_comp, n_sel, \
            n_head, q->nb[1], q->nb[2], kr->nb[2], kc->nb[2], mr->nb[1]/sizeof(half), mc->nb[1]/sizeof(half), \
            tk->nb[1]/sizeof(int32_t), dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), ggml_get_op_params_f32(dst, 0), \
            rpos ? (const int32_t *) rpos->data : nullptr, rp, spec, wcompact)
        // speculative compressed row loads when every top-k pick is a real row (opt-in GGML_CUDA_DSA_SPEC=1: tg d4K 84.6 vs
        // 84.95 without, the mask round trip is hidden behind the rope table and the query conversion)
        static const bool spec_env = [] { const char * e = getenv("GGML_CUDA_DSA_SPEC"); return e && atoi(e) != 0; }();
        const bool spec = spec_env && n_comp >= n_sel + 256;
        // compact the window to its valid cells (opt-in GGML_CUDA_DSA_WCOMPACT=1, up to DSA_WMAX slots; slower: d16K 79.0 vs
        // 79.9 - wave 0's mask scan + barrier doubles every workgroup's resolve phase and empty chunks still hold their slots)
        static const bool wc_env = [] { const char * e = getenv("GGML_CUDA_DSA_WCOMPACT"); return e && atoi(e) != 0; }();
        const bool wcompact = wc_env && n_raw <= DSA_WMAX;
        if (nw == 4 && cpw == 16) {
            DSA_LAUNCH(4, 16);
        } else if (nw == 4) {
            DSA_LAUNCH(4, 8);
        } else if (cpw == 16) {
            DSA_LAUNCH(8, 16);
        } else {
            DSA_LAUNCH(8, 8);
        }
#undef DSA_LAUNCH
        CUDA_CHECK(cudaGetLastError());
#if DSA_TRACE
        static int ntrace = 0;
        static const int want = [] { const char * e = getenv("GGML_CUDA_DSA_TRACE"); return e ? atoi(e) : 0; }();
        if (ntrace < want) {
            ++ntrace;
            CUDA_CHECK(cudaStreamSynchronize(stream));
            static unsigned long long h[4096][8];
            CUDA_CHECK(hipMemcpyFromSymbol(h, HIP_SYMBOL(dsa_trace), sizeof(h)));
            const int nwg = (n_head/DSA_HG)*nch*nt;
            unsigned long long t0 = ~0ull, tend = 0;
            for (int i = 0; i < nwg; ++i) { t0 = std::min(t0, h[i][0]); }
            std::vector<double> ph[7];
            for (int i = 0; i < nwg; ++i) {
                for (int k = 1; k <= 5; ++k) { ph[k].push_back((h[i][k] - h[i][k-1])*0.04); }
                ph[0].push_back((h[i][0] - t0)*0.04);
                if (h[i][6] > h[i][5] && h[i][6] - h[i][5] < 100000) { ph[6].push_back((h[i][6] - h[i][5])*0.04); tend = std::max(tend, h[i][6]); }
            }
            fprintf(stderr, "DSA trace: n_raw %d n_comp %d n_sel %d nt %d nwg %d, total %.2f us\n", n_raw, n_comp, n_sel, nt, nwg, (tend - t0)*0.04);
            const char * nm[7] = {"start", "resolve", "rows+QK", "scores+bar", "softmax+PV", "reduce+part+atomic", "merge"};
            for (int k = 0; k < 7; ++k) {
                std::vector<double> & v = ph[k];
                if (v.empty()) continue;
                std::sort(v.begin(), v.end());
                fprintf(stderr, "  %-20s min %6.2f med %6.2f max %6.2f us\n", nm[k], v[0], v[v.size()/2], v.back());
            }
        }
#endif
        return;
    }

    GGML_ASSERT(rpos == nullptr && "dsv4 sparse attention: the rope is only in the direct kernel");
    ggml_cuda_pool_alloc<char> kg(ctx.pool(), (size_t) n_pad*row_bytes);
    ggml_cuda_pool_alloc<half> mg(ctx.pool(), (size_t) nt*n_pad);
    ggml_cuda_pool_alloc<int>  kvm(ctx.pool(), 8);

    dsv4sa_gather<<<n_pad, 64, 0, stream>>>((const char *) kr->data, (const char *) kc->data, (const half *) mr->data,
        (const half *) mc->data, (const int32_t *) tk->data, kg.get(), mg.get(), n_raw, n_comp, n_sel, nt, n_pad, row_bytes,
        kr->nb[2], kc->nb[2], mr->nb[1]/sizeof(half), mc->nb[1]/sizeof(half), tk->nb[1]/sizeof(int32_t), kvm.get());
    CUDA_CHECK(cudaGetLastError());

    // flash attention view: Q [D, nt, n_head, 1], K = V [D, n_pad, 1, 1], mask [n_pad, nt, 1, 1], dst [D, n_head, nt, 1]
    ggml_tensor Q2 = *q, K2 = *kr, M2 = *mr, dst2 = *dst;
    Q2.ne[0] = D; Q2.ne[1] = nt; Q2.ne[2] = n_head; Q2.ne[3] = 1;
    Q2.nb[0] = sizeof(float); Q2.nb[1] = q->nb[2]; Q2.nb[2] = q->nb[1]; Q2.nb[3] = q->nb[2]*nt;
    Q2.view_src = nullptr; Q2.view_offs = 0;
    K2.data  = kg.get();
    K2.ne[0] = D; K2.ne[1] = n_pad; K2.ne[2] = 1; K2.ne[3] = 1;
    K2.nb[0] = sizeof(half); K2.nb[1] = row_bytes; K2.nb[2] = (size_t) n_pad*row_bytes; K2.nb[3] = K2.nb[2];
    K2.view_src = nullptr; K2.view_offs = 0;
    M2.data  = mg.get();
    M2.ne[0] = n_pad; M2.ne[1] = nt; M2.ne[2] = 1; M2.ne[3] = 1;
    M2.nb[0] = sizeof(half); M2.nb[1] = (size_t) n_pad*sizeof(half); M2.nb[2] = M2.nb[1]*nt; M2.nb[3] = M2.nb[2];
    M2.view_src = nullptr; M2.view_offs = 0;
    dst2.op = GGML_OP_FLASH_ATTN_EXT;
    dst2.ne[0] = D; dst2.ne[1] = n_head; dst2.ne[2] = nt; dst2.ne[3] = 1;
    memset(dst2.op_params, 0, sizeof(dst2.op_params));
    ggml_set_op_params_f32(&dst2, 0, ggml_get_op_params_f32(dst, 0)); // scale
    ggml_set_op_params_f32(&dst2, 1, 0.0f);                          // max_bias
    ggml_set_op_params_f32(&dst2, 2, 0.0f);                          // logit_softcap
    ggml_set_op_params_i32(&dst2, 3, GGML_PREC_F32);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        dst2.src[i] = nullptr;
    }
    dst2.src[0] = &Q2;
    dst2.src[1] = &K2;
    dst2.src[2] = &K2;
    dst2.src[3] = &M2;
    dst2.src[4] = dst->src[6];

    ggml_cuda_fattn_kv_max_override = kvm.get();
    ggml_cuda_flash_attn_ext_tile(ctx, &dst2);
    ggml_cuda_fattn_kv_max_override = nullptr;
}
