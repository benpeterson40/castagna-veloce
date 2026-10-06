#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi(const T *            x,
                                  T *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope,
                                  const bool           inplace,
                                  const int            nr) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    // theta_scale^(iw/2) depends on the dimension only: computed once per block column (the rows packed into a block
    // share it) instead of a powf per element, which dominated this kernel for many short rows (pooled indexer keys)
    __shared__ float s_pw[CUDA_ROPE_BLOCK_SIZE];
    if (threadIdx.x == 0) {
        s_pw[threadIdx.y] = powf(theta_scale, max(i0 - n_offs, 0) / 2.0f);
    }
    __syncthreads();

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;
    if (row_dst >= nr) {
        return; // several rows per block (short rows): the last block may be partial
    }

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];

        return;
    }

    const int iw = i0 - n_offs; // relative idx
    const float pw = s_pw[threadIdx.y];

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * pw;
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * pw;
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * pw;
        } else {
            theta_base = pos[i2 + ne02 * 3] * pw;
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * pw;
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * pw;
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * pw;
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * pw;
        }
    }

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_offs/2 + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T>
static void rope_multi_cuda(const T *            x,
                            T *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            const bool           inplace,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    dim3 block_nums(nr, n_blocks_x, 1);
    // short rows (e.g. 128-wide pooled indexer keys, one head, thousands of rows): pack several rows per block,
    // a 256-thread block per 128-value row left three quarters of every block idle
    if (2 * CUDA_ROPE_BLOCK_SIZE >= 2 * ne00) {
        const int per_row = ne00 / 2;
        const int rows    = CUDA_ROPE_BLOCK_SIZE / per_row;
        if (rows > 1) {
            block_dims = dim3(rows, per_row, 1);
            n_blocks_x = 1;
            block_nums = dim3((nr + rows - 1) / rows, 1, 1);
        }
    }

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace, nr);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace, nr);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    mrope_sections sections;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = dst_d == src0->data;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
template <int block_size, bool has_ff, typename D>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const int n_offs) {
    ggml_cuda_pdl_lc();
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x += sample*s03 + channel*s02 + row*s01;

    const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
    mul += mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float scale = rsqrtf(tmp/ncols + eps);

    int64_t idst = sample*s3 + channel*s2 + row*s1;
    if (set_rows_stride != 0) {
        idst = row*s1 + row_indices[channel]*set_rows_stride;
    }
    dst += idst;

    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
        int ix0;
        int ix1;
        if (is_neox && i0 < n_dims) {
            ix0 = i0/2;
            ix1 = i0/2 + n_dims/2;
        } else {
            ix0 = i0 + 0;
            ix1 = i0 + 1;
        }

        const float x0 = scale * x[ix0] * mul[fastmodulo(ix0, mul_ncols_packed)];
        const float x1 = scale * x[ix1] * mul[fastmodulo(ix1, mul_ncols_packed)];

        // ggml_rope_set_offset (norm mode only): the rotated pairs are [n_offs, n_offs + n_dims)
        if (i0 < n_offs || i0 >= n_offs + n_dims) {
            dst[ix0] = ggml_cuda_cast<D>(x0);
            dst[ix1] = ggml_cuda_cast<D>(x1);
            continue;
        }
        const int iw = i0 - n_offs; // relative idx

        const float theta_base  = pos[channel]*powf(theta_scale, iw/2.0f);
        const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

        dst[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
        dst[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
    }
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const int n_offs, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);
    GGML_ASSERT(n_offs == 0 || !is_neox);

    const dim3 blocks_num(nrows, nchannels, nsamples);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, n_offs);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, n_offs);
        }
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, n_offs);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, n_offs);
        }
    }
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->type == GGML_TYPE_F32);

    void *          dst_d           = rope->data;
    ggml_type       dst_type        = rope->type;
    const int64_t * row_indices     = nullptr;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        dst_d           = set_rows->data;
        dst_type        = set_rows->type;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const int  n_offs  = ((const int32_t *) rope->op_params)[15];

    const int32_t * pos = (const int32_t *) rope->src[1]->data;

    const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_dst = ggml_type_size(rope->type);
    const int64_t s1 = rope->nb[1] / ts_dst;
    const int64_t s2 = rope->nb[2] / ts_dst;
    const int64_t s3 = rope->nb[3] / ts_dst;

    cudaStream_t stream = ctx.stream();

    if (dst_type == GGML_TYPE_F32) {
        rms_norm_mul_rope_cuda((const float *) x->data, (float *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, n_offs, stream);
    } else if (dst_type == GGML_TYPE_F16) {
        rms_norm_mul_rope_cuda((const float *) x->data, (half *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, n_offs, stream);
    } else {
        GGML_ABORT("fatal error");
    }
}

// ---- decode attention q/k chain in one kernel (GGML_CUDA_ATTN_QK_FUSE=0 off) -------------------------------------------
// RMS_NORM(q) -> MUL(w_q) -> ROPE(multi) ; RMS_NORM(k) -> MUL(w_k) -> ROPE -> SET_ROWS(k cache) ; SET_ROWS(v cache):
// one block per (token, head), one thread per channel (head dim 256). The weight and the rotation are the same float
// operations as the separate kernels (rms_norm: scale*x then *w; rope_multi: powf pw, sector position, rope_yarn); only
// the norm's sum order differs. K and V are converted to f16 into their cache rows.
struct attn_qk_args {
    const float * q; int64_t q_s1, q_s2;          // q view: head / token strides (floats)
    const float * k; int64_t k_s1, k_s2;
    const float * v; int64_t v_s1, v_s2;
    const float * wq; const float * wk;
    float eps_q, eps_k;
    float * q_out;                                // [hd, nqh, T] contiguous
    half * ck; int64_t ck_row; const int64_t * k_idx;
    half * cv; int64_t cv_row; const int64_t * v_idx;
    const int32_t * pos;
    int n_dims, n_offs, T, nqh, nkvh;
    float theta_scale, freq_scale, ext_factor, attn_factor;
    rope_corr_dims corr;
    mrope_sections sec;
    bool imrope;
};

template <int HD>
static __global__ void __launch_bounds__(HD) attn_qk_fused(const attn_qk_args a) {
    const int nh = a.nqh + a.nkvh;
    const int t = blockIdx.x / nh, h = blockIdx.x % nh, c = threadIdx.x;
    const bool isq = h < a.nqh;
    const int hk = h - a.nqh;
    const float x = isq ? a.q[t*a.q_s2 + h*a.q_s1 + c] : a.k[t*a.k_s2 + hk*a.k_s1 + c];
    __shared__ float red[HD/WARP_SIZE];
    __shared__ float row[HD];
    float ss = warp_reduce_sum(x*x);
    if (c % WARP_SIZE == 0) { red[c / WARP_SIZE] = ss; }
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int w = 0; w < HD/WARP_SIZE; ++w) { ss += red[w]; }
    const float scale = rsqrtf(ss / HD + (isq ? a.eps_q : a.eps_k));
    row[c] = (scale*x)*(isq ? a.wq[c] : a.wk[c]);
    __syncthreads();
    float out = row[c];
    const int rel = c - a.n_offs;
    if (rel >= 0 && rel < a.n_dims) {
        const int half_dims = a.n_dims/2;
        const int j = rel % half_dims;          // pair index; iw = 2*j
        const bool first = rel < half_dims;
        const float x0 = row[a.n_offs + j], x1 = row[a.n_offs + j + half_dims];
        const float pw = powf(a.theta_scale, j / 1.0f);
        const int sect_dims = a.sec.v[0] + a.sec.v[1] + a.sec.v[2] + a.sec.v[3];
        const int sec_w = a.sec.v[1] + a.sec.v[0];
        const int sector = j % sect_dims;
        const int T = a.T;
        float theta_base;
        if (a.imrope) {
            if (sector % 3 == 1 && sector < 3*a.sec.v[1]) {
                theta_base = a.pos[t + T*1]*pw;
            } else if (sector % 3 == 2 && sector < 3*a.sec.v[2]) {
                theta_base = a.pos[t + T*2]*pw;
            } else if (sector % 3 == 0 && sector < 3*a.sec.v[0]) {
                theta_base = a.pos[t]*pw;
            } else {
                theta_base = a.pos[t + T*3]*pw;
            }
        } else {
            if (sector < a.sec.v[0]) {
                theta_base = a.pos[t]*pw;
            } else if (sector < sec_w) {
                theta_base = a.pos[t + T*1]*pw;
            } else if (sector < sec_w + a.sec.v[2]) {
                theta_base = a.pos[t + T*2]*pw;
            } else {
                theta_base = a.pos[t + T*3]*pw;
            }
        }
        float cos_theta, sin_theta;
        rope_yarn<true>(theta_base/1.0f, a.freq_scale, a.corr, 2*j, a.ext_factor, a.attn_factor, cos_theta, sin_theta);
        out = first ? x0*cos_theta - x1*sin_theta : x0*sin_theta + x1*cos_theta;
    }
    if (isq) {
        a.q_out[((int64_t) t*a.nqh + h)*HD + c] = out;
    } else {
        a.ck[a.k_idx[t]*a.ck_row + hk*HD + c] = __float2half(out);
        a.cv[a.v_idx[t]*a.cv_row + hk*HD + c] = __float2half(a.v[t*a.v_s2 + hk*a.v_s1 + c]);
    }
}

static bool attn_qk_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
           t->op == GGML_OP_TRANSPOSE;
}
static const ggml_tensor * attn_qk_base(const ggml_tensor * t) {
    while (t && attn_qk_is_view(t) && t->src[0]) { t = t->src[0]; }
    return t;
}

int ggml_cuda_attn_qk_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
                            bool (*compute_middle)(ggml_backend_cuda_context & ctx, ggml_tensor * t)) {
    static const bool off = [] { const char * e = getenv("GGML_CUDA_ATTN_QK_FUSE"); return e && atoi(e) == 0; }();
    if (off) {
        return 0;
    }
    // the next 8 compute nodes (views skipped): RMS_NORM MUL ROPE RMS_NORM MUL ROPE SET_ROWS SET_ROWS; the k / v
    // projections (MUL_MAT of the layer input) may sit between the q rope and the k norm: they are computed first
    int idx[8];
    int middle[4];
    int n = 0, nm = 0, j = i;
    for (; j < cgraph->n_nodes && n < 8; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (attn_qk_is_view(t)) {
            continue;
        }
        if (n == 3 && t->op == GGML_OP_MUL_MAT && nm < 4) {
            middle[nm++] = j;
            continue;
        }
        idx[n++] = j;
    }
    if (n < 8) {
        return 0;
    }
    ggml_tensor * const * nd = cgraph->nodes;
    const ggml_tensor * rq = nd[idx[0]], * mq = nd[idx[1]], * pq = nd[idx[2]], * rk = nd[idx[3]], * mk = nd[idx[4]],
                      * pk = nd[idx[5]], * sk = nd[idx[6]], * sv = nd[idx[7]];
    const ggml_op ops[8] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE,
                             GGML_OP_SET_ROWS, GGML_OP_SET_ROWS };
    for (int q = 0; q < 8; ++q) {
        if (nd[idx[q]]->op != ops[q]) {
            return 0;
        }
    }
    const auto other = [](const ggml_tensor * m, const ggml_tensor * x) { return m->src[0] == x ? m->src[1] : m->src[0]; };
    const ggml_tensor * q = rq->src[0], * k = rk->src[0], * wq = other(mq, rq), * wk = other(mk, rk);
    const int HD = 256;
    const int64_t T = q->ne[2];
    const bool ok =
        (mq->src[0] == rq || mq->src[1] == rq) && (mk->src[0] == rk || mk->src[1] == rk) &&
        pq->src[0] == mq && pk->src[0] == mk && pq->src[2] == nullptr && pk->src[2] == nullptr &&
        q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && q->ne[0] == HD && k->ne[0] == HD && q->nb[0] == sizeof(float) &&
        k->nb[0] == sizeof(float) && k->ne[2] == T && T >= 1 && T <= 16 && q->ne[3] == 1 && k->ne[3] == 1 &&
        wq->type == GGML_TYPE_F32 && wk->type == GGML_TYPE_F32 && ggml_nelements(wq) == HD && ggml_nelements(wk) == HD &&
        ggml_is_contiguous(wq) && ggml_is_contiguous(wk) &&
        pq->type == GGML_TYPE_F32 && ggml_is_contiguous(pq) && ggml_are_same_shape(pq, q) &&
        pq->src[1] == pk->src[1] && pq->src[1]->type == GGML_TYPE_I32 && pq->src[1]->ne[0] == 4*T &&
        memcmp(pq->op_params, pk->op_params, sizeof(int32_t)*16) == 0 &&
        // K: rope output (through views) into the k cache rows; V rows into the v cache
        // (SET_ROWS: src[0] the rows, src[1] the indices, the node itself the view of the cache it writes)
        attn_qk_base(sk->src[0]) == pk && sk->type == GGML_TYPE_F16 && sk->src[1]->type == GGML_TYPE_I64 &&
        sv->type == GGML_TYPE_F16 && sv->src[1]->type == GGML_TYPE_I64 && sv->src[0]->type == GGML_TYPE_F32 &&
        sk->nb[0] == sizeof(half) && sv->nb[0] == sizeof(half) &&
        sk->src[0]->ne[0] == k->ne[0]*k->ne[1] && sv->src[0]->ne[0] == k->ne[0]*k->ne[1] &&
        ggml_is_contiguous(sk->src[0]) && ggml_is_contiguous(sv->src[0]) &&
        ggml_nelements(sk->src[1]) == T && ggml_nelements(sv->src[1]) == T &&
        sk->ne[0] == k->ne[0]*k->ne[1] && sv->ne[0] == k->ne[0]*k->ne[1] &&
        // intermediates read only inside the chain
        ggml_node_get_use_count(cgraph, idx[0]) == 1 && ggml_node_get_use_count(cgraph, idx[1]) == 1 &&
        ggml_node_get_use_count(cgraph, idx[3]) == 1 && ggml_node_get_use_count(cgraph, idx[4]) == 1 &&
        !(rq->flags & GGML_TENSOR_FLAG_OUTPUT) && !(mq->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        !(rk->flags & GGML_TENSOR_FLAG_OUTPUT) && !(mk->flags & GGML_TENSOR_FLAG_OUTPUT) && !(pk->flags & GGML_TENSOR_FLAG_OUTPUT);
    if (!ok) {
        return 0;
    }
    const int mode = ((const int32_t *) pq->op_params)[2];
    if (!(mode & GGML_ROPE_TYPE_MROPE) || mode == GGML_ROPE_TYPE_VISION) {
        return 0;
    }
    // K rope output is read only by its SET_ROWS (through views)
    int users = 0;
    for (int u = idx[5] + 1; u < cgraph->n_nodes; ++u) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = cgraph->nodes[u]->src[s];
            if (src && attn_qk_base(src) == pk && cgraph->nodes[u] != sk && !attn_qk_is_view(cgraph->nodes[u])) {
                users++;
            }
        }
    }
    if (users > 0) {
        return 0;
    }
    attn_qk_args a = {};
    a.q = (const float *) q->data; a.q_s1 = q->nb[1]/sizeof(float); a.q_s2 = q->nb[2]/sizeof(float);
    a.k = (const float *) k->data; a.k_s1 = k->nb[1]/sizeof(float); a.k_s2 = k->nb[2]/sizeof(float);
    const ggml_tensor * v = sv->src[0];
    a.v = (const float *) v->data; a.v_s1 = HD; a.v_s2 = v->nb[1]/sizeof(float);
    a.wq = (const float *) wq->data; a.wk = (const float *) wk->data;
    memcpy(&a.eps_q, rq->op_params, sizeof(float));
    memcpy(&a.eps_k, rk->op_params, sizeof(float));
    a.q_out = (float *) pq->data;
    a.ck = (half *) sk->data; a.ck_row = sk->nb[1]/sizeof(half); a.k_idx = (const int64_t *) sk->src[1]->data;
    a.cv = (half *) sv->data; a.cv_row = sv->nb[1]/sizeof(half); a.v_idx = (const int64_t *) sv->src[1]->data;
    a.pos = (const int32_t *) pq->src[1]->data;
    a.T = (int) T; a.nqh = (int) q->ne[1]; a.nkvh = (int) k->ne[1];
    const int32_t * op = (const int32_t *) pq->op_params;
    a.n_dims = op[1];
    a.n_offs = op[15];
    const int n_ctx_orig = op[4];
    float freq_base, beta_fast, beta_slow;
    memcpy(&freq_base,     op + 5, sizeof(float));
    memcpy(&a.freq_scale,  op + 6, sizeof(float));
    memcpy(&a.ext_factor,  op + 7, sizeof(float));
    memcpy(&a.attn_factor, op + 8, sizeof(float));
    memcpy(&beta_fast,     op + 9, sizeof(float));
    memcpy(&beta_slow,     op + 10, sizeof(float));
    memcpy(&a.sec.v,       op + 11, sizeof(int)*4);
    a.imrope = mode == GGML_ROPE_TYPE_IMROPE;
    a.theta_scale = powf(freq_base, -2.0f/a.n_dims);
    ggml_rope_yarn_corr_dims(a.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, a.corr.v);
    if (a.n_offs < 0 || a.n_offs + a.n_dims > HD || a.n_dims % 2 != 0) {
        return 0;
    }
    // the middle projections read the layer input only (never the chain's nodes)
    for (int m = 0; m < nm; ++m) {
        const ggml_tensor * t = cgraph->nodes[middle[m]];
        for (int s2 = 0; s2 < GGML_MAX_SRC; ++s2) {
            for (int q2 = 0; q2 < 3; ++q2) {
                if (t->src[s2] && attn_qk_base(t->src[s2]) == nd[idx[q2]]) {
                    return 0;
                }
            }
        }
    }
    // the skipped nodes free q's buffer early, so the allocator may have put the rope output or a middle projection over it
    const auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * x0 = (const char *) x->data, * y0 = (const char *) y->data;
        return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
    };
    if (overlap(pq, q) || overlap(pq, k) || overlap(pq, v)) {
        return 0;
    }
    for (int m = 0; m < nm; ++m) {
        if (overlap(cgraph->nodes[middle[m]], q)) {
            return 0;
        }
    }
    for (int m = 0; m < nm; ++m) {
        if (!compute_middle(ctx, cgraph->nodes[middle[m]])) {
            return 0;
        }
    }
    attn_qk_fused<256><<<(unsigned) (T*(a.nqh + a.nkvh)), 256, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    return idx[7] - i;
}
