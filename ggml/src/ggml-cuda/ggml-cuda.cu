#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include <chrono>
#include <deque>
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/moe-f16.cuh"
#include "ggml-cuda/moe-vec.cuh"
#include "ggml-cuda/hc-dec.cuh"
#include "ggml-cuda/gcn-q8-gemm.cuh"
#include "ggml-cuda/gcn-q8-matvec.cuh"
#include "ggml-cuda/gcn-f16-matvec.cuh"
#include "ggml-cuda/gcn-kq-matvec.cuh"
#include "ggml-cuda/mm-tile-f.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/hc-persist.cuh"
#include "ggml-cuda/fattn-qsa-gcn.cuh"
#include "ggml-cuda/fattn-sparse-gcn.cuh"
#include "ggml-cuda/dsv4-comp-pool.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <set>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
#if defined(GGML_USE_HIP)
    // ROCr's default kernel-argument pool fills when a device has a few thousand launches queued (a pipelined prompt
    // queues several ubatches of ~1700 kernels per device); every launch then waits for the GPU to recycle kernargs
    // and blocks the one thread that feeds all devices. 64 MiB removes those stalls (4x MI50: pp2048 +4-7%).
    // Must be set before the HSA runtime starts; a user setting wins.
    setenv("HSA_KERNARG_POOL_SIZE", "67108864", 0);
    // more hardware queues per GPU: with several virtual devices per GPU (see GGML_CUDA_VIRTUAL_PER_GPU) the default 4
    // made their streams share queues and serialize (4x MI50, 3 per GPU: ~1400 -> ~2540 t/s pp2048)
    setenv("GPU_MAX_HW_QUEUES", "8", 0);
#endif
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    // GCN (gfx9, e.g. MI50) multi-GPU: several virtual devices per GPU by default (3, or 2 above 5 GPUs). With the
    // default layer split the pipeline then visits every GPU repeatedly (interleaved stages, 0 1 2 3 0 1 2 3 ...), which
    // shrinks its fill/drain bubble for a prompt; the streams of a GPU overlap (4x MI50 pp2048: 2094 -> ~2360 t/s with
    // 2 per GPU, ~2540 with 3; decode unchanged).
    // GGML_CUDA_VIRTUAL_PER_GPU=N overrides (1 = off); GGML_CUDA_DEVICES takes precedence. Limited to 7 GPUs so the
    // scheduler stays within GGML_SCHED_MAX_BACKENDS (16, incl. the CPU).
    {
        const char * per_env = getenv("GGML_CUDA_VIRTUAL_PER_GPU");
        int per = per_env ? atoi(per_env) : -1;
        if (per < 0) {
            per = 1;
            bool all_gcn = info.physical_device_count >= 2 && info.physical_device_count <= 7;
            // 3 per GPU when the scheduler has room (3*n + CPU <= 16 backends), else 2
            for (int id = 0; all_gcn && id < info.physical_device_count; ++id) {
                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
#if defined(GGML_USE_HIP)
                const char * a = prop.gcnArchName;
                all_gcn = all_gcn && (strncmp(a, "gfx900", 6) == 0 || strncmp(a, "gfx906", 6) == 0 ||
                                      strncmp(a, "gfx909", 6) == 0 || strncmp(a, "gfx90c", 6) == 0);
#else
                all_gcn = false;
#endif
            }
            if (all_gcn) {
                per = 3*info.physical_device_count + 1 <= 16 ? 3 : 2;
            }
        }
        if (per > 1 && getenv("GGML_CUDA_DEVICES") == nullptr) {
            info.device_count = std::min(per*info.physical_device_count, GGML_CUDA_MAX_DEVICES);
        }
    }

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES / GGML_CUDA_VIRTUAL_PER_GPU)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    q8_cache.clear(); // its buffers belong to the pools destroyed below
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (hc_mix_counters != nullptr) {
        CUDA_CHECK(cudaFree(hc_mix_counters));
    }
    if (hc_step_gen != nullptr) {
        (void) cudaFree(hc_step_gen);
    }
    if (moe1_counters != nullptr) {
        (void) cudaFree(moe1_counters);
    }
    if (gemv1_epi_counter != nullptr) {
        (void) cudaFree(gemv1_epi_counter);
    }
    if (router1_counter != nullptr) {
        CUDA_CHECK(cudaFree(router1_counter));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// Blocking host <-> device copies of pageable host memory, staged through a pinned double buffer. On ROCm the
// runtime copies large pageable ranges by locking the user pages in place; restoring a sequence state
// (llama_state_seq_set_data: one host buffer whose consecutive regions go to different GPUs, which with
// GGML_CUDA_VIRTUAL_PER_GPU happens on most layer boundaries) faulted on gfx906 with illegal memory accesses at
// host addresses (llama-bench -d 8192 -r 2: 4 of 4 runs). Pinned or device-visible sources are copied directly.
// GGML_CUDA_STAGED_HOST_COPY=0 disables.
static bool ggml_cuda_host_ptr_is_pageable(const void * ptr) {
    cudaPointerAttributes attr;
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        (void) cudaGetLastError(); // unregistered host memory reports an error on some runtimes
        return true;
    }
    return attr.type == cudaMemoryTypeUnregistered;
}

static bool ggml_cuda_host_copy_staged(void * dst, const void * src, size_t size, cudaMemcpyKind kind) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_STAGED_HOST_COPY"); return !e || atoi(e) != 0; }();
    constexpr size_t CHUNK = 8u << 20;
    if (!enabled || size < (1u << 20) || !ggml_cuda_host_ptr_is_pageable(kind == cudaMemcpyHostToDevice ? src : dst)) {
        return false;
    }
    // one staging pair per thread and physical device, kept for the process lifetime (never freed: thread_local
    // destructors can run after the runtime is torn down)
    struct staging {
        char *      buf[2] = {};
        cudaEvent_t ev[2]  = {};
    };
    thread_local staging st[GGML_CUDA_MAX_DEVICES];
    const int dev = ggml_cuda_get_device();
    staging & s = st[dev];
    if (s.buf[0] == nullptr) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaHostAlloc((void **) &s.buf[i], CHUNK, cudaHostAllocPortable));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.ev[i], cudaEventDisableTiming));
        }
    }
    bool used[2] = {false, false};
    for (size_t off = 0, i = 0; off < size; off += CHUNK, ++i) {
        const size_t n = std::min(CHUNK, size - off);
        const int    h = i % 2;
        if (used[h]) {
            CUDA_CHECK(cudaEventSynchronize(s.ev[h]));
            if (kind == cudaMemcpyDeviceToHost) {
                memcpy((char *) dst + (off - 2*CHUNK), s.buf[h], CHUNK);
            }
        }
        if (kind == cudaMemcpyHostToDevice) {
            memcpy(s.buf[h], (const char *) src + off, n);
            CUDA_CHECK(cudaMemcpyAsync((char *) dst + off, s.buf[h], n, kind, cudaStreamPerThread));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(s.buf[h], (const char *) src + off, n, kind, cudaStreamPerThread));
        }
        CUDA_CHECK(cudaEventRecord(s.ev[h], cudaStreamPerThread));
        used[h] = true;
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    if (kind == cudaMemcpyDeviceToHost) {
        // the last one or two chunks are still in the staging buffers
        const size_t nchunks = (size + CHUNK - 1) / CHUNK;
        for (size_t i = nchunks >= 2 ? nchunks - 2 : 0; i < nchunks; ++i) {
            const size_t off = i*CHUNK;
            memcpy((char *) dst + off, s.buf[i % 2], std::min(CHUNK, size - off));
        }
    }
    return true;
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_host_copy_staged((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice)) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_host_copy_staged(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost)) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// 2D host -> device copy of pageable memory through the per-thread pinned double buffer: rows are packed into the
// staging buffer and sent with one cudaMemcpy2DAsync per chunk. Tensor-split (meta) uploads are 2D copies of row
// slices; straight from pageable memory they ran at ~5.7 GB/s per GPU and did not scale across GPUs, staged per
// uploader thread ~6.2 GB/s per GPU and 12.4 GB/s for 3 GPUs in parallel (MI50, tmpfs source)
static bool ggml_cuda_host_copy_staged_2d(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_STAGED_HOST_COPY"); return !e || atoi(e) != 0; }();
    constexpr size_t CHUNK = 8u << 20;
    if (!enabled || width == 0 || height == 0 || width > CHUNK || width*height < (1u << 20) || !ggml_cuda_host_ptr_is_pageable(src)) {
        return false;
    }
    struct staging {
        char *      buf[2]  = {};
        cudaEvent_t ev[2]   = {};
        char *      dbuf[2] = {}; // device copies of the packed chunks (pitched destinations)
    };
    thread_local staging st[GGML_CUDA_MAX_DEVICES];
    const int dev = ggml_cuda_get_device();
    staging & s = st[dev];
    if (s.buf[0] == nullptr) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaHostAlloc((void **) &s.buf[i], CHUNK, cudaHostAllocPortable));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.ev[i], cudaEventDisableTiming));
        }
    }
    // a pitched host -> device cudaMemcpy2DAsync runs one DMA per row on ROCm (tensor-split DeepSeek V4.1: ~2M rows of
    // ~500 B per expert down tensor, minutes per GPU): send the packed chunk contiguously and let a device -> device 2D
    // copy (a blit kernel) scatter the rows
    const bool pitched = dpitch != width;
    if (pitched && s.dbuf[0] == nullptr) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaMalloc((void **) &s.dbuf[i], CHUNK));
        }
    }
    const size_t rows_per_chunk = CHUNK / width;
    bool used[2] = {false, false};
    for (size_t r0 = 0, i = 0; r0 < height; r0 += rows_per_chunk, ++i) {
        const size_t nr = std::min(rows_per_chunk, height - r0);
        const int    h  = i % 2;
        if (used[h]) {
            CUDA_CHECK(cudaEventSynchronize(s.ev[h]));
        }
        const char * sp = (const char *) src + r0*spitch;
        if (spitch == width) {
            memcpy(s.buf[h], sp, nr*width);
        } else {
            for (size_t r = 0; r < nr; ++r) {
                memcpy(s.buf[h] + r*width, sp + r*spitch, width);
            }
        }
        if (pitched) {
            CUDA_CHECK(cudaMemcpyAsync(s.dbuf[h], s.buf[h], nr*width, cudaMemcpyHostToDevice, cudaStreamPerThread));
            CUDA_CHECK(cudaMemcpy2DAsync((char *) dst + r0*dpitch, dpitch, s.dbuf[h], width, width, nr,
                cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
            CUDA_CHECK(cudaMemcpyAsync((char *) dst + r0*dpitch, s.buf[h], nr*width, cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
        CUDA_CHECK(cudaEventRecord(s.ev[h], cudaStreamPerThread));
        used[h] = true;
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_host_copy_staged_2d((char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies)) {
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

    // n-way (n > 2) peer all-reduce (ggml_cuda_arn_allreduce): 0 untested, 1 ready, -1 unavailable
    int                                        arn_state = 0;
    std::vector<std::array<cudaEvent_t, 3>>    arn_ev;
    // small tensors: push AllReduce through uncached inboxes + flags in each GPU's VRAM
    std::vector<float *>                       arn_inbox; // per device: [2 halves][n senders][ARN_PUSH_MAX_ELEMS]
    std::vector<int *>                         arn_flag;  // per device: [n senders][ARN_PUSH_MAX_BLOCKS] x 64 B
    int                                        arn_token = 0;
    int                                        arn_push_state = 0; // 0 untested, 1 ready, -1 unavailable
    // large tensors, 16-bit transfer format: per device staging for the packed partial / packed shard sum
    std::vector<uint16_t *>                    arn_stage;
    std::vector<uint16_t *>                    arn_rx;    // GGML_CUDA_ARN_DMA: the peers' packed shards (reduce input)
    size_t                                     arn_stage_elems = 0;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
        for (size_t i = 0; i < arn_ev.size(); ++i) {
            ggml_cuda_set_device(dev_ids[i]);
            for (cudaEvent_t e : arn_ev[i]) {
                if (e != nullptr) {
                    (void) cudaEventDestroy(e);
                }
            }
        }
        for (size_t i = 0; i < arn_inbox.size(); ++i) {
            ggml_cuda_set_device(dev_ids[i]);
            if (arn_inbox[i] != nullptr) {
                (void) cudaFree(arn_inbox[i]);
            }
            if (arn_flag[i] != nullptr) {
                (void) cudaFree(arn_flag[i]);
            }
        }
        for (size_t i = 0; i < arn_stage.size(); ++i) {
            ggml_cuda_set_device(dev_ids[i]);
            if (arn_stage[i] != nullptr) {
                (void) cudaFree(arn_stage[i]);
            }
        }
        for (size_t i = 0; i < arn_rx.size(); ++i) {
            ggml_cuda_set_device(dev_ids[i]);
            if (arn_rx[i] != nullptr) {
                (void) cudaFree(arn_rx[i]);
            }
        }
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
// n-way (n > 2) direct peer all-reduce of large F32 tensors (-sm tensor groups of 4, e.g. DeepSeek V4.1 prefill: 10 MB per
// AllReduce at 512 tokens). The internal pipeline (allreduce.cu) handles pairs only; the meta backend's butterfly moves
// log2(n) whole tensors per GPU in dependent steps. Here:
//   reduce-scatter: GPU i writes shard i of the sum of all n partials (its own plus peer reads over PCIe) into its tensor,
//   all-gather:     GPU i copies shard j of GPU j's tensor for every j != i,
// with cross-device event waits before each phase and after the all-gather (GPU i may not overwrite shard i before its
// peers read it). Per GPU 2*(n-1)/n tensors cross its link, all links busy in both phases. Every shard is summed in
// device order (deterministic). A tensor without GGML_TENSOR_FLAG_COMPUTE (zero-sized slice) contributes zero.
// GGML_CUDA_ARN=0 off, GGML_CUDA_ARN_MIN_BYTES (default 1 MiB): smaller tensors keep the butterfly.
struct ggml_cuda_arn_ptrs {
    const float * p[GGML_CUDA_MAX_DEVICES];
};

static __global__ void ggml_cuda_arn_reduce_f32(float * __restrict__ dst, const ggml_cuda_arn_ptrs src, const int n,
        const int64_t off, const int64_t len) {
    const int64_t n4 = len / 4;
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    for (int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < n4; k += stride) {
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = 0; j < n; ++j) {
            if (src.p[j] == nullptr) {
                continue;
            }
            const float4 b = ((const float4 *) (src.p[j] + off))[k];
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        }
        ((float4 *) (dst + off))[k] = a;
    }
    for (int64_t k = 4*n4 + (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < len; k += stride) {
        float a = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (src.p[j] != nullptr) {
                a += src.p[j][off + k];
            }
        }
        dst[off + k] = a;
    }
}

// dst[j*shard .. min(ne, (j+1)*shard)) = src.p[j][same] for every j != self
static __global__ void ggml_cuda_arn_gather_f32(float * __restrict__ dst, const ggml_cuda_arn_ptrs src, const int n,
        const int self, const int64_t ne, const int64_t shard) {
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    for (int j = 0; j < n; ++j) {
        if (j == self) {
            continue;
        }
        const int64_t off = j*shard;
        const int64_t len = shard < ne - off ? shard : ne - off;
        if (len <= 0) {
            continue;
        }
        const int64_t n4 = len / 4;
        for (int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < n4; k += stride) {
            ((float4 *) (dst + off))[k] = ((const float4 *) (src.p[j] + off))[k];
        }
        for (int64_t k = 4*n4 + (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < len; k += stride) {
            dst[off + k] = src.p[j][off + k];
        }
    }
}

// Small tensors (decode: 5120 floats per token): the event-ordered phases above cost ~0.4 ms per AllReduce on 4 GPUs
// (as did the butterfly's chain of small copy/add graphs), about half of V4.1 decode. One kernel per GPU instead: rank r
// writes its partial into slot r of every peer's inbox (uncached VRAM: incoming peer writes bypass the receiver's L2), then
// sets a per-block flag in every peer's flag array; it polls its own flags until all peers have written the same token,
// and sums the n contributions in device order (own data at position r), so every GPU gets bitwise the same result.
// Inbox halves alternate with the token parity: a GPU can be at most one AllReduce ahead of a peer (finishing AR t needs
// the peer's flag for t, which the peer sets only after finishing AR t-1), so half (t+1)%2 is never still being read.
#include "ktrace.cuh"
#define ARN_PUSH_MAX_ELEMS  (64*1024)
#define ARN_PUSH_MAX_BLOCKS 32
#define ARN_FLAG_INTS       16 // 64 B per flag

struct ggml_cuda_arn_push_ptrs {
    float * inbox[GGML_CUDA_MAX_DEVICES]; // each device's inbox half for this token
    int   * flag[GGML_CUDA_MAX_DEVICES];
};

// transfer format FMT (GGML_CUDA_ARN_PUSH_FMT): 0 f32, 1 f16, 2 bf16 (round to nearest even). With a 16-bit format every
// GPU also rounds its own partial before the sum, so all GPUs still add bitwise the same values in the same order.
template <int FMT> static __device__ __forceinline__ uint32_t arn_pack2(const float a, const float b) {
    if (FMT == 1) {
        const half2 h = __floats2half2_rn(a, b);
        return *(const uint32_t *) &h;
    }
    uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
    ua += 0x7FFF + ((ua >> 16) & 1);
    ub += 0x7FFF + ((ub >> 16) & 1);
    return (ua >> 16) | (ub & 0xFFFF0000u);
}
template <int FMT> static __device__ __forceinline__ float2 arn_unpack2(const uint32_t v) {
    if (FMT == 1) {
        const half2 h = *(const half2 *) &v;
        return __half22float2(h);
    }
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xFFFF0000u));
}

template <int FMT>
static __global__ void __launch_bounds__(256) ggml_cuda_arn_push_f32(float * __restrict__ data, const ggml_cuda_arn_push_ptrs P,
        const int n, const int rank, const int count, const int token, const bool flag_fence) {
    const int tid = threadIdx.x, bid = blockIdx.x;
    const int gtid = bid*blockDim.x + tid, gnt = gridDim.x*blockDim.x;
    const int n4 = count / 4;
    KT_DECL;
    const bool kt_on = tid == 0 && bid == 0;
    if (kt_on) KT_STAMP(0);
    if (FMT == 0) {
        for (int i = gtid; i < n4; i += gnt) {
            const float4 v = ((const float4 *) data)[i];
            for (int j = 0; j < n; ++j) {
                if (j != rank) {
                    ((float4 *) (P.inbox[j] + (size_t) rank*ARN_PUSH_MAX_ELEMS))[i] = v;
                }
            }
        }
        for (int i = 4*n4 + gtid; i < count; i += gnt) {
            for (int j = 0; j < n; ++j) {
                if (j != rank) {
                    P.inbox[j][(size_t) rank*ARN_PUSH_MAX_ELEMS + i] = data[i];
                }
            }
        }
    } else {
        // 16-bit slots: the sender's slot holds count 16-bit values (8 bytes per float4)
        for (int i = gtid; i < n4; i += gnt) {
            const float4 v = ((const float4 *) data)[i];
            const uint2 pk = make_uint2(arn_pack2<FMT>(v.x, v.y), arn_pack2<FMT>(v.z, v.w));
            for (int j = 0; j < n; ++j) {
                if (j != rank) {
                    ((uint2 *) (P.inbox[j] + (size_t) rank*ARN_PUSH_MAX_ELEMS))[i] = pk;
                }
            }
        }
        for (int i = 4*n4 + gtid; i < count; i += gnt) {
            const uint32_t pk = arn_pack2<FMT>(data[i], 0.0f);
            for (int j = 0; j < n; ++j) {
                if (j != rank) {
                    ((unsigned short *) (P.inbox[j] + (size_t) rank*ARN_PUSH_MAX_ELEMS))[i] = (unsigned short) (pk & 0xFFFF);
                }
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (kt_on) KT_STAMP(1);
    // lane j (!= rank) of wave 0 sets our flag in peer j's array, then polls peer j's flag in ours: the peers' flags are
    // read in parallel (one uncached round trip, not one per peer). Nothing waits on our flag writes landing (only the
    // peers poll them): no fence after them (GGML_CUDA_ARN_FLAG_FENCE=1 restores the old order with a fence)
    if (flag_fence) {
        if (tid == 0) {
            for (int j = 0; j < n; ++j) {
                if (j != rank) {
                    *(volatile int *) (P.flag[j] + (rank*ARN_PUSH_MAX_BLOCKS + bid)*ARN_FLAG_INTS) = token;
                }
            }
            __threadfence_system();
            for (int j = 0; j < n; ++j) {
                if (j == rank) {
                    continue;
                }
                while (*(const volatile int *) (P.flag[rank] + (j*ARN_PUSH_MAX_BLOCKS + bid)*ARN_FLAG_INTS) - token < 0) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
        }
    } else if (tid < n && tid != rank) {
        *(volatile int *) (P.flag[tid] + (rank*ARN_PUSH_MAX_BLOCKS + bid)*ARN_FLAG_INTS) = token;
        while (*(const volatile int *) (P.flag[rank] + (tid*ARN_PUSH_MAX_BLOCKS + bid)*ARN_FLAG_INTS) - token < 0) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    if (kt_on) KT_STAMP(2);
    __threadfence_system();
    const float * mine = P.inbox[rank];
    if (FMT == 0) {
        for (int i = gtid; i < n4; i += gnt) {
            float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int j = 0; j < n; ++j) {
                const float4 b = j == rank ? ((const float4 *) data)[i] : ((const float4 *) (mine + (size_t) j*ARN_PUSH_MAX_ELEMS))[i];
                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
            }
            ((float4 *) data)[i] = a;
        }
        for (int i = 4*n4 + gtid; i < count; i += gnt) {
            float a = 0.0f;
            for (int j = 0; j < n; ++j) {
                a += j == rank ? data[i] : mine[(size_t) j*ARN_PUSH_MAX_ELEMS + i];
            }
            data[i] = a;
        }
    } else {
        for (int i = gtid; i < n4; i += gnt) {
            const float4 own = ((const float4 *) data)[i];
            const uint2 own_pk = make_uint2(arn_pack2<FMT>(own.x, own.y), arn_pack2<FMT>(own.z, own.w));
            float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int j = 0; j < n; ++j) {
                const uint2 pk = j == rank ? own_pk : ((const uint2 *) (mine + (size_t) j*ARN_PUSH_MAX_ELEMS))[i];
                const float2 lo = arn_unpack2<FMT>(pk.x), hi = arn_unpack2<FMT>(pk.y);
                a.x += lo.x; a.y += lo.y; a.z += hi.x; a.w += hi.y;
            }
            ((float4 *) data)[i] = a;
        }
        for (int i = 4*n4 + gtid; i < count; i += gnt) {
            float a = 0.0f;
            for (int j = 0; j < n; ++j) {
                const uint32_t pk = j == rank ? (arn_pack2<FMT>(data[i], 0.0f) & 0xFFFF) :
                    (uint32_t) ((const unsigned short *) (mine + (size_t) j*ARN_PUSH_MAX_ELEMS))[i];
                a += arn_unpack2<FMT>(pk).x;
            }
            data[i] = a;
        }
    }
    if (kt_on) { KT_STAMP(3); KT_COMMIT(0, 4); }
}

static bool ggml_cuda_arn_push_allreduce(ggml_backend_cuda_comm_context * comm, struct ggml_tensor ** tensors, const int64_t ne) {
    const int n = (int) comm->backends.size();
    if (comm->arn_push_state < 0) {
        return false;
    }
    // GGML_CUDA_ARN_SKIP=1 (ablation only, results are wrong): no transfer and no wait, the partials stay as they are
    static const bool skip = [] { const char * e = getenv("GGML_CUDA_ARN_SKIP"); return e && atoi(e) != 0; }();
    if (skip) {
        return true;
    }
    if (comm->arn_push_state == 0) {
        comm->arn_push_state = 1;
        comm->arn_inbox.assign(n, nullptr);
        comm->arn_flag.assign(n, nullptr);
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(comm->dev_ids[i]);
            const size_t flag_bytes = (size_t) n*ARN_PUSH_MAX_BLOCKS*ARN_FLAG_INTS*sizeof(int);
            if (hipExtMallocWithFlags((void **) &comm->arn_inbox[i], 2*(size_t) n*ARN_PUSH_MAX_ELEMS*sizeof(float),
                        hipDeviceMallocUncached) != hipSuccess ||
                    hipExtMallocWithFlags((void **) &comm->arn_flag[i], flag_bytes, hipDeviceMallocUncached) != hipSuccess ||
                    cudaMemset(comm->arn_flag[i], 0, flag_bytes) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                (void) cudaGetLastError();
                comm->arn_push_state = -1; // push path unavailable (the event path stays)
                return false;
            }
        }
    }
    const int token = ++comm->arn_token;
    ggml_cuda_arn_push_ptrs P = {};
    for (int j = 0; j < n; ++j) {
        P.inbox[j] = comm->arn_inbox[j] + (size_t) (token & 1)*n*ARN_PUSH_MAX_ELEMS;
        P.flag[j]  = comm->arn_flag[j];
    }
    const int blocks = (int) std::max<int64_t>(1, std::min<int64_t>(ARN_PUSH_MAX_BLOCKS, (ne/4 + 255)/256));
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        cudaStream_t st = ((ggml_backend_cuda_context *) comm->backends[i]->context)->stream();
        if (!(tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ne*sizeof(float), st)); // zero-sized slice: contributes 0
        }
        static const bool flag_fence = [] { const char * e = getenv("GGML_CUDA_ARN_FLAG_FENCE"); return e && atoi(e) != 0; }();
        // the transfer format (GGML_CUDA_ARN_PUSH_FMT: 0 f32, 1 f16, 2 bf16 default): the pushes are link bound (60 KB per
        // GPU and AllReduce for 4 GPUs x 5120 values, ~9 us on a gen3 x8 link), 16-bit halves the bytes. V4.1 decode KLD vs
        // f32 (c512 x 4, b1): bf16 0.027, f16 0.029 = the noise of a norm sum-order change (0.026); tg 91.0 -> 92.6.
        // bf16 keeps the f32 range (f16 would turn an activation above 65504 into inf)
        static const int fmt = [] { const char * e = getenv("GGML_CUDA_ARN_PUSH_FMT"); return e ? atoi(e) : 2; }();
        if (fmt == 1) {
            ggml_cuda_arn_push_f32<1><<<blocks, 256, 0, st>>>((float *) tensors[i]->data, P, n, i, (int) ne, token, flag_fence);
        } else if (fmt == 2) {
            ggml_cuda_arn_push_f32<2><<<blocks, 256, 0, st>>>((float *) tensors[i]->data, P, n, i, (int) ne, token, flag_fence);
        } else {
            ggml_cuda_arn_push_f32<0><<<blocks, 256, 0, st>>>((float *) tensors[i]->data, P, n, i, (int) ne, token, flag_fence);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    // after every GPU's kernel is launched (a sync inside the loop would wait on peers that are not launched yet)
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        KT_HOST_TICK(0, "arn_push", 4, ((ggml_backend_cuda_context *) comm->backends[i]->context)->stream());
    }
    return true;
}

// Large tensors with a 16-bit transfer format (GGML_CUDA_ARN_FMT: 0 f32, 1 f16, 2 bf16 default): the f32 phases above are
// link bound (V4.1 prefill, 1024 tokens: 21 MB per AllReduce, ~1.2 ms per phase over the x8 links; 15% of the encoder
// GPUs' time). Every GPU packs its partial into its staging buffer; the reduce phase reads the n packed partials of its
// shard, sums them in device order and rounds the sum, which goes to the tensor and back into its own staging shard; the
// gather phase expands the peers' packed sums. Half the PCIe bytes, and every GPU ends with bitwise the same values.
// The gather reads all peers at once (k outer, peers inner): the f32 gather walks the peers in order, so at any moment
// all GPUs read the same peer and only one link sends.
struct ggml_cuda_arn_ptrs16 {
    uint16_t * p[GGML_CUDA_MAX_DEVICES];
};

template <int FMT> static __device__ __forceinline__ void arn_unpack8(float * a, const uint4 v) {
    const float2 x0 = arn_unpack2<FMT>(v.x), x1 = arn_unpack2<FMT>(v.y), x2 = arn_unpack2<FMT>(v.z), x3 = arn_unpack2<FMT>(v.w);
    a[0] = x0.x; a[1] = x0.y; a[2] = x1.x; a[3] = x1.y; a[4] = x2.x; a[5] = x2.y; a[6] = x3.x; a[7] = x3.y;
}

template <int FMT>
static __global__ void __launch_bounds__(256) ggml_cuda_arn_pack16(uint16_t * __restrict__ dst, const float * __restrict__ src,
        const int64_t ne) {
    const int64_t n8 = ne / 8;
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    for (int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < n8; k += stride) {
        const float4 a = ((const float4 *) src)[2*k + 0];
        const float4 b = ((const float4 *) src)[2*k + 1];
        ((uint4 *) dst)[k] = make_uint4(arn_pack2<FMT>(a.x, a.y), arn_pack2<FMT>(a.z, a.w),
                                        arn_pack2<FMT>(b.x, b.y), arn_pack2<FMT>(b.z, b.w));
    }
    for (int64_t k = 8*n8 + (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < ne; k += stride) {
        dst[k] = (uint16_t) (arn_pack2<FMT>(src[k], 0.0f) & 0xFFFF);
    }
}

// shard [off, off + len) of this GPU: the rounded sum of the N packed partials, into dst (f32) and own (packed; own is
// this GPU's staging = src.p[self], each element is read before it is overwritten by the same thread)
template <int FMT, int N>
static __global__ void __launch_bounds__(256) ggml_cuda_arn_reduce16(float * __restrict__ dst, uint16_t * own,
        const ggml_cuda_arn_ptrs16 src, const int64_t off, const int64_t len) {
    const int64_t n8 = len / 8;
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    for (int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < n8; k += stride) {
        uint4 v[N];
#pragma unroll
        for (int j = 0; j < N; ++j) {
            v[j] = ((const uint4 *) (src.p[j] + off))[k];
        }
        float a[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float b[8];
            arn_unpack8<FMT>(b, v[j]);
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                a[l] += b[l];
            }
        }
        const uint4 r = make_uint4(arn_pack2<FMT>(a[0], a[1]), arn_pack2<FMT>(a[2], a[3]),
                                   arn_pack2<FMT>(a[4], a[5]), arn_pack2<FMT>(a[6], a[7]));
        ((uint4 *) (own + off))[k] = r;
        arn_unpack8<FMT>(a, r);
        ((float4 *) (dst + off))[2*k + 0] = make_float4(a[0], a[1], a[2], a[3]);
        ((float4 *) (dst + off))[2*k + 1] = make_float4(a[4], a[5], a[6], a[7]);
    }
    for (int64_t k = 8*n8 + (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < len; k += stride) {
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            a += arn_unpack2<FMT>((uint32_t) src.p[j][off + k]).x;
        }
        const uint32_t r = arn_pack2<FMT>(a, 0.0f) & 0xFFFF;
        own[off + k] = (uint16_t) r;
        dst[off + k] = arn_unpack2<FMT>(r).x;
    }
}

// dst[j*shard .. min(ne, (j+1)*shard)) = expanded src.p[j][same] for every j != self; [0, common) of every peer's shard
// in one pass (all peers' loads in flight together), the rest of a shorter-or-longer last shard after
template <int FMT, int N>
static __global__ void __launch_bounds__(256) ggml_cuda_arn_gather16(float * __restrict__ dst, const ggml_cuda_arn_ptrs16 src,
        const int self, const int64_t ne, const int64_t shard, const int64_t common) {
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    const int64_t c8 = common / 8;
    for (int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < c8; k += stride) {
        uint4 v[N - 1];
#pragma unroll
        for (int t = 1; t < N; ++t) {
            const int j = (self + t) % N;
            v[t - 1] = ((const uint4 *) (src.p[j] + j*shard))[k];
        }
#pragma unroll
        for (int t = 1; t < N; ++t) {
            const int j = (self + t) % N;
            float a[8];
            arn_unpack8<FMT>(a, v[t - 1]);
            ((float4 *) (dst + j*shard))[2*k + 0] = make_float4(a[0], a[1], a[2], a[3]);
            ((float4 *) (dst + j*shard))[2*k + 1] = make_float4(a[4], a[5], a[6], a[7]);
        }
    }
    for (int t = 1; t < N; ++t) {
        const int j = (self + t) % N;
        const int64_t off = j*shard;
        const int64_t len = shard < ne - off ? shard : ne - off;
        for (int64_t k = 8*c8 + (int64_t) blockIdx.x*blockDim.x + threadIdx.x; k < len; k += stride) {
            dst[off + k] = arn_unpack2<FMT>((uint32_t) src.p[j][off + k]).x;
        }
    }
}

// The 16-bit large-tensor AllReduce moves its shards with peer copies (copy engines) instead of the reduce / gather
// kernels reading peer memory (GGML_CUDA_ARN_DMA=0: kernel reads): 4 MI50s all-to-all, per GPU in: hipMemcpyPeerAsync
// 10.2 GB/s vs kernel pull 8.1 (opt1003/p2pdma.hip). Same sums in the same order (bit-identical results); prefill DeepSeek
// V4 Flash TP4 ub2048 1142 -> 1162 t/s, GLM-5.3 Q2 ub1024 960 -> 979, a 17.8K-token DS4 prompt 888 -> 900.
static bool ggml_cuda_arn_dma_env() {
    static const bool e = [] { const char * v = getenv("GGML_CUDA_ARN_DMA"); return !v || atoi(v) != 0; }();
    return e;
}

// grow the staging buffers (drains every stream: queued kernels may still read the old ones); never while capturing
static bool ggml_cuda_arn_stage_reserve(ggml_backend_cuda_comm_context * comm, const int64_t ne, cudaStream_t * st) {
    const int n = (int) comm->backends.size();
    if ((size_t) ne <= comm->arn_stage_elems) {
        return true;
    }
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(st[i], &cs) != hipSuccess || cs != hipStreamCaptureStatusNone) {
            (void) cudaGetLastError();
            return false;
        }
    }
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        CUDA_CHECK(cudaStreamSynchronize(st[i]));
    }
    const size_t elems = GGML_PAD((size_t) ne, (size_t) 1 << 20);
    comm->arn_stage.resize(n, nullptr);
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        if (comm->arn_stage[i] != nullptr) {
            CUDA_CHECK(cudaFree(comm->arn_stage[i]));
            comm->arn_stage[i] = nullptr;
        }
        if (ok && cudaMalloc((void **) &comm->arn_stage[i], elems*sizeof(uint16_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            comm->arn_stage[i] = nullptr;
            ok = false;
        }
        if (ggml_cuda_arn_dma_env()) {
            comm->arn_rx.resize(n, nullptr);
            if (comm->arn_rx[i] != nullptr) {
                CUDA_CHECK(cudaFree(comm->arn_rx[i]));
                comm->arn_rx[i] = nullptr;
            }
            if (ok && cudaMalloc((void **) &comm->arn_rx[i], elems*sizeof(uint16_t)) != cudaSuccess) {
                (void) cudaGetLastError();
                comm->arn_rx[i] = nullptr;
                ok = false;
            }
        }
    }
    if (!ok) {
        for (int i = 0; i < n; ++i) {
            if (comm->arn_stage[i] != nullptr) {
                ggml_cuda_set_device(comm->dev_ids[i]);
                (void) cudaFree(comm->arn_stage[i]);
                comm->arn_stage[i] = nullptr;
            }
            if (i < (int) comm->arn_rx.size() && comm->arn_rx[i] != nullptr) {
                ggml_cuda_set_device(comm->dev_ids[i]);
                (void) cudaFree(comm->arn_rx[i]);
                comm->arn_rx[i] = nullptr;
            }
        }
        comm->arn_rx.clear();
        comm->arn_stage_elems = 0;
        return false;
    }
    comm->arn_stage_elems = elems;
    return true;
}

template <int FMT, int N>
static void ggml_cuda_arn_allreduce16(ggml_backend_cuda_comm_context * comm, struct ggml_tensor ** tensors, const int64_t ne,
        cudaStream_t * st) {
    const int64_t shard = GGML_PAD((ne + N - 1)/N, 64);
    ggml_cuda_arn_ptrs16 stage = {};
    for (int i = 0; i < N; ++i) {
        stage.p[i] = comm->arn_stage[i];
    }
    auto wait_peers = [&](const int i, const int ph) {
        for (int j = 0; j < N; ++j) {
            if (j != i) {
                CUDA_CHECK(cudaStreamWaitEvent(st[i], comm->arn_ev[j][ph], 0));
            }
        }
    };
    const int threads = 256;
    if (ggml_cuda_arn_dma_env() && (int) comm->arn_rx.size() == N) {
        auto len_of = [&](const int s) { return std::max<int64_t>(0, std::min(shard, ne - s*shard)); };
        int phys[N];
        for (int i = 0; i < N; ++i) {
            phys[i] = ggml_cuda_get_physical_device(comm->dev_ids[i]);
        }
        // pack, then push the peers' shards of the partial into their rx (sender i -> slot i, or i - 1 past the receiver)
        for (int i = 0; i < N; ++i) {
            ggml_cuda_set_device(comm->dev_ids[i]);
            if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
                const int blocks = (int) std::min<int64_t>(480, (ne/8 + threads - 1)/threads + 1);
                ggml_cuda_arn_pack16<FMT><<<blocks, threads, 0, st[i]>>>(stage.p[i], (const float *) tensors[i]->data, ne);
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(stage.p[i], 0, ne*sizeof(uint16_t), st[i])); // zero-sized slice: contributes 0
            }
            for (int t = 1; t < N; ++t) {
                const int j = (i + t) % N;
                const int64_t len = len_of(j);
                if (len > 0) {
                    const int slot = i < j ? i : i - 1;
                    CUDA_CHECK(cudaMemcpyPeerAsync(comm->arn_rx[j] + slot*shard, phys[j], stage.p[i] + j*shard, phys[i],
                                                   len*sizeof(uint16_t), st[i]));
                }
            }
            CUDA_CHECK(cudaEventRecord(comm->arn_ev[i][0], st[i]));
        }
        // reduce the own shard from local memory (same order as the pull version), then push it into every peer's staging
        for (int i = 0; i < N; ++i) {
            ggml_cuda_set_device(comm->dev_ids[i]);
            wait_peers(i, 0);
            const int64_t off = i*shard;
            const int64_t len = len_of(i);
            if (len > 0) {
                ggml_cuda_arn_ptrs16 loc = {};
                for (int j = 0; j < N; ++j) {
                    loc.p[j] = j == i ? stage.p[i] : comm->arn_rx[i] + (j < i ? j : j - 1)*shard - off;
                }
                const int blocks = (int) std::min<int64_t>(480, (len/8 + threads - 1)/threads + 1);
                ggml_cuda_arn_reduce16<FMT, N><<<blocks, threads, 0, st[i]>>>((float *) tensors[i]->data, stage.p[i], loc, off, len);
                CUDA_CHECK(cudaGetLastError());
                for (int t = 1; t < N; ++t) {
                    const int j = (i + t) % N;
                    CUDA_CHECK(cudaMemcpyPeerAsync(stage.p[j] + off, phys[j], stage.p[i] + off, phys[i], len*sizeof(uint16_t), st[i]));
                }
            }
            CUDA_CHECK(cudaEventRecord(comm->arn_ev[i][1], st[i]));
        }
        // expand the peers' reduced shards, now in the own staging
        for (int i = 0; i < N; ++i) {
            ggml_cuda_set_device(comm->dev_ids[i]);
            wait_peers(i, 1);
            int64_t common = shard;
            for (int j = 0; j < N; ++j) {
                if (j != i) {
                    common = std::min(common, len_of(j));
                }
            }
            ggml_cuda_arn_ptrs16 loc = {};
            for (int j = 0; j < N; ++j) {
                loc.p[j] = stage.p[i];
            }
            const int blocks = (int) std::min<int64_t>(480, (shard/8 + threads - 1)/threads + 1);
            ggml_cuda_arn_gather16<FMT, N><<<blocks, threads, 0, st[i]>>>((float *) tensors[i]->data, loc, i, ne, shard, common);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(comm->arn_ev[i][2], st[i]));
        }
        for (int i = 0; i < N; ++i) {
            ggml_cuda_set_device(comm->dev_ids[i]);
            wait_peers(i, 2);
        }
        return;
    }
    for (int i = 0; i < N; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            const int blocks = (int) std::min<int64_t>(480, (ne/8 + threads - 1)/threads + 1);
            ggml_cuda_arn_pack16<FMT><<<blocks, threads, 0, st[i]>>>(stage.p[i], (const float *) tensors[i]->data, ne);
            CUDA_CHECK(cudaGetLastError());
        } else {
            CUDA_CHECK(cudaMemsetAsync(stage.p[i], 0, ne*sizeof(uint16_t), st[i])); // zero-sized slice: contributes 0
        }
        CUDA_CHECK(cudaEventRecord(comm->arn_ev[i][0], st[i]));
    }
    for (int i = 0; i < N; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        wait_peers(i, 0);
        const int64_t off = i*shard;
        const int64_t len = std::max<int64_t>(0, std::min(shard, ne - off));
        if (len > 0) {
            const int blocks = (int) std::min<int64_t>(480, (len/8 + threads - 1)/threads + 1);
            ggml_cuda_arn_reduce16<FMT, N><<<blocks, threads, 0, st[i]>>>((float *) tensors[i]->data, stage.p[i], stage, off, len);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(comm->arn_ev[i][1], st[i]));
    }
    for (int i = 0; i < N; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        wait_peers(i, 1);
        int64_t common = shard;
        for (int j = 0; j < N; ++j) {
            if (j != i) {
                common = std::min(common, std::max<int64_t>(0, std::min(shard, ne - j*shard)));
            }
        }
        const int blocks = (int) std::min<int64_t>(480, (shard/8 + threads - 1)/threads + 1);
        ggml_cuda_arn_gather16<FMT, N><<<blocks, threads, 0, st[i]>>>((float *) tensors[i]->data, stage, i, ne, shard, common);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(comm->arn_ev[i][2], st[i]));
    }
    // no GPU may repack its staging (next AllReduce) or reuse its tensor before every peer finished reading it
    for (int i = 0; i < N; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        wait_peers(i, 2);
    }
}

static bool ggml_cuda_arn_allreduce(ggml_backend_cuda_comm_context * comm, struct ggml_tensor ** tensors) {
    { static const bool skip_all = [] { const char * e = getenv("GGML_CUDA_ARN_SKIP"); return e && atoi(e) >= 2; }(); if (skip_all) { return true; } }
    static const bool     enabled   = [] { const char * e = getenv("GGML_CUDA_ARN"); return !e || atoi(e) != 0; }();
    static const uint64_t min_bytes = [] { const char * e = getenv("GGML_CUDA_ARN_MIN_BYTES"); return e ? (uint64_t) atoll(e) : (uint64_t) (1u << 20); }();
    static const bool debug = [] { const char * e = getenv("GGML_CUDA_ARN_DEBUG"); return e && atoi(e) != 0; }();
    const int n = (int) comm->backends.size();
    if (debug) {
        static int left = 20;
        if (left-- > 0) {
            fprintf(stderr, "%s: n %d state %d %s type %s ne [%lld, %lld, %lld, %lld] nbytes %zu contig %d align %d flags %d\n",
                __func__, n, comm->arn_state, tensors[0] ? tensors[0]->name : "(null)",
                tensors[0] ? ggml_type_name(tensors[0]->type) : "-", tensors[0] ? (long long) tensors[0]->ne[0] : 0LL,
                tensors[0] ? (long long) tensors[0]->ne[1] : 0LL, tensors[0] ? (long long) tensors[0]->ne[2] : 0LL,
                tensors[0] ? (long long) tensors[0]->ne[3] : 0LL, tensors[0] ? ggml_nbytes(tensors[0]) : (size_t) 0,
                tensors[0] ? (int) ggml_is_contiguously_allocated(tensors[0]) : -1,
                tensors[0] ? (int) (((uintptr_t) tensors[0]->data & 0xF) == 0) : -1, tensors[0] ? (int) tensors[0]->flags : -1);
        }
    }
    static const bool push_on = [] { const char * e = getenv("GGML_CUDA_ARN_PUSH"); return !e || atoi(e) != 0; }();
    const bool push = push_on && tensors[0] != nullptr && ggml_nelements(tensors[0]) <= ARN_PUSH_MAX_ELEMS;
    if (!enabled || n <= 2 || n > GGML_CUDA_MAX_DEVICES || comm->arn_state < 0 || tensors[0] == nullptr ||
            tensors[0]->type != GGML_TYPE_F32 || (ggml_nbytes(tensors[0]) < min_bytes && !push)) {
        return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    for (int i = 0; i < n; ++i) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) || ((uintptr_t) tensors[i]->data & 0xF) != 0) {
            return false;
        }
    }
    if (comm->arn_state == 0) {
        // every GPU must read every peer's memory
        comm->arn_state = 1;
        for (int i = 0; i < n && comm->arn_state == 1; ++i) {
            const int pi = ggml_cuda_get_physical_device(comm->dev_ids[i]);
            CUDA_CHECK(cudaSetDevice(pi));
            for (int j = 0; j < n; ++j) {
                if (j == i) {
                    continue;
                }
                const int pj = ggml_cuda_get_physical_device(comm->dev_ids[j]);
                if (pj == pi) {
                    comm->arn_state = -1; // two virtual devices on one GPU: not this path
                    break;
                }
                int can = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can, pi, pj));
                if (!can) {
                    comm->arn_state = -1;
                    break;
                }
                const cudaError_t rc = cudaDeviceEnablePeerAccess(pj, 0);
                if (rc != cudaSuccess && rc != cudaErrorPeerAccessAlreadyEnabled) {
                    (void) cudaGetLastError();
                    comm->arn_state = -1;
                    break;
                }
                (void) cudaGetLastError();
            }
        }
        if (comm->arn_state == 1) {
            comm->arn_ev.resize(n);
            for (int i = 0; i < n; ++i) {
                ggml_cuda_set_device(comm->dev_ids[i]);
                for (auto & e : comm->arn_ev[i]) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                }
            }
        }
        if (debug) {
            fprintf(stderr, "%s: %d-GPU peer AllReduce %s\n", __func__, n, comm->arn_state == 1 ? "enabled" : "unavailable");
        }
        if (comm->arn_state != 1) {
            return false;
        }
    }

    if (push && ggml_cuda_arn_push_allreduce(comm, tensors, ne)) {
        return true;
    }
    if (ggml_nbytes(tensors[0]) < min_bytes) {
        return false;
    }

    cudaStream_t st[GGML_CUDA_MAX_DEVICES];
    for (int i = 0; i < n; ++i) {
        st[i] = ((ggml_backend_cuda_context *) comm->backends[i]->context)->stream();
    }
    static const int fmt16 = [] { const char * e = getenv("GGML_CUDA_ARN_FMT"); return e ? atoi(e) : 2; }();
    if ((fmt16 == 1 || fmt16 == 2) && (n == 4 || n == 8) && ggml_cuda_arn_stage_reserve(comm, ne, st)) {
        if (fmt16 == 1) {
            if (n == 4) { ggml_cuda_arn_allreduce16<1, 4>(comm, tensors, ne, st); }
            else        { ggml_cuda_arn_allreduce16<1, 8>(comm, tensors, ne, st); }
        } else {
            if (n == 4) { ggml_cuda_arn_allreduce16<2, 4>(comm, tensors, ne, st); }
            else        { ggml_cuda_arn_allreduce16<2, 8>(comm, tensors, ne, st); }
        }
        return true;
    }

    ggml_cuda_arn_ptrs all = {};
    for (int i = 0; i < n; ++i) {
        all.p[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) ? (const float *) tensors[i]->data : nullptr;
    }
    const int64_t shard = GGML_PAD((ne + n - 1)/n, 64);
    auto record = [&](const int ph) {
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(comm->dev_ids[i]);
            CUDA_CHECK(cudaEventRecord(comm->arn_ev[i][ph], st[i]));
        }
    };
    auto wait_peers = [&](const int i, const int ph) {
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                CUDA_CHECK(cudaStreamWaitEvent(st[i], comm->arn_ev[j][ph], 0));
            }
        }
    };
    const int threads = 256;
    record(0);
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        wait_peers(i, 0);
        const int64_t off = i*shard;
        const int64_t len = std::max<int64_t>(0, std::min(shard, ne - off));
        if (len > 0) {
            const int blocks = (int) std::min<int64_t>(480, (len/4 + threads - 1)/threads + 1);
            ggml_cuda_arn_reduce_f32<<<blocks, threads, 0, st[i]>>>((float *) tensors[i]->data, all, n, off, len);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    record(1);
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        wait_peers(i, 1);
        ggml_cuda_arn_ptrs peers = {};
        for (int j = 0; j < n; ++j) {
            peers.p[j] = (const float *) tensors[j]->data;
        }
        const int blocks = (int) std::min<int64_t>(480, ((n - 1)*shard/4 + threads - 1)/threads + 1);
        ggml_cuda_arn_gather_f32<<<blocks, threads, 0, st[i]>>>((float *) tensors[i]->data, peers, n, i, ne, shard);
        CUDA_CHECK(cudaGetLastError());
    }
    record(2);
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(comm->dev_ids[i]);
        wait_peers(i, 2);
    }
    return true;
}

static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    void * streams[4] = {};
    for (size_t j = 0; j < comm_ctx->backends.size() && j < 4; ++j) {
        streams[j] = ((ggml_backend_cuda_context *) comm_ctx->backends[j]->context)->stream();
    }
    if ((comm_ctx->backends.size() == 2 || comm_ctx->backends.size() == 4) &&
            ggml_cuda_hcp_ar_defer(comm_ctx->dev_ids.data(), (int) comm_ctx->dev_ids.size(), tensors, streams)) {
        return true; // done by the next graph's persistent HC kernel (or its fallback exchange)
    }
    if (comm_ctx->backends.size() > 2 && ggml_cuda_arn_allreduce(comm_ctx, tensors)) {
        return true;
    }
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

// split-K F16 GEMM (see ggml_cuda_mul_mat_cublas_impl): dst = sum of the ks partial products, in f32
static __global__ void k_sum_split_k_f16(const half * __restrict__ part, float * __restrict__ dst, const int64_t n, const int ks) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float s = 0.0f;
    for (int k = 0; k < ks; ++k) {
        s += __half2float(part[k*n + i]);
    }
    dst[i] = s;
}

// split-K F32 GEMM: dst = sum of the ks partial products in order
static __global__ void k_sum_split_k_f32(const float * __restrict__ part, float * __restrict__ dst, const int64_t n, const int ks) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float s = 0.0f;
    for (int k = 0; k < ks; ++k) {
        s += part[k*n + i];
    }
    dst[i] = s;
}

template<ggml_type compute_type>
// acc_f32: F16 inputs, F32 accumulation and output (otherwise GCN and older GPUs accumulate F16 GEMMs in F16)
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const bool acc_f32 = false) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = acc_f32 || cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // GCN, F16 weights with few rows and a long K (DeepSeek V4's compressor / indexer / router projections at prefill:
    // 256 x 4096 x 512 tokens ran at 3.4 TFLOPS, rocBLAS gets too few output tiles for 60 CUs): K split into ks slices as
    // one strided-batched GEMM (ks times the tiles, F16 partial products), summed in f32 by one kernel (replacing the
    // F16 -> F32 conversion). GGML_CUDA_F16_SPLITK=0 off.
    if constexpr (compute_type == GGML_TYPE_F16) {
        static const int splitk_env = [] { const char * e = getenv("GGML_CUDA_F16_SPLITK"); return e ? atoi(e) : 1; }();
        const int64_t tiles = ((ne01 + 63)/64)*((ne11 + 63)/64);
        int ks = 1;
        while (ks < 8 && tiles*ks < 240 && ne10 % (128*ks*2) == 0) {
            ks *= 2;
        }
        if (splitk_env && GGML_CUDA_CC_IS_GCN(cc) && !prefer_f32_output && ne12 == 1 && ne13 == 1 && ks > 1 && ne10 >= 1024 &&
                ne11 >= 16) {
            const int64_t kk = ne10/ks;
            ggml_cuda_pool_alloc<half> part(ctx.pool(), (size_t) ks*ne01*ne11);
            CUBLAS_CHECK(
            cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, kk,
                    alpha, src0_ptr, cu_data_type_a, s01, kk,
                           src1_ptr, cu_data_type_b, s11, kk,
                    beta,  part.get(), cu_data_type, ne01, ne01*ne11,
                    ks,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            const int64_t n = ne01*ne11;
            k_sum_split_k_f16<<<(unsigned) ((n + 255)/256), 256, 0, main_stream>>>(part.get(), dst_ddf, n, ks);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        // GCN, few output values (DeepSeek V4's router, F32 precision: 256 rows x 4096): rocBLAS sgemm leaves most CUs idle
        // (MI50, 2048 tokens 1496 us = 2.9 TFLOPS, 16-256 tokens 260-465 us). K split into ks slices as one strided-batched
        // sgemm, the partial products summed in order: 2048 tokens 704 us (ks 4), 1024: 818 -> 418, 256: 464 -> 119,
        // 16: 259 -> 28 (ks 8); 512 x 4096 x 2048: 1320 -> 1062, 1024 rows: 2442 -> 1748 (ks 4). rocBLAS is erratic in ks
        // (GLM's f32 router 288 rows, 1024 tokens: ks 2 437, 4 538, 8 464 us); this rule is within ~6% of the best ks for
        // every shape measured. GGML_CUDA_F32_SPLITK=0 off.
        static const int splitk32_env = [] { const char * e = getenv("GGML_CUDA_F32_SPLITK"); return e ? atoi(e) : 1; }();
        const int64_t n_out = ne01*ne11;
        const int ks = n_out <= 600*1024 ? 8 : n_out <= 2304*1024 ? 4 : 1;
        if (splitk32_env && GGML_CUDA_CC_IS_GCN(cc) && ks > 1 && ne10 >= 1024 && ne10 % (128*ks) == 0 && ne11 >= 2) {
            const int64_t kk = ne10/ks;
            ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) ks*n_out);
            CUBLAS_CHECK(
            cublasSgemmStridedBatched(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, kk,
                    (const float *) alpha, (const float *) src0_ptr, s01, kk,
                                           (const float *) src1_ptr, s11, kk,
                    (const float *) beta,  part.get(), ne01, n_out,
                    ks));
            k_sum_split_k_f32<<<(unsigned) ((n_out + 255)/256), 256, 0, main_stream>>>(part.get(), dst_ddf, n_out, ks);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F32 accumulation requested: when src1 may also be rounded to F16 (ggml_prec_set_src(dst, GGML_PREC_F16, 1)), keep
    // the F16 inputs and only accumulate (and output) in F32 instead of converting everything to F32
    bool acc_f32 = false;
    if (dst->op_params[0] == GGML_PREC_F32) {
        if (compute_type == GGML_TYPE_F16 && dst->op_params[3] >= GGML_PREC_F16) {
            acc_f32 = true;
        } else {
            compute_type = GGML_TYPE_F32;
        }
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst, acc_f32);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

// the bias-free gate/up GLU pair with 2-4 columns (MTP verify) on q8_0 MMVQ (fused kernels for 2-4 columns exist for
// q8_0 only); opt-in GGML_CUDA_GLU_COLS=1: GLM shared expert at 4 tokens 25.17 -> 25.70 ms per verify batch (slower)
static bool ggml_cuda_should_fuse_mmvq_glu_cols(const ggml_tensor * up) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_GLU_COLS"); return e && atoi(e) != 0; }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return env && up->op == GGML_OP_MUL_MAT && up->src[0]->type == GGML_TYPE_Q8_0 && up->src[1]->type == GGML_TYPE_F32 &&
        up->type == GGML_TYPE_F32 && up->ne[1] >= 2 && up->ne[1] <= 4 && up->ne[2] == 1 && up->ne[3] == 1 &&
        cc > GGML_CUDA_CC_PASCAL && ggml_cuda_should_use_mmvq(GGML_TYPE_Q8_0, cc, up->ne[1]) &&
        ggml_backend_buffer_get_usage(up->src[0]->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
}

// 2-token q6_K matvec + an addend of its output's shape in one MMVQ launch, where the plain MUL_MAT would use MMVQ too
// (long K; short K: gcn_kq_mv with the addend). GGML_CUDA_MMVQ_ADD2=0 off
static bool ggml_cuda_mmvq_add2_ok(const ggml_tensor * mm, const ggml_tensor * add, const ggml_tensor * addend) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_MMVQ_ADD2"); return !e || atoi(e) != 0; }();
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    return env && mm->op == GGML_OP_MUL_MAT && src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_F32 &&
        mm->type == GGML_TYPE_F32 && mm->ne[1] == 2 && mm->ne[2] == 1 && mm->ne[3] == 1 &&
        ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        add->type == GGML_TYPE_F32 && addend->type == GGML_TYPE_F32 && ggml_is_contiguous(add) && ggml_is_contiguous(addend) &&
        ggml_are_same_shape(add, addend) && addend->nb[1] == add->nb[1] &&
        GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

// F32 matmul with very few output rows (ne01 <= 64) over a batch of tokens, e.g. the qwen4exp
// hyper-connection inject projections (10240 -> 4) and ssm_alpha/beta (2560 -> 48). BLAS covers a
// 4 x n_tokens output with a handful of 32x32 tiles, so a few workgroups walk the whole K dimension
// while the rest of the GPU idles. Here one block per token reads its activation row once (float4),
// every thread accumulates all R outputs against the small, cache-resident weight matrix, and a block
// reduction writes the R results. Bandwidth-bound on the activations; no matrix cores involved.
#define MM_FEW_ROWS_THREADS 256

// Tiled variant for more rows or long K: a block owns RB rows x TB tokens with NT threads along K, so the per-block
// reduction is RB*TB values (not R*TB for all rows) and there are (rows/RB) x (tokens/TB) blocks to fill the GPU.
template <int RB, int TB, int NT, typename TX>
static __global__ void __launch_bounds__(NT)
mul_mat_f32_rows_tiled(const float * __restrict__ w, const TX * __restrict__ x, float * __restrict__ dst,
                       const int K, const int nrows, const int64_t ntok, const int64_t stride_x, const int64_t stride_dst) {
    const int64_t t0 = (int64_t) blockIdx.x*TB;
    const int     r0 = blockIdx.y*RB;
    const int K4 = K / 4;

    float acc[TB][RB];
#pragma unroll
    for (int b = 0; b < TB; ++b) {
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            acc[b][r] = 0.0f;
        }
    }
    for (int k = threadIdx.x; k < K4; k += NT) {
        float4 xv[TB];
#pragma unroll
        for (int b = 0; b < TB; ++b) {
            const int64_t t = min(t0 + b, ntok - 1);
            if constexpr (std::is_same_v<TX, half>) {
                const half2 * h = (const half2 *) (x + t*stride_x) + 2*k;
                const float2 lo = __half22float2(h[0]), hi = __half22float2(h[1]);
                xv[b] = make_float4(lo.x, lo.y, hi.x, hi.y);
            } else {
                xv[b] = ((const float4 *) (x + t*stride_x))[k];
            }
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            const float4 wv = ((const float4 *) (w + (int64_t) min(r0 + r, nrows - 1)*K))[k];
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                acc[b][r] += xv[b].x*wv.x + xv[b].y*wv.y + xv[b].z*wv.z + xv[b].w*wv.w;
            }
        }
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = NT / warp_size;
    __shared__ float red[nwarps][TB*RB];
    const int lane = threadIdx.x % warp_size;
    const int wid  = threadIdx.x / warp_size;
#pragma unroll
    for (int b = 0; b < TB; ++b) {
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            const float v = warp_reduce_sum<warp_size>(acc[b][r]);
            if (lane == 0) {
                red[wid][b*RB + r] = v;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TB*RB; i += NT) {
        const int b = i / RB, r = i % RB;
        if (r0 + r >= nrows || t0 + b >= ntok) {
            continue;
        }
        float sum = 0.0f;
#pragma unroll
        for (int q = 0; q < nwarps; ++q) {
            sum += red[q][i];
        }
        dst[(t0 + b)*stride_dst + r0 + r] = sum;
    }
}

// GGML_CUDA_FEW_ROWS_TILED=0 keeps the untiled kernels; GGML_CUDA_FR_CFG=RB,TB,NT forces a tile (experiments)
template <typename TX>
static bool ggml_cuda_few_rows_tiled(const float * w, const TX * x, float * d, int64_t K, int64_t nrows, int64_t ntok,
        int64_t sx, int64_t sd, cudaStream_t stream) {
    static const int env = [] { const char * e = getenv("GGML_CUDA_FEW_ROWS_TILED"); return e ? atoi(e) : 1; }();
    static const int cfg = [] { const char * e = getenv("GGML_CUDA_FR_CFG"); return e ? atoi(e) : 0; }();
    if (!env) {
        return false;
    }
    int sel = cfg;
    if (!sel) {
        // gfx906, 384 tokens: 48 x 2560 148 -> 68 us (8 rows x 4 tokens x 64 threads); 4 x 10240 62 -> 49 us
        sel = nrows > 16 ? 5 : (K >= 8192 ? 2 : 0);
    }
    const int64_t nt = ntok;
#define FR_TILED(RB, TB, NT) mul_mat_f32_rows_tiled<RB, TB, NT, TX><<<dim3((nt + TB - 1)/TB, (nrows + RB - 1)/RB), NT, 0, stream>>>( \
        w, x, d, (int) K, (int) nrows, nt, sx, sd)
    switch (sel) {
        case 1:  FR_TILED(8, 4, 128); break;   // e.g. 48 x 2560
        case 2:  FR_TILED(4, 2, 256); break;   // e.g. 4 x 10240
        case 3:  FR_TILED(8, 8, 64);  break;
        case 4:  FR_TILED(16, 4, 64); break;
        case 5:  FR_TILED(8, 4, 64);  break;
        case 6:  FR_TILED(4, 4, 128); break;
        case 7:  FR_TILED(4, 1, 256); break;
        case 8:  FR_TILED(4, 2, 128); break;
        default: return false;
    }
#undef FR_TILED
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// TB tokens per block: each weight float4 is loaded once and used for all TB tokens (per-token blocks re-read the
// whole R x K weight from cache for every token). A warp-per-token mapping was measured slower (48 rows: 209 vs 111 us).
template <int R, int TB, typename TX = float>
static __global__ void __launch_bounds__(MM_FEW_ROWS_THREADS)
mul_mat_f32_few_rows(const float * __restrict__ w, const TX * __restrict__ x, float * __restrict__ dst,
                     const int K, const int nrows, const int64_t ntok, const int64_t stride_x, const int64_t stride_dst) {
    const int64_t t0 = (int64_t) blockIdx.x*TB;
    const int K4 = K / 4;

    float acc[TB][R];
#pragma unroll
    for (int b = 0; b < TB; ++b) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[b][r] = 0.0f;
        }
    }

    for (int k = threadIdx.x; k < K4; k += MM_FEW_ROWS_THREADS) {
        float4 xv[TB];
#pragma unroll
        for (int b = 0; b < TB; ++b) {
            if (t0 + b >= ntok) {
                xv[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else if constexpr (std::is_same_v<TX, half>) {
                const half2 * h = (const half2 *) (x + (t0 + b)*stride_x) + 2*k;
                const float2 lo = __half22float2(h[0]), hi = __half22float2(h[1]);
                xv[b] = make_float4(lo.x, lo.y, hi.x, hi.y);
            } else {
                xv[b] = ((const float4 *) (x + (t0 + b)*stride_x))[k];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < nrows) {
                const float4 wv = ((const float4 *) (w + (int64_t) r*K))[k];
#pragma unroll
                for (int b = 0; b < TB; ++b) {
                    acc[b][r] += xv[b].x*wv.x + xv[b].y*wv.y + xv[b].z*wv.z + xv[b].w*wv.w;
                }
            }
        }
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = MM_FEW_ROWS_THREADS / warp_size;
    __shared__ float red[nwarps][TB*R];

    const int lane = threadIdx.x % warp_size;
    const int wid  = threadIdx.x / warp_size;
#pragma unroll
    for (int b = 0; b < TB; ++b) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = warp_reduce_sum<warp_size>(acc[b][r]);
            if (lane == 0) {
                red[wid][b*R + r] = v;
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < TB*R; i += MM_FEW_ROWS_THREADS) {
        const int b = i / R, r = i % R;
        if (r >= nrows || t0 + b >= ntok) {
            continue;
        }
        float s = 0.0f;
#pragma unroll
        for (int q = 0; q < nwarps; ++q) {
            s += red[q][i];
        }
        dst[(t0 + b)*stride_dst + r] = s;
    }
}

static bool ggml_cuda_few_rows_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool disabled = [] { const char * e = getenv("GGML_CUDA_NO_FEW_ROWS"); return e && atoi(e) != 0; }();
    const int64_t K     = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    const int64_t ntok  = src1->ne[1];
    return !disabled && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        nrows <= 64 && ntok > 8 && K % 4 == 0 && K <= INT_MAX &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 &&
        dst->nb[0] == sizeof(float) && ((uintptr_t) src0->data) % 16 == 0 && ((uintptr_t) src1->data) % 16 == 0;
}

// F32 x F32 GEMM for dst[n][m] = sum_k w[m][k] * x[n][k] with both operands K-contiguous (e.g. the MoE router,
// 512 x 2560): 64x64 tiles, 4x4 outputs per thread, K staged through shared memory in steps of 16. rocBLAS picked
// slow kernels for this shape (gfx906: 0.56 TFLOPS; gfx1151: 3.8 TFLOPS).
static constexpr int SGEMM_TN_BM = 64, SGEMM_TN_BN = 64, SGEMM_TN_BK = 16;

static __global__ void __launch_bounds__(256)
sgemm_tn_f32(const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ dst,
             const int M, const int N, const int K, const int64_t sw, const int64_t sx, const int64_t sd) {
    __shared__ float ws[SGEMM_TN_BK][SGEMM_TN_BM + 4];
    __shared__ float xs[SGEMM_TN_BK][SGEMM_TN_BN + 4];
    const int tid = threadIdx.x;
    const int tx  = tid % 16, ty = tid / 16;
    const int m0  = blockIdx.x*SGEMM_TN_BM, n0 = blockIdx.y*SGEMM_TN_BN;
    // loaders: row lr of the tile, 4 consecutive k at 4*lk
    const int lr = tid / 4, lk = tid % 4;
    const int wm = min(m0 + lr, M - 1), xn = min(n0 + lr, N - 1);
    const float * wp = w + (int64_t) wm*sw + 4*lk;
    const float * xp = x + (int64_t) xn*sx + 4*lk;

    float acc[4][4] = {};
    float4 wr = *(const float4 *) wp, xr = *(const float4 *) xp;
    for (int k0 = 0; k0 < K; k0 += SGEMM_TN_BK) {
        ws[4*lk + 0][lr] = wr.x; ws[4*lk + 1][lr] = wr.y; ws[4*lk + 2][lr] = wr.z; ws[4*lk + 3][lr] = wr.w;
        xs[4*lk + 0][lr] = xr.x; xs[4*lk + 1][lr] = xr.y; xs[4*lk + 2][lr] = xr.z; xs[4*lk + 3][lr] = xr.w;
        __syncthreads();
        if (k0 + SGEMM_TN_BK < K) {
            wr = *(const float4 *) (wp + k0 + SGEMM_TN_BK);
            xr = *(const float4 *) (xp + k0 + SGEMM_TN_BK);
        }
#pragma unroll
        for (int k = 0; k < SGEMM_TN_BK; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = ws[k][tx + 16*i];
                b[i] = xs[k][ty + 16*i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a[i]*b[j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = n0 + ty + 16*j;
        if (n >= N) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = m0 + tx + 16*i;
            if (m < M) {
                dst[(int64_t) n*sd + m] = acc[i][j];
            }
        }
    }
}

static bool ggml_cuda_mul_mat_sgemm_tn(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const int env = [] { const char * e = getenv("GGML_CUDA_SGEMM_TN"); return e ? atoi(e) : -1; }();
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    // on GCN (gfx906) rocBLAS is faster in steady state (router at 2048 tokens: 0.74 vs 0.87 ms)
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (env == 0 || (GGML_CUDA_CC_IS_GCN(cc) && env != 1) || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            M < 32 || N < 64 || K % SGEMM_TN_BK != 0 || K > INT_MAX || M > INT_MAX || N > INT_MAX ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src0->nb[1] % 16 != 0 || src1->nb[1] % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0 || ((uintptr_t) src1->data) % 16 != 0) {
        return false;
    }
    const dim3 grid((M + SGEMM_TN_BM - 1)/SGEMM_TN_BM, (N + SGEMM_TN_BN - 1)/SGEMM_TN_BN, 1);
    sgemm_tn_f32<<<grid, 256, 0, ctx.stream()>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        (int) M, (int) N, (int) K, src0->nb[1]/sizeof(float), src1->nb[1]/sizeof(float), dst->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// F16 activations (stored as F16 in place by their producer): each thread handles 8 consecutive K values per step
// (one 16-byte load per token), TB tokens per block
template <int R, int TB>
static __global__ void __launch_bounds__(MM_FEW_ROWS_THREADS)
mul_mat_f32_few_rows_x16(const float * __restrict__ w, const half * __restrict__ x, float * __restrict__ dst,
                         const int K, const int nrows, const int64_t ntok, const int64_t stride_dst, const int64_t ldx) {
    const int64_t t0 = (int64_t) blockIdx.x*TB;
    const int K8 = K / 8;
    float acc[TB][R];
#pragma unroll
    for (int b = 0; b < TB; ++b) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[b][r] = 0.0f;
        }
    }
    for (int k = threadIdx.x; k < K8; k += MM_FEW_ROWS_THREADS) {
        float xv[TB][8];
#pragma unroll
        for (int b = 0; b < TB; ++b) {
            if (t0 + b < ntok) {
                const uint4 u = ((const uint4 *) (x + (t0 + b)*ldx))[k];
                const half2 * h = (const half2 *) &u;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float2 f = __half22float2(h[q]);
                    xv[b][2*q] = f.x; xv[b][2*q + 1] = f.y;
                }
            } else {
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    xv[b][q] = 0.0f;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < nrows) {
                const float4 w0 = ((const float4 *) (w + (int64_t) r*K))[2*k];
                const float4 w1 = ((const float4 *) (w + (int64_t) r*K))[2*k + 1];
#pragma unroll
                for (int b = 0; b < TB; ++b) {
                    acc[b][r] += xv[b][0]*w0.x + xv[b][1]*w0.y + xv[b][2]*w0.z + xv[b][3]*w0.w +
                                 xv[b][4]*w1.x + xv[b][5]*w1.y + xv[b][6]*w1.z + xv[b][7]*w1.w;
                }
            }
        }
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = MM_FEW_ROWS_THREADS / warp_size;
    __shared__ float red[nwarps][TB*R];
    const int lane = threadIdx.x % warp_size;
    const int wid  = threadIdx.x / warp_size;
#pragma unroll
    for (int b = 0; b < TB; ++b) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = warp_reduce_sum<warp_size>(acc[b][r]);
            if (lane == 0) {
                red[wid][b*R + r] = v;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TB*R; i += MM_FEW_ROWS_THREADS) {
        const int b = i / R, r = i % R;
        if (r >= nrows || t0 + b >= ntok) {
            continue;
        }
        float s = 0.0f;
#pragma unroll
        for (int q = 0; q < nwarps; ++q) {
            s += red[q][i];
        }
        dst[(t0 + b)*stride_dst + r] = s;
    }
}

// Decode / MTP-verify batches (<= 8 tokens) of small F32 projections (HC inject 4 x 10240, GDN alpha/beta 48 x 2560):
// one block per weight row, K split over the block, all tokens at once. The generic paths take ~26 us per call on
// gfx906 for a few hundred KB.
template <int T>
static __global__ void __launch_bounds__(256)
mul_mat_f32_rows_small_t(const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ dst,
                         const int K, const int64_t stride_x, const int64_t stride_dst) {
    const int r = blockIdx.x;
    const float4 * wr = (const float4 *) (w + (int64_t) r*K);
    float acc[T] = {};
    for (int k = threadIdx.x; k < K/4; k += 256) {
        const float4 wv = wr[k];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xv = ((const float4 *) (x + t*stride_x))[k];
            acc[t] += xv.x*wv.x + xv.y*wv.y + xv.z*wv.z + xv.w*wv.w;
        }
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    __shared__ float red[256/warp_size][T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_reduce_sum<warp_size>(acc[t]);
        if (threadIdx.x % warp_size == 0) {
            red[threadIdx.x / warp_size][t] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 256/warp_size; ++i) {
            sum += red[i][threadIdx.x];
        }
        dst[threadIdx.x*stride_dst + r] = sum;
    }
}

// Split-K variant for few rows (HC inject 4 x 10240): one block per row left 4 blocks each walking 10 dependent float4
// iterations (8 us at 1 token, 21 us at 4 on gfx906). Grid (rows, KS): each block reduces its K chunk, the last block of
// a row (per-row arrival counter, reset by that block) sums the KS partials in a fixed order, so the result does not
// depend on block timing. Counters are per (virtual device, stream): streams of one GPU never share a slot.
static __device__ unsigned int g_rows_small_t_cnt[GGML_CUDA_MAX_DEVICES*GGML_CUDA_MAX_STREAMS][64];

template <int T>
static __global__ void __launch_bounds__(256)
mul_mat_f32_rows_small_t_splitk(const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ dst,
                                float * __restrict__ part, unsigned int * __restrict__ cnt,
                                const int K, const int64_t stride_x, const int64_t stride_dst) {
    const int r  = blockIdx.x;
    const int ks = blockIdx.y, KS = gridDim.y;
    const int K4 = K/4;
    const int chunk = (K4 + KS - 1)/KS;
    const int k0 = ks*chunk, k1 = min(K4, k0 + chunk);
    const float4 * wr = (const float4 *) (w + (int64_t) r*K);
    float acc[T] = {};
    for (int k = k0 + threadIdx.x; k < k1; k += 256) {
        const float4 wv = wr[k];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xv = ((const float4 *) (x + t*stride_x))[k];
            acc[t] += xv.x*wv.x + xv.y*wv.y + xv.z*wv.z + xv.w*wv.w;
        }
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    __shared__ float red[256/warp_size][T];
    __shared__ bool last;
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_reduce_sum<warp_size>(acc[t]);
        if (threadIdx.x % warp_size == 0) {
            red[threadIdx.x / warp_size][t] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 256/warp_size; ++i) {
            sum += red[i][threadIdx.x];
        }
        part[((int64_t) r*KS + ks)*T + threadIdx.x] = sum;
        __threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(&cnt[r], 1u) == (unsigned int) KS - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    if (threadIdx.x < T) {
        const volatile float * pv = part + (int64_t) r*KS*T + threadIdx.x;
        float sum = 0.0f;
        for (int i = 0; i < KS; ++i) {
            sum += pv[i*T];
        }
        dst[threadIdx.x*stride_dst + r] = sum;
    }
    if (threadIdx.x == 0) {
        cnt[r] = 0;
    }
}

static bool ggml_cuda_mul_mat_f32_rows_small_t(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool disabled = [] { const char * e = getenv("GGML_CUDA_NO_ROWS_SMALL_T"); return e && atoi(e) != 0; }();
    const int64_t K = src0->ne[0], nrows = src0->ne[1], ntok = src1->ne[1];
    if (disabled || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            nrows > 64 || ntok < 1 || ntok > 8 || K % 4 != 0 || K < 1024 || K > INT_MAX ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            !ggml_is_contiguous(src0) || src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || dst->nb[0] != sizeof(float) ||
            ((uintptr_t) src0->data) % 16 != 0 || ((uintptr_t) src1->data) % 16 != 0 || ggml_cuda_is_f16_inplace(ctx, src1)) {
        return false;
    }
    const float * w = (const float *) src0->data;
    const float * x = (const float *) src1->data;
    float       * d = (float *) dst->data;
    const int64_t sx = src1->nb[1] / sizeof(float), sd = dst->nb[1] / sizeof(float);
    cudaStream_t stream = ctx.stream();
    // split K when the rows alone give too few blocks (GGML_CUDA_ROWS_SMALL_T_KS=1: off, N: fixed split)
    static const int ks_env = [] { const char * e = getenv("GGML_CUDA_ROWS_SMALL_T_KS"); return e ? atoi(e) : 0; }();
    const int ks = ks_env > 0 ? ks_env : nrows > 8 || ntok < 2 ? 1 : (int) std::max<int64_t>(1, std::min<int64_t>((32 + nrows - 1)/nrows, (K/4 + 255)/256));
    if (ks > 1) {
        ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) nrows*ks*ntok);
        static unsigned int * cnt_base[GGML_CUDA_MAX_DEVICES] = {}; // per physical device (module globals are)
        const int phys = ggml_cuda_get_physical_device(ctx.device);
        if (cnt_base[phys] == nullptr) {
            CUDA_CHECK(cudaGetSymbolAddress((void **) &cnt_base[phys], (const void *) &g_rows_small_t_cnt));
        }
        unsigned int * cnt = cnt_base[phys];
        cnt += (size_t) (ctx.device*GGML_CUDA_MAX_STREAMS + ctx.curr_stream_no)*64;
        const dim3 grid((unsigned) nrows, (unsigned) ks);
#define ROWS_SMALL_T_SPLITK(T) mul_mat_f32_rows_small_t_splitk<T><<<grid, 256, 0, stream>>>(w, x, d, part.get(), cnt, (int) K, sx, sd)
        switch (ntok) {
            case 1: ROWS_SMALL_T_SPLITK(1); break;
            case 2: ROWS_SMALL_T_SPLITK(2); break;
            case 3: ROWS_SMALL_T_SPLITK(3); break;
            case 4: ROWS_SMALL_T_SPLITK(4); break;
            case 5: ROWS_SMALL_T_SPLITK(5); break;
            case 6: ROWS_SMALL_T_SPLITK(6); break;
            case 7: ROWS_SMALL_T_SPLITK(7); break;
            default: ROWS_SMALL_T_SPLITK(8); break;
        }
#undef ROWS_SMALL_T_SPLITK
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    switch (ntok) {
        case 1: mul_mat_f32_rows_small_t<1><<<nrows, 256, 0, stream>>>(w, x, d, (int) K, sx, sd); break;
        case 2: mul_mat_f32_rows_small_t<2><<<nrows, 256, 0, stream>>>(w, x, d, (int) K, sx, sd); break;
        case 3: mul_mat_f32_rows_small_t<3><<<nrows, 256, 0, stream>>>(w, x, d, (int) K, sx, sd); break;
        case 4: mul_mat_f32_rows_small_t<4><<<nrows, 256, 0, stream>>>(w, x, d, (int) K, sx, sd); break;
        case 5: mul_mat_f32_rows_small_t<5><<<nrows, 256, 0, stream>>>(w, x, d, (int) K, sx, sd); break;
        case 6: mul_mat_f32_rows_small_t<6><<<nrows, 256, 0, stream>>>(w, x, d, (int) K, sx, sd); break;
        case 7: mul_mat_f32_rows_small_t<7><<<nrows, 256, 0, stream>>>(w, x, d, (int) K, sx, sd); break;
        default: mul_mat_f32_rows_small_t<8><<<nrows, 256, 0, stream>>>(w, x, d, (int) K, sx, sd); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// returns false if the shapes/layouts are not handled (the caller falls back)
static bool ggml_cuda_mul_mat_f32_few_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_few_rows_ok(src0, src1, dst)) {
        return false;
    }
    const int64_t K     = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    const int64_t ntok  = src1->ne[1];
    const float * w = (const float *) src0->data;
    float       * d = (float *) dst->data;
    const int64_t sd = dst->nb[1]  / sizeof(float);
    cudaStream_t stream = ctx.stream();
    if (ggml_cuda_is_f16_inplace(ctx, src1)) { // stored as F16 in place by its producer (contiguous rows of K halves)
        GGML_ASSERT(ggml_is_contiguous(src1));
        const half * x16 = (const half *) src1->data;
        const int64_t ldx = ggml_cuda_f16_inplace_ld(ctx, src1, K);
        GGML_ASSERT(K % 8 == 0 && ldx % 8 == 0);
        if (ggml_cuda_few_rows_tiled<half>(w, x16, d, K, nrows, ntok, ldx, sd, stream)) {
            return true;
        }
#define FEW_ROWS_CASE16(R, TB) mul_mat_f32_few_rows_x16<R, TB><<<(ntok + TB - 1)/TB, MM_FEW_ROWS_THREADS, 0, stream>>>( \
        w, x16, d, (int) K, (int) nrows, ntok, sd, ldx)
        if (nrows <= 8) {
            FEW_ROWS_CASE16(8, 4);
        } else if (nrows <= 16) {
            FEW_ROWS_CASE16(16, 4);
        } else if (nrows <= 32) {
            FEW_ROWS_CASE16(32, 2);
        } else if (nrows <= 48) {
            FEW_ROWS_CASE16(48, 2);
        } else {
            FEW_ROWS_CASE16(64, 2);
        }
#undef FEW_ROWS_CASE16
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    const float * x = (const float *) src1->data;
    const int64_t sx = src1->nb[1] / sizeof(float);
    if (ggml_cuda_few_rows_tiled<float>(w, x, d, K, nrows, ntok, sx, sd, stream)) {
        return true;
    }
#define FEW_ROWS_CASE(R, TB) mul_mat_f32_few_rows<R, TB><<<(ntok + TB - 1)/TB, MM_FEW_ROWS_THREADS, 0, stream>>>( \
        w, x, d, (int) K, (int) nrows, ntok, sx, sd)
    if (nrows <= 8) {
        FEW_ROWS_CASE(8, 4);
    } else if (nrows <= 16) {
        FEW_ROWS_CASE(16, 4);
    } else if (nrows <= 32) {
        FEW_ROWS_CASE(32, 2);
    } else if (nrows <= 48) {
        FEW_ROWS_CASE(48, 2);
    } else {
        FEW_ROWS_CASE(64, 2);
    }
#undef FEW_ROWS_CASE
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_mul_mat_f32_rows_small_t(ctx, src0, src1, dst)) {
        return;
    }

    // src1 stored as F16 in place: its producer verified that this few-rows path takes it
    if (ggml_cuda_is_f16_inplace(ctx, src1)) {
        const bool ok = ggml_cuda_mul_mat_f32_few_rows(ctx, src0, src1, dst);
        GGML_ASSERT(ok);
        return;
    }

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    if (ggml_cuda_mm_tile_f_supported(src0, src1, dst, ggml_cuda_info().devices[ctx.device].cc)) {
        ggml_cuda_mm_tile_f(ctx, src0, src1, dst);
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    // GCN: F16 weights at 1..8 tokens (coalesced 16-byte loads, f32 sums; MMVF reaches ~200 GB/s there)
    if (ggml_cuda_gcn_f16_matvec(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_gcn_f32_matvec(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    // GCN, 9..32 tokens of an F32 weight (the MoE router, 288 x 4096 for GLM-5.3): MMVF over column slices of <= 8
    // instead of the tiled SGEMM (decode profile at 12 sequences: router 7.9 s vs 2.4 s at 8; GGML_CUDA_GCN_MMVF_SPLIT=0: off)
    static const int gcn_mmvf_split = [] { const char * e = getenv("GGML_CUDA_GCN_MMVF_SPLIT"); return e ? atoi(e) : 1; }();
    if (gcn_mmvf_split && GGML_CUDA_CC_IS_GCN(cc) && src0->type == GGML_TYPE_F32 && ne11 > MMVF_MAX_BATCH_SIZE &&
            ne11 <= 4*MMVF_MAX_BATCH_SIZE && ne12 == 1 && ne13 == 1 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
            ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, MMVF_MAX_BATCH_SIZE)) {
        ggml_tensor s1 = *src1;
        ggml_tensor d  = *dst;
        for (int64_t c0 = 0; c0 < ne11; c0 += MMVF_MAX_BATCH_SIZE) {
            const int64_t nc = std::min<int64_t>(MMVF_MAX_BATCH_SIZE, ne11 - c0);
            s1.ne[1] = nc;
            s1.data  = (char *) src1->data + c0*src1->nb[1];
            d.ne[1]  = nc;
            d.data   = (char *) dst->data + c0*dst->nb[1];
            ggml_cuda_mul_mat_vec_f(ctx, src0, &s1, nullptr, &d);
        }
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_gcn_q8_matvec(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_gemv1_q2k_supported(cc, src0, src1, dst)) {
        ggml_cuda_gemv1_q2k(ctx, src0, src1, dst);
        return;
    }
    if (ggml_cuda_rowlane_dense_supported(cc, src0, src1, dst)) {
        ggml_cuda_rowlane_dense(ctx, src0, src1, dst);
        return;
    }
    // GCN: q4_K / q5_K / q6_K / iq4_xs at 1..4 tokens (coalesced quant runs; MMVQ ~400 GB/s there)
    if (ggml_cuda_gcn_kq_matvec_supported(cc, src0, src1, dst)) {
        ggml_cuda_gcn_kq_matvec(ctx, src0, src1, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    // GCN, 9..16 tokens (e.g. DFlash2 verification of 2 sequences) of q5_K / iq4_xs: MMVQ over column slices of <= 8
    // beats MMQ (8704x5120 at 16 tokens: q5_K 405 -> 2 x 186 us, iq4_xs 326 -> 2 x 130 us; q4_K, q6_K, iq4_nl lose)
    static const int gcn_mmvq_split = [] { const char * e = getenv("GGML_CUDA_GCN_MMVQ_SPLIT"); return e ? atoi(e) : 1; }();
    if (gcn_mmvq_split && GGML_CUDA_CC_IS_GCN(cc) && ne11 > MMVQ_MAX_BATCH_SIZE && ne11 <= 2*MMVQ_MAX_BATCH_SIZE &&
            (src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_IQ4_XS) && ne12 == 1 && ne13 == 1 &&
            src0->ne[2] == 1 && src0->ne[3] == 1 && ggml_cuda_should_use_mmvq(src0->type, cc, MMVQ_MAX_BATCH_SIZE)) {
        ggml_tensor s1 = *src1;
        ggml_tensor d  = *dst;
        ggml_cuda_q8_cache_bypass = true; // the q8_1 activation cache is keyed by the tensor pointer: &s1 is reused
        for (int64_t c0 = 0; c0 < ne11; c0 += MMVQ_MAX_BATCH_SIZE) {
            const int64_t nc = std::min<int64_t>(MMVQ_MAX_BATCH_SIZE, ne11 - c0);
            s1.ne[1] = nc;
            s1.data  = (char *) src1->data + c0*src1->nb[1];
            d.ne[1]  = nc;
            d.data   = (char *) dst->data + c0*dst->nb[1];
            ggml_cuda_mul_mat_vec_q(ctx, src0, &s1, nullptr, &d);
        }
        ggml_cuda_q8_cache_bypass = false;
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    // 32..64 rows at large batch (GDN alpha/beta, 48 x 2560): the tiled SGEMM beats the few-rows kernel
    if (src0->ne[1] >= 32 && src1->ne[1] >= 256 && ggml_cuda_mul_mat_sgemm_tn(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_mul_mat_f32_few_rows(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_mul_mat_sgemm_tn(ctx, src0, src1, dst)) {
        return;
    }

    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // src1 stored as F16 in place by the fused gate/up kernel, which verified this F16 MoE path takes it
    if (ggml_cuda_is_f16_inplace(ctx, src1)) {
        GGML_ASSERT(ggml_cuda_moe_f16_supported(cc, src0, src1, ids, dst));
        ggml_cuda_moe_f16(ctx, src0, src1, ids, dst);
        return;
    }

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        // GCN decode: the K-split row-lane kernel beats MMVQ (short K rows leave most MMVQ lanes idle)
        if (ggml_cuda_moe_vec_decode_supported(cc, src0, src1, ids, dst)) {
            ggml_cuda_moe_vec(ctx, src0, src1, ids, dst);
            return;
        }
        // q3_K experts with short K (DeepSeek V4.1 down at TP4): the row-lane tiles beat MMVQ from 1 token; q2_K from the
        // token count its threshold allows (K-split row-lane at decode sizes)
        if ((src0->type == GGML_TYPE_Q3_K || src0->type == GGML_TYPE_Q2_K) && ggml_cuda_moe_vec_supported(cc, src0, src1, ids, dst)) {
            ggml_cuda_moe_vec(ctx, src0, src1, ids, dst);
            return;
        }
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        // prefill: K-quant experts grouped by expert on the GCN v4 GEMM tiles
        if (ggml_cuda_gcn_kq_moe_supported(cc, src0, src1, ids, dst)) {
            ggml_cuda_gcn_kq_moe(ctx, src0, src1, ids, dst);
            return;
        }
        if (ggml_cuda_moe_vec_supported(cc, src0, src1, ids, dst)) {
            ggml_cuda_moe_vec(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward_impl(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);

// GGML_CUDA_DUMP_NAME=prefix[,prefix2] (debug, graphs off): after computing a node whose name starts with a prefix, print
// the device, sum, sum of squares and first values of its (F32) result
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const bool ok = ggml_cuda_compute_forward_impl(ctx, dst);
    static const char * names = getenv("GGML_CUDA_DUMP_NAME");
    if (names && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst)) {
        bool match = false;
        for (const char * p = names; *p; ) {
            const char * q = strchr(p, ',');
            const size_t len = q ? (size_t) (q - p) : strlen(p);
            match = match || (len > 0 && strncmp(dst->name, p, len) == 0);
            p = q ? q + 1 : p + len;
        }
        if (match) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            std::vector<float> v(ggml_nelements(dst));
            CUDA_CHECK(cudaMemcpy(v.data(), dst->data, v.size()*sizeof(float), cudaMemcpyDeviceToHost));
            double sum = 0.0, sq = 0.0;
            for (float f : v) { sum += f; sq += (double) f*f; }
            fprintf(stderr, "DUMP dev %d %-28s [%lld,%lld,%lld] sum %.6g sq %.6g first %.6g %.6g %.6g %.6g\n", ctx.device, dst->name,
                (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], sum, sq,
                v.size() > 0 ? v[0] : 0.0f, v.size() > 1 ? v[1] : 0.0f, v.size() > 2 ? v[2] : 0.0f, v.size() > 3 ? v[3] : 0.0f);
        }
    }
    return ok;
}

static bool ggml_cuda_compute_forward_impl(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_MIX:
            ggml_cuda_op_dsv4_hc_mix(ctx, dst);
            break;
        case GGML_OP_DSV4_SPARSE_ATTN:
            ggml_cuda_dsv4_sparse_attn(ctx, dst);
            break;
        case GGML_OP_DSV4_COMP_POOL:
            ggml_cuda_dsv4_comp_pool(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_MEAN:
            ggml_cuda_op_get_rows_mean(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

// Host<->device copies of pinned memory as a kernel on the stream's own queue (GGML_CUDA_KERNEL_COPY=1): the DMA
// engines are shared by all streams of a device, so a copy that waits on another device blocks every copy queued
// behind it on that engine, e.g. the input uploads of an independent pipeline stage on the same GPU.
static __global__ void ggml_cuda_copy_bytes(uint8_t * __restrict__ dst, const uint8_t * __restrict__ src, const size_t n) {
    const size_t n16 = n / 16;
    const size_t stride = (size_t) gridDim.x*blockDim.x;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n16; i += stride) {
        ((uint4 *) dst)[i] = ((const uint4 *) src)[i];
    }
    for (size_t i = n16*16 + (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// default on for GCN; GGML_CUDA_KERNEL_COPY=0/1 forces it
static bool ggml_cuda_kernel_copy_enabled() {
    static const int env = [] { const char * e = getenv("GGML_CUDA_KERNEL_COPY"); return e ? atoi(e) : -1; }();
    if (env >= 0) {
        return env != 0;
    }
    return GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

static bool ggml_cuda_is_pinned(const void * p) {
#if defined(GGML_USE_HIP)
    hipPointerAttribute_t attr;
    if (hipPointerGetAttributes(&attr, p) != hipSuccess) {
        (void) hipGetLastError();
        return false;
    }
    return attr.type == hipMemoryTypeHost;
#else
    GGML_UNUSED(p);
    return false;
#endif
}

static void ggml_cuda_kernel_copy(void * dst, const void * src, size_t n, cudaStream_t stream) {
    const bool aligned = ((uintptr_t) dst % 16 == 0) && ((uintptr_t) src % 16 == 0);
    if (!aligned) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, stream)); // (unaligned: device pointers or mapped pinned memory)
        return;
    }
    const int nblocks = (int) std::min<size_t>(120, (n/16 + 255)/256 + 1);
    ggml_cuda_copy_bytes<<<nblocks, 256, 0, stream>>>((uint8_t *) dst, (const uint8_t *) src, n);
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (ggml_cuda_kernel_copy_enabled() && ggml_cuda_is_pinned(data)) {
        ggml_cuda_set_device(cuda_ctx->device);
        ggml_cuda_kernel_copy((char *) tensor->data + offset, data, size, cuda_ctx->stream());
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (ggml_cuda_kernel_copy_enabled() && ggml_cuda_is_pinned(data)) {
        ggml_cuda_set_device(cuda_ctx->device);
        ggml_cuda_kernel_copy(data, (const char *) tensor->data + offset, size, cuda_ctx->stream());
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

// GGML_CUDA_TIMELINE=1: host-clock timeline of every graph compute (enqueue span on the host, execution span on the GPU
// via stream host callbacks), printed at exit, to see pipeline gaps across devices
struct ggml_cuda_timeline {
    struct entry { int dev; int n_nodes; int64_t h0, h1; std::atomic<int64_t> g0{0}, g1{0}; };
    std::deque<entry> entries;
    std::mutex mtx;
    bool enabled = false;
    size_t n_print = 200; // GGML_CUDA_TIMELINE=N > 1: print the last N graphs
    ggml_cuda_timeline() { const char * e = getenv("GGML_CUDA_TIMELINE"); enabled = e && atoi(e) != 0; if (enabled && atoi(e) > 1) { n_print = atoi(e); } }
    ~ggml_cuda_timeline() {
        if (entries.empty()) {
            return;
        }
        const size_t n = entries.size();
        const size_t first = n > n_print ? n - n_print : 0;
        const int64_t t0 = entries[first].h0;
        fprintf(stderr, "ggml_cuda_timeline (ms, last %zu graphs): dev nodes  enq0  enq1  gpu0  gpu1\n", n - first);
        for (size_t i = first; i < n; ++i) {
            const auto & e = entries[i];
            fprintf(stderr, "TL %d %5d %8.2f %8.2f %8.2f %8.2f\n", e.dev, e.n_nodes, (e.h0 - t0)/1e3, (e.h1 - t0)/1e3,
                    (e.g0.load() - t0)/1e3, (e.g1.load() - t0)/1e3);
        }
    }
    static int64_t now() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    static void cb_start(void * p) { ((entry *) p)->g0 = now(); }
    static void cb_end(void * p)   { ((entry *) p)->g1 = now(); }
};
static ggml_cuda_timeline g_timeline;

// Device -> device copy staged through pinned host memory, fully asynchronous for the host: D2H on the source stream,
// the destination stream waits on an event, H2D on the destination stream. Without a GPU-to-GPU path (e.g. MI50 over
// PCIe without P2P) cudaMemcpyPeerAsync blocks the host until the source device is idle, which serializes the
// scheduler's pipeline parallelism (the host cannot queue the next ubatch on the first GPU). A small ring of staging
// slots is reused; a slot is refilled only after its previous H2D finished.
struct ggml_cuda_staging_slot {
    void *      ptr     = nullptr;
    size_t      cap     = 0;
    cudaEvent_t done    = nullptr; // recorded on the destination stream after the H2D
    int         done_dev = -1;
};
// one ring per destination device: a slot's event then always lives on that device and reusing it never needs a
// host-side wait (a shared ring handing slots to another device synchronized on every such change)
static constexpr int          GGML_CUDA_STAGING_SLOTS = 16;
static std::mutex             g_cuda_staging_mutex;
static ggml_cuda_staging_slot g_cuda_staging[GGML_CUDA_MAX_DEVICES][GGML_CUDA_STAGING_SLOTS];
static int                    g_cuda_staging_next[GGML_CUDA_MAX_DEVICES] = {};

static bool ggml_cuda_use_staged_copy(int src_device) {
    static const int env = [] { const char * e = getenv("GGML_CUDA_STAGED_COPY"); return e ? atoi(e) : -1; }();
    if (env >= 0) {
        return env != 0;
    }
    return GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[src_device].cc);
}

static void ggml_cuda_staged_copy(ggml_backend_cuda_context * ctx_src, ggml_backend_cuda_context * ctx_dst,
                                  const ggml_tensor * src, ggml_tensor * dst) {
    const size_t n = ggml_nbytes(dst);
    std::lock_guard<std::mutex> lock(g_cuda_staging_mutex);
    ggml_cuda_staging_slot & slot = g_cuda_staging[ctx_dst->device][g_cuda_staging_next[ctx_dst->device]];
    g_cuda_staging_next[ctx_dst->device] = (g_cuda_staging_next[ctx_dst->device] + 1) % GGML_CUDA_STAGING_SLOTS;

    if (slot.cap < n) {
        // pinned (de)allocations are slow and cudaFreeHost synchronizes: size slots like the largest copy seen for
        // this destination (at least 16 MiB) so a slot first used by a small ubatch does not have to grow later
        static size_t max_seen[GGML_CUDA_MAX_DEVICES] = {};
        max_seen[ctx_dst->device] = std::max({max_seen[ctx_dst->device], n, (size_t) 16 << 20});
        if (slot.ptr) {
            if (slot.done) {
                CUDA_CHECK(cudaEventSynchronize(slot.done));
            }
            CUDA_CHECK(cudaFreeHost(slot.ptr));
        }
        CUDA_CHECK(cudaMallocHost(&slot.ptr, max_seen[ctx_dst->device]));
        slot.cap = max_seen[ctx_dst->device];
    }
    if (slot.done && slot.done_dev != ctx_dst->device) {
        CUDA_CHECK(cudaEventSynchronize(slot.done));
        ggml_cuda_set_device(slot.done_dev);
        CUDA_CHECK(cudaEventDestroy(slot.done));
        slot.done = nullptr;
    }

    ggml_cuda_set_device(ctx_src->device);
    const bool user_input = (src->flags & GGML_TENSOR_FLAG_INPUT) != 0;
    if (user_input) {
        // a graph input written by the user (never by the compute stream): read it now on a side stream and wait for
        // that copy only, since the user overwrites it for the next ubatch right after this graph is queued
        static cudaStream_t side[GGML_CUDA_MAX_DEVICES] = {};
        if (!side[ctx_src->device]) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&side[ctx_src->device], cudaStreamNonBlocking));
        }
        if (slot.done) {
            CUDA_CHECK(cudaEventSynchronize(slot.done));
        }
        CUDA_CHECK(cudaMemcpyAsync(slot.ptr, src->data, n, cudaMemcpyDeviceToHost, side[ctx_src->device]));
        CUDA_CHECK(cudaStreamSynchronize(side[ctx_src->device]));
    } else {
        if (slot.done) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx_src->stream(), slot.done, 0)); // the previous H2D from this slot is done
        }
        if (ggml_cuda_kernel_copy_enabled()) {
            ggml_cuda_kernel_copy(slot.ptr, src->data, n, ctx_src->stream());
        } else {
            CUDA_CHECK(cudaMemcpyAsync(slot.ptr, src->data, n, cudaMemcpyDeviceToHost, ctx_src->stream()));
        }
        if (!ctx_src->copy_event) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx_src->copy_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(ctx_src->copy_event, ctx_src->stream()));
    }

    ggml_cuda_set_device(ctx_dst->device);
    cudaStream_t h2d_stream = ctx_dst->stream();
    if (!user_input) {
        CUDA_CHECK(cudaStreamWaitEvent(h2d_stream, ctx_src->copy_event, 0));
    }
    ggml_cuda_timeline::entry * tle = nullptr;
    if (g_timeline.enabled) {
        std::lock_guard<std::mutex> lock(g_timeline.mtx);
        tle = &g_timeline.entries.emplace_back();
        tle->dev = 100 + ctx_dst->device; // copies show as dev 100+d, n_nodes = source device
        tle->n_nodes = ctx_src->device;
        tle->h0 = tle->h1 = ggml_cuda_timeline::now();
        CUDA_CHECK(cudaLaunchHostFunc(h2d_stream, ggml_cuda_timeline::cb_start, tle));
    }
    if (ggml_cuda_kernel_copy_enabled()) {
        ggml_cuda_kernel_copy(dst->data, slot.ptr, n, h2d_stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, slot.ptr, n, cudaMemcpyHostToDevice, h2d_stream));
    }
    if (tle) {
        CUDA_CHECK(cudaLaunchHostFunc(h2d_stream, ggml_cuda_timeline::cb_end, tle));
    }
    if (!slot.done) {
        CUDA_CHECK(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming));
        slot.done_dev = ctx_dst->device;
    }
    CUDA_CHECK(cudaEventRecord(slot.done, h2d_stream));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            // on the destination stream after the source's work: ordered after the destination's earlier reads of dst
            // (the scheduler overwrites a split's input copies while the previous graph may still be queued there)
            if (!cuda_ctx_src->copy_event) {
                ggml_cuda_set_device(cuda_ctx_src->device);
                CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
            }
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));
            ggml_cuda_set_device(cuda_ctx_dst->device);
            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_dst->stream()));
            return true;
        } else if (ggml_cuda_use_staged_copy(cuda_ctx_src->device)) {
            ggml_cuda_staged_copy(cuda_ctx_src, cuda_ctx_dst, src, dst);
            return true;
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// opt-in GGML_CUDA_GRAPH_KEY_SHAPE=1: the first node plus the graph shape and every leaf pointer, so a context that
// alternates between batch shapes (the MTP drafter: multi-token catch-up, then 1-token drafts) or rotates its split inputs
// through the scheduler pipeline copies keeps one captured graph per variant instead of resetting a shared one (a
// collision only costs a reset: the node properties are still compared). Off by default: GLM TP decode 63.3 -> 59.2 t/s
// with it (8x the graph instances, replayed in rotation), MTP +5%
static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    static const bool shape = [] { const char * e = getenv("GGML_CUDA_GRAPH_KEY_SHAPE"); return e && atoi(e) != 0; }();
    if (!shape) {
        return cgraph->nodes[0];
    }
    const ggml_tensor * n0 = cgraph->nodes[0];
    uint64_t h = (uint64_t) (uintptr_t) n0;
    auto mix = [&h](uint64_t v) { h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); };
    mix((uint64_t) cgraph->n_nodes);
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        mix((uint64_t) n0->ne[k]);
    }
    mix((uint64_t) (uintptr_t) cgraph->nodes[cgraph->n_nodes - 1]);
    // the scheduler rotates split inputs (leaves of the split graph) through its pipeline copies
    // (GGML_SCHED_MAX_COPIES): one graph per copy
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        for (int k = 0; k < GGML_MAX_SRC && t->src[k]; ++k) {
            if (t->src[k]->op == GGML_OP_NONE) {
                mix((uint64_t) (uintptr_t) t->src[k]->data);
            }
        }
    }
    return (const void *) (uintptr_t) h;
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            // GGML_CUDA_GRAPH_DIFF=N (debug): the first changed node of the first N changed graphs, and what changed
            static int diff_left = [] { const char * e = getenv("GGML_CUDA_GRAPH_DIFF"); return e ? atoi(e) : 0; }();
            if (!res && diff_left > 0) {
                --diff_left;
                const ggml_tensor & a = graph->node_props[i].node;
                const ggml_tensor & b = prop.node;
                const char * what = a.data != b.data ? "data" : memcmp(a.ne, b.ne, sizeof(a.ne)) ? "ne" :
                    memcmp(a.nb, b.nb, sizeof(a.nb)) ? "nb" : memcmp(a.op_params, b.op_params, sizeof(a.op_params)) ? "op_params" :
                    a.view_offs != b.view_offs ? "view_offs" : a.view_src != b.view_src ? "view_src" : a.op != b.op ? "op" : "";
                int js = -1;
                for (int j = 0; j < GGML_MAX_SRC && !*what; ++j) {
                    if (graph->node_props[i].node_src_data_ptrs[j] != prop.node_src_data_ptrs[j]) { what = "src data"; js = j; }
                    else if (memcmp(graph->node_props[i].node_src_ne[j], prop.node_src_ne[j], sizeof(prop.node_src_ne[j]))) { what = "src ne"; js = j; }
                    else if (memcmp(graph->node_props[i].node_src_nb[j], prop.node_src_nb[j], sizeof(prop.node_src_nb[j]))) { what = "src nb"; js = j; }
                }
                fprintf(stderr, "graph diff dev %d: %d nodes, node %d %s '%s' [%lld,%lld,%lld,%lld] changed: %s %d (src '%s')\n",
                        cuda_ctx->device, cgraph->n_nodes, i, ggml_op_desc(cgraph->nodes[i]), cgraph->nodes[i]->name,
                        (long long) b.ne[0], (long long) b.ne[1], (long long) b.ne[2], (long long) b.ne[3],
                        *what ? what : "other", js, js >= 0 && cgraph->nodes[i]->src[js] ? cgraph->nodes[i]->src[js]->name : "");
            }
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset: norm mode only (the rotated pairs are [n_offs, n_offs + n_dims))
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    static const bool offs_ok = [] { const char * e = getenv("GGML_CUDA_NORM_ROPE_OFFS"); return !e || atoi(e) != 0; }();
    if (n_offs != 0 && (!offs_ok || mode != GGML_ROPE_TYPE_NORMAL || n_offs % 2 != 0 || n_offs + n_dims > rope->src[0]->ne[0])) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    // the reduction kernels take the token from blockIdx.y (at most 65535)
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first) ||
            first->ne[2]*first->ne[3] > 65535) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            // a quantized MUL_MAT pair runs on MMVQ, which reads the q8_1 copy of src1 made before the kernel: the GLU
            // output may lie over the then dead src1 (the allocator puts it there at decode, e.g. the GLM shared expert)
            static const bool mmvq_alias = [] { const char * e = getenv("GGML_CUDA_GLU_MMVQ_ALIAS"); return !e || atoi(e) != 0; }();
            if (mmvq_alias && is_equal(mul_mat_glu_ops, ops) && ggml_is_quantized(ffn_up->src[0]->type) &&
                    (ggml_cuda_should_fuse_mul_mat_vec_q(ffn_up) || ggml_cuda_should_fuse_mmvq_glu_cols(ffn_up)) &&
                    !ggml_cuda_should_fuse_mul_mat_vec_f(ffn_up)) {
                return true;
            }
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        // other: the unary's shape, or one row broadcast over all of the unary's rows (GDN gates at a multi-token verify:
        // softplus(alpha + dt_bias) [heads, tokens] * A [heads]; GGML_CUDA_UNARY_MUL_BCAST=0 off)
        static const bool bcast_env = [] { const char * e = getenv("GGML_CUDA_UNARY_MUL_BCAST"); return !e || atoi(e) != 0; }();
        const bool other_row = bcast_env && ggml_nrows(other) == 1 && other->ne[0] == unary->ne[0] && ggml_are_same_shape(mul, unary);
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || (!ggml_are_same_shape(other, unary) && !other_row)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// try and fuse nodes and return the number of nodes to skip
// True if every consumer of the tensor at node mul_idx (directly or through contiguous RESHAPE/VIEW nodes) reads it
// through an F16-capable path: the fused down projection mm, a DSV4_HC_PRE fused into its up projection
// (hc_up_mix), or the F32 few-rows matmul. Then the producer may store it as F16 in place and skip the F32 copy.
static bool ggml_cuda_xn_consumers_f16_ok(int cc, const ggml_cgraph * cgraph, int mul_idx, const ggml_tensor * mm) {
    static const bool disabled = [] { const char * e = getenv("GGML_HC_NO_F16_INPLACE"); return e && atoi(e) != 0; }();
    if (disabled) {
        return false;
    }
    if (cgraph->nodes[mul_idx]->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false; // may be read outside this graph
    }
    std::vector<const ggml_tensor *> views = { cgraph->nodes[mul_idx] };
    int expected = ggml_node_get_use_count(cgraph, mul_idx);
    int found    = 0;
    for (int j = mul_idx + 1; j < std::min(cgraph->n_nodes, mul_idx + 512) && found < expected; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (!t->src[k] || std::find(views.begin(), views.end(), t->src[k]) == views.end()) {
                continue;
            }
            found++;
            if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) {
                if (!ggml_is_contiguous(t) || t->view_offs != 0 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    return false;
                }
                views.push_back(t);
                expected += ggml_node_get_use_count(cgraph, j);
            } else if (t == mm) {
                // fused into the producer
            } else if (t->op == GGML_OP_DSV4_HC_PRE && k == 0 && j >= 2 &&
                       cgraph->nodes[j - 2]->op == GGML_OP_MUL_MAT && cgraph->nodes[j - 1]->op == GGML_OP_RESHAPE &&
                       cgraph->nodes[j - 1]->src[0] == cgraph->nodes[j - 2] && t->src[1] == cgraph->nodes[j - 1] &&
                       ggml_can_fuse_subgraph(cgraph, j - 2, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE }, { j }) &&
                       ggml_cuda_hc_up_mix_supported(cc, cgraph->nodes[j - 2], t)) {
                // hc_up_mix reads xn as F16
            } else if (t->op == GGML_OP_MUL_MAT && k == 1 && ggml_is_contiguous(t->src[1]) &&
                       ggml_cuda_few_rows_ok(t->src[0], t->src[1], t)) {
                // few-rows matmul reads F16
            } else {
                return false;
            }
        }
    }
    return found == expected;
}

// A deferred GET_ROWS reads its row ids only when its consumer runs, but the allocator treats the GET_ROWS as their
// last reader: it may place the outputs of the nodes in between (or of the consumer) over the ids' memory (gfx1151
// server: q/k of the GDN over s_copy -> garbage row index -> aperture violation). A write over the ids by nodes
// i+1..j rejects the deferral. (Writes elsewhere in the cache tensor are not checked: other rows, e.g. the rollback
// snapshots, are written in between by design; a whole-tensor check rejected every MI50 deferral, -4% decode.)
static bool ggml_cuda_deferred_rows_intact(const ggml_cgraph * cgraph, int i, int j) {
    const ggml_tensor * g = cgraph->nodes[i];
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    for (int k = i + 1; k <= j; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (ggml_op_is_empty(t->op) || t->data == nullptr) {
            continue;
        }
        if (overlap(t, g->src[1])) {
            return false;
        }
    }
    return true;
}

// GET_ROWS of a recurrent state whose only consumer (through a RESHAPE) is a decode GDN that can read the cache row
// itself: registers the GDN and returns true (the gather is not run)
static bool ggml_cuda_defer_gdn_state_rows(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * g = cgraph->nodes[i];
    if (g->op != GGML_OP_GET_ROWS || ggml_node_get_use_count(cgraph, i) != 1 || (g->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const ggml_tensor * via = nullptr;
    for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 96); ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (!via && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->src[0] == g && ggml_node_get_use_count(cgraph, j) == 1 &&
                !(t->flags & GGML_TENSOR_FLAG_OUTPUT) && t->view_offs == 0) {
            via = t;
            continue;
        }
        if (t->op == GGML_OP_GATED_DELTA_NET && (t->src[5] == g || (via && t->src[5] == via))) {
            if (!ggml_cuda_gdn_state_rows_ok(g, t) || !ggml_cuda_deferred_rows_intact(cgraph, i, j)) {
                return false;
            }
            cuda_ctx->gdn_state_rows.push_back({t, (const float *) g->src[0]->data, (const int32_t *) g->src[1]->data});
            return true;
        }
    }
    return false;
}

// conv-state gather (GET_ROWS of the conv cache by the sequence id, reshaped into the concat's state operand): deferred
// into the GDN conv block kernel, which reads the cache row directly (one sequence)
static bool ggml_cuda_defer_conv_state_rows(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool fusion_off = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    static const bool disabled = [] { const char * e = getenv("GGML_GDN_CONV_ROWS"); return e && atoi(e) == 0; }();
    const ggml_tensor * g = cgraph->nodes[i];
    if (fusion_off || disabled || g->op != GGML_OP_GET_ROWS || ggml_node_get_use_count(cgraph, i) != 1 ||
            (g->flags & GGML_TENSOR_FLAG_OUTPUT) || g->type != GGML_TYPE_F32 || g->src[0]->type != GGML_TYPE_F32 ||
            g->src[1]->type != GGML_TYPE_I32 || g->src[1]->ne[0] != 1 || g->ne[1] != 1 || g->src[0]->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(g)) {
        return false;
    }
    const ggml_tensor * via = nullptr;
    for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 96); ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (!via && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->src[0] == g && ggml_node_get_use_count(cgraph, j) == 1 &&
                !(t->flags & GGML_TENSOR_FLAG_OUTPUT) && t->view_offs == 0) {
            via = t;
            continue;
        }
        if (t->op == GGML_OP_CONCAT && t->src[1]->ne[0] <= 16 && (t->src[0] == g || (via && t->src[0] == via)) &&
                t->src[0]->ne[0] == 3 && t->src[0]->ne[2] == 1) {
            if (!ggml_cuda_deferred_rows_intact(cgraph, i, j)) {
                return false;
            }
            cuda_ctx->conv_state_rows.push_back({ t, g });
            return true;
        }
    }
    return false;
}

// the deferred gather of this concat, if the conv block kernel does not take it (run it now)
static const ggml_tensor * ggml_cuda_conv_state_rows_take(ggml_backend_cuda_context * cuda_ctx, const ggml_tensor * concat, bool flush) {
    for (auto & e : cuda_ctx->conv_state_rows) {
        if (e.first == concat && e.second) {
            const ggml_tensor * g = e.second;
            e.second = nullptr;
            if (flush) {
                GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, const_cast<ggml_tensor *>(g)));
                return nullptr;
            }
            return g;
        }
    }
    return nullptr;
}

// DeepSeek V4 hyper-connection gates (build_hc_mixes): x*scale + base -> sigmoid -> scale(a) + bias(c), with x a strided
// view of the mixes ([hc, T], row stride of the mix row), scale and base broadcast along rows (one value or one per
// column). 4 of these chains per layer ran as 16 tiny kernels at decode. GGML_CUDA_FUSE_HC_GATE=0 off.
static __global__ void ggml_cuda_hc_gate_f32(const float * __restrict__ x, const float * __restrict__ sc,
        const float * __restrict__ bs, float * __restrict__ dst, const int n0, const int n1, const int64_t x_s1,
        const int sc_per_col, const int bs_per_col, const float a, const float c) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n0*n1) {
        return;
    }
    const int i0 = i % n0;
    const int i1 = i / n0;
    const float v = x[(int64_t) i1*x_s1 + i0]*sc[sc_per_col ? i0 : 0] + bs[bs_per_col ? i0 : 0];
    dst[i] = a/(1.0f + expf(-v)) + c;
}

// returns the number of extra nodes consumed (3) or 0
static int ggml_cuda_try_fuse_hc_gate(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_FUSE_HC_GATE"); return !e || atoi(e) != 0; }();
    if (!on || i + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mul = cgraph->nodes[i];
    const ggml_tensor * add = cgraph->nodes[i + 1];
    const ggml_tensor * sig = cgraph->nodes[i + 2];
    ggml_tensor       * scl = cgraph->nodes[i + 3];
    if (mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD || sig->op != GGML_OP_UNARY ||
            ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || scl->op != GGML_OP_SCALE) {
        return 0;
    }
    if (!ggml_cuda_can_fuse(cgraph, i, {GGML_OP_MUL, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE}, {GGML_UNARY_OP_SIGMOID})) {
        return 0;
    }
    if (add->src[0] != mul || sig->src[0] != add || scl->src[0] != sig) {
        return 0;
    }
    const ggml_tensor * x  = mul->src[0];
    const ggml_tensor * sc = mul->src[1];
    const ggml_tensor * bs = add->src[1];
    const int64_t n0 = mul->ne[0], n1 = mul->ne[1];
    auto bcast_ok = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && (t->ne[0] == 1 || t->ne[0] == n0) &&
            t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1;
    };
    if (x->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || scl->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, mul) ||
            mul->ne[2] != 1 || mul->ne[3] != 1 || x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 ||
            !bcast_ok(sc) || !bcast_ok(bs) || !ggml_is_contiguous(scl) || n0*n1 > INT_MAX) {
        return 0;
    }
    float a, c;
    memcpy(&a, (const float *) scl->op_params + 0, sizeof(float));
    memcpy(&c, (const float *) scl->op_params + 1, sizeof(float));
    const int n = (int) (n0*n1);
    ggml_cuda_hc_gate_f32<<<(n + 255)/256, 256, 0, cuda_ctx->stream()>>>((const float *) x->data, (const float *) sc->data,
        (const float *) bs->data, (float *) scl->data, (int) n0, (int) n1, (int64_t) (x->nb[1]/sizeof(float)),
        sc->ne[0] == n0 && n0 > 1, bs->ne[0] == n0 && n0 > 1, a, c);
    CUDA_CHECK(cudaGetLastError());
    return 3;
}

// GLM-5-Next KDA gate (glm5next build_kda_layer): ADD(f, dt_b) -> RESHAPE -> MUL(., a per head) -> SCALE -> SIGMOID ->
// SCALE, i.e. g = lb*sigmoid(-a_h*(f + dt)), 5 kernels per KDA layer as one, with the unfused kernels' arithmetic (so
// bit-identical). Only views may sit between the chain's nodes. GGML_CUDA_FUSE_KDA_GATE=0 off.
static __global__ void ggml_cuda_kda_gate_f32(const float * __restrict__ f, const float * __restrict__ dt,
        const float * __restrict__ a, float * __restrict__ dst, const int n0, const int n1, const int64_t f_s1,
        const int head_dim, const float s0, const float b0, const float s1, const float b1) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n0*n1) {
        return;
    }
    const int   i0 = i % n0;
    const int   i1 = i / n0;
    float x = f[(int64_t) i1*f_s1 + i0];
    if (dt) {
        x += dt[i0];
    }
    const float y  = x*a[i0/head_dim];
    const float z  = s0 * y + b0;
    const float sg = 1.0f / (1.0f + expf(-z));
    dst[i] = s1 * sg + b1;
}

// returns the number of extra nodes consumed or 0
static int ggml_cuda_try_fuse_kda_gate(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_FUSE_KDA_GATE"); return !e || atoi(e) != 0; }();
    if (!on) {
        return 0;
    }
    // the next compute nodes after i (4 from an ADD, 3 from a MUL), allowing only views in between
    const bool has_add = cgraph->nodes[i]->op == GGML_OP_ADD;
    const int  nc      = has_add ? 5 : 4;
    int idx[5] = { i, -1, -1, -1, -1 };
    int n = 1;
    for (int j = i + 1; j < cgraph->n_nodes && n < nc && j <= i + 12; ++j) {
        const ggml_op op = cgraph->nodes[j]->op;
        if (op == GGML_OP_RESHAPE || op == GGML_OP_VIEW) {
            continue;
        }
        idx[n++] = j;
    }
    if (n < nc) {
        return 0;
    }
    const int o = has_add ? 1 : 0;
    const ggml_tensor * add = has_add ? cgraph->nodes[idx[0]] : nullptr;
    const ggml_tensor * mul = cgraph->nodes[idx[o + 0]];
    const ggml_tensor * sc0 = cgraph->nodes[idx[o + 1]];
    const ggml_tensor * sig = cgraph->nodes[idx[o + 2]];
    ggml_tensor       * sc1 = cgraph->nodes[idx[o + 3]];
    if ((has_add && add->op != GGML_OP_ADD) || mul->op != GGML_OP_MUL || sc0->op != GGML_OP_SCALE || sig->op != GGML_OP_UNARY ||
            ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || sc1->op != GGML_OP_SCALE) {
        return 0;
    }
    // mul->src[0] must be a reshape (of add, or of the f tensor when the add was fused upstream); the rest a plain chain
    const ggml_tensor * mx = mul->src[0];
    if (mx->op != GGML_OP_RESHAPE || (has_add && mx->src[0] != add) || sc0->src[0] != mul || sig->src[0] != sc0 ||
            sc1->src[0] != sig) {
        return 0;
    }
    // every intermediate consumed once, within the chain, and not a graph output (without the add the reshape sits
    // before i)
    int mx_idx = -1;
    for (int j = has_add ? idx[0] + 1 : std::max(0, i - 16); j < (has_add ? idx[1] : i); ++j) {
        if (cgraph->nodes[j] == mx) {
            mx_idx = j;
        }
    }
    if (mx_idx < 0 || (has_add && ggml_node_get_use_count(cgraph, idx[0]) != 1) || ggml_node_get_use_count(cgraph, mx_idx) != 1 ||
            ggml_node_get_use_count(cgraph, idx[o + 0]) != 1 || ggml_node_get_use_count(cgraph, idx[o + 1]) != 1 ||
            ggml_node_get_use_count(cgraph, idx[o + 2]) != 1) {
        return 0;
    }
    for (const ggml_tensor * t : { add, mx, mul, sc0, sig }) {
        if (t && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }
    const ggml_tensor * f  = has_add ? add->src[0] : mx->src[0];
    const ggml_tensor * dt = has_add ? add->src[1] : nullptr;
    const ggml_tensor * fr = has_add ? add : mx->src[0]; // the [n0, n1] tensor the reshape views
    const ggml_tensor * a  = mul->src[1];
    const int64_t n0 = fr->ne[0], n1 = fr->ne[1];
    // f [n0, n1] (rows may be strided), dt [n0], a [1, n0/head_dim, 1] read per head, everything f32
    if (f->type != GGML_TYPE_F32 || (dt && dt->type != GGML_TYPE_F32) || a->type != GGML_TYPE_F32 || fr->type != GGML_TYPE_F32 ||
            sc1->type != GGML_TYPE_F32 || fr->ne[2] != 1 || fr->ne[3] != 1 || !ggml_are_same_shape(f, fr) ||
            f->nb[0] != sizeof(float) || f->nb[1] % sizeof(float) != 0 || (dt && (ggml_nelements(dt) != n0 || !ggml_is_contiguous(dt))) ||
            !ggml_is_contiguous(a) || mx->ne[0]*mx->ne[1] != n0 || a->ne[0] != 1 || a->ne[1] != mx->ne[1] ||
            ggml_nelements(a) != mx->ne[1] || !ggml_is_contiguous(sc1) || ggml_nelements(sc1) != n0*n1 || n0*n1 > INT_MAX) {
        return 0;
    }
    float s0, b0, s1, b1;
    memcpy(&s0, (const float *) sc0->op_params + 0, sizeof(float));
    memcpy(&b0, (const float *) sc0->op_params + 1, sizeof(float));
    memcpy(&s1, (const float *) sc1->op_params + 0, sizeof(float));
    memcpy(&b1, (const float *) sc1->op_params + 1, sizeof(float));
    const int nel = (int) (n0*n1);
    ggml_cuda_kda_gate_f32<<<(nel + 255)/256, 256, 0, cuda_ctx->stream()>>>((const float *) f->data, dt ? (const float *) dt->data : nullptr,
        (const float *) a->data, (float *) sc1->data, (int) n0, (int) n1, (int64_t) (f->nb[1]/sizeof(float)), (int) mx->ne[0],
        s0, b0, s1, b1);
    CUDA_CHECK(cudaGetLastError());
    return idx[o + 3] - i;
}

// single-token MoE FFN with a shared expert (DeepSeek V4.1 decode): MUL_MAT_ID gate, up -> SWIGLU_CLAMP -> MUL_MAT_ID down
// -> weighted expert reduction (MUL, VIEWs, ADDs) -> MUL_MAT shared gate, up -> SWIGLU_CLAMP -> MUL_MAT shared down -> ADD
// as 2 kernels (ggml_cuda_moe1_ffn); returns the number of extra nodes consumed. GGML_CUDA_MOE1=0 off.
static int ggml_cuda_try_fuse_moe1(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const int nn = cgraph->n_nodes;
    static const bool dbg = [] { const char * e = getenv("GGML_CUDA_MOE1_DEBUG"); return e && atoi(e) != 0; }();
    auto fail = [&](int w) {
        if (dbg) {
            fprintf(stderr, "moe1: node %d %s not fused (check %d)\n", i, cgraph->nodes[i]->name, w);
        }
        return 0;
    };
    if (i + 4 >= nn) {
        return fail(1);
    }
    ggml_tensor * g  = cgraph->nodes[i];
    ggml_tensor * u  = cgraph->nodes[i + 1];
    ggml_tensor * gl = cgraph->nodes[i + 2];
    ggml_tensor * dn = cgraph->nodes[i + 3];
    if (g->op != GGML_OP_MUL_MAT_ID || u->op != GGML_OP_MUL_MAT_ID || gl->op != GGML_OP_GLU || dn->op != GGML_OP_MUL_MAT_ID) {
        return fail(2);
    }
    if (ggml_get_glu_op(gl) != GGML_GLU_OP_SWIGLU_CLAMP || gl->src[0] != g || gl->src[1] != u || ggml_get_op_params_i32(gl, 1) != 0) {
        return fail(3);
    }
    const ggml_tensor * x   = g->src[1];
    const ggml_tensor * ids = g->src[2];
    if (u->src[1] != x || u->src[2] != ids || dn->src[1] != gl || dn->src[2] != ids || x->ne[1] != 1 || x->ne[2] != 1 ||
            ids->ne[1] != 1) {
        return fail(4);
    }
    ggml_cuda_moe_weighted_reduction_match wr;
    if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i + 4, wr) || wr.experts != dn || wr.expert_scale != nullptr) {
        return fail(5);
    }
    // no-op views of unrelated tensors can sit between the reduction and the shared expert (the next layer's KV state
    // views before a KV source layer): skipped, and left out of the subgraph check
    int j = i + 4 + wr.node_count;
    while (j < nn && (cgraph->nodes[j]->op == GGML_OP_VIEW || cgraph->nodes[j]->op == GGML_OP_RESHAPE) &&
            cgraph->nodes[j]->view_src != nullptr && cgraph->nodes[j]->view_src != wr.dst) {
        ++j;
    }
    if (j + 4 >= nn) {
        return fail(6);
    }
    ggml_tensor * sg  = cgraph->nodes[j];
    ggml_tensor * su  = cgraph->nodes[j + 1];
    ggml_tensor * sgl = cgraph->nodes[j + 2];
    ggml_tensor * sd  = cgraph->nodes[j + 3];
    ggml_tensor * add = cgraph->nodes[j + 4];
    if (sg->op != GGML_OP_MUL_MAT || su->op != GGML_OP_MUL_MAT || sgl->op != GGML_OP_GLU || sd->op != GGML_OP_MUL_MAT ||
            add->op != GGML_OP_ADD) {
        if (dbg) {
            for (int k = j - 1; k < j + 6 && k < nn; ++k) {
                fprintf(stderr, "moe1:   node %d %s %s\n", k, ggml_op_desc(cgraph->nodes[k]), cgraph->nodes[k]->name);
            }
        }
        return fail(7);
    }
    if (ggml_get_glu_op(sgl) != GGML_GLU_OP_SWIGLU_CLAMP || sgl->src[0] != sg || sgl->src[1] != su ||
            ggml_get_op_params_i32(sgl, 1) != 0 || ggml_get_op_params_f32(sgl, 3) != ggml_get_op_params_f32(gl, 3)) {
        return fail(8);
    }
    if (sg->src[1]->data != x->data || su->src[1] != sg->src[1] || sd->src[1] != sgl || !ggml_is_contiguous(sg->src[1])) {
        return fail(9);
    }
    const ggml_tensor * moe_out = wr.dst;
    if (!((add->src[0] == moe_out && add->src[1] == sd) || (add->src[1] == moe_out && add->src[0] == sd))) {
        return fail(10);
    }
    const int n_moe = 4 + wr.node_count; // nodes i .. i + n_moe - 1, then j .. j + 4
    const int count = n_moe + 5;
    if (count >= 32) {
        return fail(11);
    }
    int          idxs[32];
    enum ggml_op ops[32];
    for (int k = 0; k < count; ++k) {
        idxs[k] = k < n_moe ? i + k : j + (k - n_moe);
        ops[k]  = cgraph->nodes[idxs[k]]->op;
    }
    const int out_idx = j + 4;
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, &out_idx, 1)) {
        return fail(12);
    }
    // the gate/up kernel reads the token, ids and router weights (copying ids and weights for the down kernel) before the
    // down kernel writes the output, so the output may reuse their memory; it must not overlap the weights
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data, * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    for (const ggml_tensor * w : { g->src[0], u->src[0], dn->src[0], sg->src[0], su->src[0], sd->src[0] }) {
        if (overlaps(add, w)) {
            return fail(14);
        }
    }
    ggml_cuda_moe1_args a;
    a.x         = sg->src[1];
    a.ids       = ids;
    a.weights   = wr.weights;
    a.gate_exps = g->src[0];
    a.up_exps   = u->src[0];
    a.down_exps = dn->src[0];
    a.gate_sh   = sg->src[0];
    a.up_sh     = su->src[0];
    a.down_sh   = sd->src[0];
    a.limit     = ggml_get_op_params_f32(gl, 3);
    a.dst       = add;
    if (!ggml_cuda_moe1_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, a)) {
        return fail(13);
    }
    ggml_cuda_moe1_ffn(*cuda_ctx, a);
    return j + 4 - i;
}

// DSV4 HC sublayer transition (decode): DSV4_HC_POST -> DSV4_HC_MIX of its output -> [views] -> DSV4_HC_PRE of its
// output -> RMS_NORM -> MUL (norm weight) as one kernel (dsv4_hc_step_f32). GGML_CUDA_HC_STEP=0 off.
static int ggml_cuda_try_fuse_hc_step(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_HC_STEP"); return !e || atoi(e) != 0; }();
    const int nn = cgraph->n_nodes;
    if (!env || i + 4 >= nn) {
        return 0;
    }
    ggml_tensor * post = cgraph->nodes[i];
    ggml_tensor * mixn = cgraph->nodes[i + 1];
    if (mixn->op != GGML_OP_DSV4_HC_MIX || mixn->src[0] != post) {
        return 0;
    }
    // yn allowed over an input of the group through a temporary (GGML_CUDA_HC_STEP_YN=0 off): DSpark's 6-token verify had
    // most HC steps unfused (the allocator puts yn over X)
    static const bool yn_env = [] { const char * e = getenv("GGML_CUDA_HC_STEP_YN"); return !e || atoi(e) != 0; }();
    auto hcstep_yn_env = [] { return yn_env; };
    bool yn_temp = false;
    int j = i + 2;
    while (j < nn && (cgraph->nodes[j]->op == GGML_OP_VIEW || cgraph->nodes[j]->op == GGML_OP_RESHAPE)) {
        ++j;
    }
    if (j + 2 >= nn) {
        return 0;
    }
    ggml_tensor * pre = cgraph->nodes[j];
    ggml_tensor * rn  = cgraph->nodes[j + 1];
    ggml_tensor * mul = cgraph->nodes[j + 2];
    if (pre->op != GGML_OP_DSV4_HC_PRE || rn->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL ||
            !ggml_cuda_dsv4_hc_step_supported(post, mixn, pre, rn, mul, hcstep_yn_env() ? &yn_temp : nullptr)) {
        return 0;
    }
    int          idxs[5] = { i, i + 1, j, j + 1, j + 2 };
    enum ggml_op ops[5]  = { GGML_OP_DSV4_HC_POST, GGML_OP_DSV4_HC_MIX, GGML_OP_DSV4_HC_PRE, GGML_OP_RMS_NORM, GGML_OP_MUL };
    int          outs[3] = { i, i + 1, j + 2 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 5, ops, outs, 3)) {
        return 0;
    }
    ggml_cuda_dsv4_hc_step(*cuda_ctx, post, mixn, pre, rn, mul, yn_temp);
    return j + 2 - i;
}

// DeepSeek V4 compressor state row copies (restore / snapshot / persist of the dsv4_*_state_* tensors): GET_ROWS -> VIEW ->
// SET_ROWS, or CONCAT(dim 1) -> GET_ROWS -> VIEW -> SET_ROWS (the snapshot's source is the restored state followed by the
// current rows), as one gather-scatter kernel: dst[dst_idx[r]] = src[src_idx[r]] with src the virtual concat. Done for
// the kv and the score tensor in one launch when the next nodes repeat the pattern with the same index tensors. The
// state planner never chains rows (restore: snapshot plane -> live rows or a row onto itself; snapshot: live / current
// rows -> planes >= 1; persist: current rows -> live rows), so copying in parallel equals gather-then-scatter.
// GGML_CUDA_DSV4_STATE_FUSE=0 off.
struct dsv4_rows_job {
    const char * a;
    const char * b;
    int64_t      nb_a, nb_b, n_a;
    const int32_t * src_idx;
    const void   * dst_idx;
    int          dst_i64;
    char       * dst;
    int64_t      nb_dst;
};

static __global__ void k_dsv4_rows_copy(const dsv4_rows_job j0, const dsv4_rows_job j1, const int w4) {
    const dsv4_rows_job & j = blockIdx.y == 0 ? j0 : j1;
    const int     r = blockIdx.x;
    const int64_t s = j.src_idx ? (int64_t) j.src_idx[r] : (int64_t) r;
    const int64_t d = j.dst_i64 ? ((const int64_t *) j.dst_idx)[r] : (int64_t) ((const int32_t *) j.dst_idx)[r];
    const float4 * src = (const float4 *) (s < j.n_a ? j.a + s*j.nb_a : j.b + (s - j.n_a)*j.nb_b);
    float4 * dst = (float4 *) (j.dst + d*j.nb_dst);
    for (int k = threadIdx.x; k < w4; k += blockDim.x) {
        dst[k] = src[k];
    }
}

// a plain SET_ROWS of a 2D f32 tensor's rows into a dsv4_*_state_* view (the decode persist, whose gather is the row
// itself): fills job (identity gather), returns 1 or 0
static int ggml_cuda_match_dsv4_set_rows(ggml_cgraph * cgraph, int i, dsv4_rows_job & job, const ggml_tensor ** idx_pair,
                                         int64_t & n_rows, int64_t & width) {
    const ggml_tensor * sr = cgraph->nodes[i];
    if (sr->op != GGML_OP_SET_ROWS || sr->src[0]->op == GGML_OP_GET_ROWS) {
        return 0;
    }
    const ggml_tensor * src = sr->src[0];
    const ggml_tensor * si  = sr->src[1];
    const ggml_tensor * v   = sr->src[2];
    const ggml_tensor * base = v->view_src ? v->view_src : v;
    if (strncmp(base->name, "dsv4_", 5) != 0 || strstr(base->name, "_state_") == nullptr) {
        return 0;
    }
    const int64_t W = src->ne[0];
    const auto rows2d = [W](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->ne[0] == W && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float) &&
            t->nb[1] % 16 == 0 && ((uintptr_t) t->data) % 16 == 0;
    };
    if (W % 4 != 0 || !rows2d(src) || !rows2d(v) || (si->type != GGML_TYPE_I32 && si->type != GGML_TYPE_I64) ||
            !ggml_is_contiguous(si) || si->ne[0] != src->ne[1] || src->ne[1] == 0 || src->ne[1] > 65535) {
        return 0;
    }
    job.a       = (const char *) src->data;
    job.b       = nullptr;
    job.nb_a    = src->nb[1];
    job.nb_b    = 0;
    job.n_a     = INT64_MAX;
    job.src_idx = nullptr;
    job.dst_idx = si->data;
    job.dst_i64 = si->type == GGML_TYPE_I64;
    job.dst     = (char *) v->data;
    job.nb_dst  = v->nb[1];
    idx_pair[0] = nullptr;
    idx_pair[1] = si;
    n_rows      = src->ne[1];
    width       = W;
    return 1;
}

// match one pattern at node i: fills job, returns the number of nodes it spans (0: no match)
static int ggml_cuda_match_dsv4_rows(ggml_cgraph * cgraph, int i, dsv4_rows_job & job, const ggml_tensor ** idx_pair, int64_t & n_rows,
                                     int64_t & width) {
    const int nn = cgraph->n_nodes;
    int k = i;
    const ggml_tensor * cc = nullptr;
    if (cgraph->nodes[k]->op == GGML_OP_CONCAT) {
        cc = cgraph->nodes[k];
        if (ggml_get_op_params_i32(cc, 0) != 1 || ggml_node_get_use_count(cgraph, k) != 1) {
            return 0;
        }
        ++k;
    }
    if (k + 2 >= nn) {
        return 0;
    }
    const ggml_tensor * gr = cgraph->nodes[k];
    const ggml_tensor * v  = cgraph->nodes[k + 1];
    const ggml_tensor * sr = cgraph->nodes[k + 2];
    if (gr->op != GGML_OP_GET_ROWS || v->op != GGML_OP_VIEW || sr->op != GGML_OP_SET_ROWS || sr->src[0] != gr ||
            sr->src[2] != v || (cc && gr->src[0] != cc) || ggml_node_get_use_count(cgraph, k) != 1 ||
            (gr->flags & GGML_TENSOR_FLAG_OUTPUT) || (cc && (cc->flags & GGML_TENSOR_FLAG_OUTPUT))) {
        return 0;
    }
    const ggml_tensor * base = v->view_src ? v->view_src : v;
    if (strncmp(base->name, "dsv4_", 5) != 0 || strstr(base->name, "_state_") == nullptr) {
        return 0;
    }
    const ggml_tensor * gi = gr->src[1];
    const ggml_tensor * si = sr->src[1];
    const int64_t W = gr->ne[0];
    const auto rows2d = [W](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->ne[0] == W && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float) &&
            t->nb[1] % 16 == 0 && ((uintptr_t) t->data) % 16 == 0;
    };
    const ggml_tensor * a = cc ? cc->src[0] : gr->src[0];
    const ggml_tensor * b = cc ? cc->src[1] : nullptr;
    if (W % 4 != 0 || gr->type != GGML_TYPE_F32 || gr->ne[2] != 1 || gr->ne[3] != 1 || !rows2d(a) || (b && !rows2d(b)) ||
            !rows2d(v) || gi->type != GGML_TYPE_I32 || !ggml_is_contiguous(gi) || gi->ne[0] != gr->ne[1] ||
            (si->type != GGML_TYPE_I32 && si->type != GGML_TYPE_I64) || !ggml_is_contiguous(si) || si->ne[0] != gr->ne[1] ||
            gr->ne[1] == 0 || gr->ne[1] > 65535) {
        return 0;
    }
    job.a       = (const char *) a->data;
    job.b       = b ? (const char *) b->data : nullptr;
    job.nb_a    = a->nb[1];
    job.nb_b    = b ? b->nb[1] : 0;
    job.n_a     = b ? a->ne[1] : INT64_MAX;
    job.src_idx = (const int32_t *) gi->data;
    job.dst_idx = si->data;
    job.dst_i64 = si->type == GGML_TYPE_I64;
    job.dst     = (char *) v->data;
    job.nb_dst  = v->nb[1];
    idx_pair[0] = gi;
    idx_pair[1] = si;
    n_rows      = gr->ne[1];
    width       = W;
    return k + 2 - i + 1;
}

static int ggml_cuda_try_fuse_dsv4_rows(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_DSV4_STATE_FUSE"); return !e || atoi(e) != 0; }();
    if (!env) {
        return 0;
    }
    dsv4_rows_job j0, j1;
    const ggml_tensor * idx0[2];
    const ggml_tensor * idx1[2];
    int64_t n0 = 0, n1 = 0, w0 = 0, w1 = 0;
    const int len0 = ggml_cuda_match_dsv4_rows(cgraph, i, j0, idx0, n0, w0);
    if (len0 == 0) {
        return 0;
    }
    int len = len0;
    int njobs = 1;
    if (i + len0 < cgraph->n_nodes) {
        const int len1 = ggml_cuda_match_dsv4_rows(cgraph, i + len0, j1, idx1, n1, w1);
        // the second state tensor (score after kv) with the same row lists: one launch
        if (len1 == len0 && n1 == n0 && w1 == w0 && idx1[0] == idx0[0] && idx1[1] == idx0[1]) {
            len += len1;
            njobs = 2;
        }
    }
    if (njobs == 1) {
        j1 = j0;
    }
    const int w4 = (int) (w0/4);
    k_dsv4_rows_copy<<<dim3((unsigned) n0, njobs), std::min(256, std::max(64, w4)), 0, cuda_ctx->stream()>>>(j0, j1, w4);
    CUDA_CHECK(cudaGetLastError());
    return len - 1;
}

// the decode persist of a compressor: SET_ROWS (kv rows) -> VIEW -> SET_ROWS (score rows) with one index list -> one launch
static int ggml_cuda_try_fuse_dsv4_set_rows(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_DSV4_STATE_FUSE"); return !e || atoi(e) != 0; }();
    if (!env || i + 2 >= cgraph->n_nodes || cgraph->nodes[i + 1]->op != GGML_OP_VIEW) {
        return 0;
    }
    dsv4_rows_job j0, j1;
    const ggml_tensor * idx0[2];
    const ggml_tensor * idx1[2];
    int64_t n0 = 0, n1 = 0, w0 = 0, w1 = 0;
    if (!ggml_cuda_match_dsv4_set_rows(cgraph, i, j0, idx0, n0, w0) ||
            !ggml_cuda_match_dsv4_set_rows(cgraph, i + 2, j1, idx1, n1, w1) || cgraph->nodes[i + 2]->src[2] != cgraph->nodes[i + 1] ||
            idx1[1] != idx0[1] || n1 != n0 || w1 != w0) {
        return 0;
    }
    const int w4 = (int) (w0/4);
    k_dsv4_rows_copy<<<dim3((unsigned) n0, 2), std::min(256, std::max(64, w4)), 0, cuda_ctx->stream()>>>(j0, j1, w4);
    CUDA_CHECK(cudaGetLastError());
    return 2;
}

// DeepSeek V4 compressor score + absolute position embedding: GET_ROWS(ape [W, ratio] f16/f32, state_pos [n] i32) ->
// ADD(score [W, n] f32, rows) as one kernel (the rows are used only by the add). GGML_CUDA_DSV4_APE_FUSE=0 off.
template <typename T>
static __global__ void k_dsv4_ape_add(const float * __restrict__ a, const T * __restrict__ ape, const int32_t * __restrict__ pos,
        float * __restrict__ dst, const int W, const int64_t sa, const int64_t sape, const int64_t sd) {
    const int t = blockIdx.x;
    const T * er = ape + (int64_t) pos[t]*sape;
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        dst[t*sd + c] = a[t*sa + c] + (float) er[c];
    }
}

static int ggml_cuda_try_fuse_dsv4_ape(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_DSV4_APE_FUSE"); return !e || atoi(e) != 0; }();
    if (!env || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * gr  = cgraph->nodes[i];
    ggml_tensor *       add = cgraph->nodes[i + 1];
    if (gr->op != GGML_OP_GET_ROWS || add->op != GGML_OP_ADD || add->src[1] != gr || ggml_node_get_use_count(cgraph, i) != 1 ||
            (gr->flags & GGML_TENSOR_FLAG_OUTPUT) || strstr(gr->src[0]->name, "compressor_ape") == nullptr) {
        return 0;
    }
    const ggml_tensor * ape = gr->src[0];
    const ggml_tensor * pos = gr->src[1];
    const ggml_tensor * a   = add->src[0];
    const int64_t W = add->ne[0], n = add->ne[1];
    if ((ape->type != GGML_TYPE_F16 && ape->type != GGML_TYPE_F32) || ape->ne[0] != W || ape->ne[2] != 1 || ape->ne[3] != 1 ||
            ape->nb[0] != ggml_type_size(ape->type) || pos->type != GGML_TYPE_I32 || !ggml_is_contiguous(pos) ||
            pos->ne[0] != n || a->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || gr->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(a, add) || !ggml_are_same_shape(gr, add) || a->nb[0] != sizeof(float) ||
            add->nb[0] != sizeof(float) || add->ne[2] != 1 || add->ne[3] != 1 || n == 0 || n > 65535) {
        return 0;
    }
    const int threads = (int) std::min<int64_t>(256, ((W + 63)/64)*64);
    if (ape->type == GGML_TYPE_F16) {
        k_dsv4_ape_add<half><<<(unsigned) n, threads, 0, cuda_ctx->stream()>>>((const float *) a->data, (const half *) ape->data,
            (const int32_t *) pos->data, (float *) add->data, (int) W, a->nb[1]/sizeof(float), ape->nb[1]/sizeof(half), add->nb[1]/sizeof(float));
    } else {
        k_dsv4_ape_add<float><<<(unsigned) n, threads, 0, cuda_ctx->stream()>>>((const float *) a->data, (const float *) ape->data,
            (const int32_t *) pos->data, (float *) add->data, (int) W, a->nb[1]/sizeof(float), ape->nb[1]/sizeof(float), add->nb[1]/sizeof(float));
    }
    CUDA_CHECK(cudaGetLastError());
    return 1;
}

// single-token router: MUL_MAT (bf16 gate_inp) -> SOFTPLUS -> SQRT -> RESHAPE -> ADD (bias) -> ARGSORT -> VIEW (ids) ->
// GET_ROWS -> RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE [-> SCALE] as one kernel (ggml_cuda_router1_topk).
// GGML_CUDA_ROUTER1=0 off.
static int ggml_cuda_try_fuse_router1(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_ROUTER1"); return !e || atoi(e) != 0; }();
    const int nn = cgraph->n_nodes;
    if (!env || i + 13 >= nn) {
        return 0;
    }
    ggml_tensor * mm = cgraph->nodes[i];
    ggml_tensor * sp = cgraph->nodes[i + 1];
    // DeepSeek V4.1: sqrt(softplus) gating; GLM-5-Next: sigmoid (one node fewer: no SQRT)
    const bool sig = sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SIGMOID;
    if (sp->op != GGML_OP_UNARY || (!sig && ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS) || sp->src[0] != mm) {
        return 0;
    }
    ggml_cuda_topk_moe_args args;
    if (!ggml_cuda_topk_moe_fusion(cgraph, i + 1, args) || (sig ? !args.sigmoid : !args.sqrt_softplus) || !args.prob_bias ||
            !args.norm || args.delayed_softmax) {
        return 0;
    }
    std::vector<ggml_op> ops = { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD,
        GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV,
        GGML_OP_RESHAPE };
    if (sig) {
        ops.erase(ops.begin() + 2);
    }
    if (args.scale) {
        ops.push_back(GGML_OP_SCALE);
    }
    const int n = (int) ops.size();
    if (i + n > nn) {
        return 0;
    }
    const int o = sig ? -1 : 0; // node offsets after the gating
    const ggml_tensor * bias    = cgraph->nodes[i + 4 + o]->src[1];
    ggml_tensor *       ids     = cgraph->nodes[i + 6 + o];
    const ggml_tensor * clamp   = cgraph->nodes[i + 10 + o];
    ggml_tensor *       weights = cgraph->nodes[i + n - 1];
    const float scale_val = args.scale ? ggml_get_op_params_f32(cgraph->nodes[i + n - 1], 0) : 1.0f;
    const int   n_used    = (int) (weights->ne[0]*weights->ne[1]);
    int outs[2] = { i + 6 + o, i + n - 1 };
    // several tokens: one row of ids (a view of the argsort, nb[1] per token) and of weights (ne[2] = tokens) per token
    const int64_t n_tok = mm->ne[1];
    const int     n_used_tok = (int) (weights->ne[0]*weights->ne[1]);
    if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) || ggml_nelements(bias) != mm->ne[0] ||
            ids->type != GGML_TYPE_I32 || weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) ||
            ggml_nrows(mm) != n_tok || n_tok > 4 || (n_tok > 1 && (weights->ne[2] != n_tok || ids->ne[1] != n_tok ||
            ids->nb[0] != sizeof(int32_t))) || !ggml_cuda_router1_supported(mm, n_tok > 1 ? n_used_tok : n_used, sig) ||
            !ggml_can_fuse_subgraph(cgraph, i, n, ops.data(), outs, 2)) {
        return 0;
    }
    ggml_cuda_router1_topk(*cuda_ctx, mm, weights, ids, bias, ggml_get_op_params_f32(clamp, 0), scale_val, sig, (int) n_tok,
                           n_tok > 1 ? (int64_t) (ids->nb[1]/sizeof(int32_t)) : 0, n_tok > 1 ? (int64_t) (weights->nb[2]/sizeof(float)) : 0);
    return n - 1;
}

#define GGML_CUDA_FUSED_SELF (-1)

// single-token q2_K matvecs that share src1 (e.g. q_a, wkv, the compressor and indexer projections of attn_norm): later
// siblings are computed with node i in one launch and skipped when reached (GGML_CUDA_GEMV1_MULTI=0 off). A sibling c
// may move up to i only if no node between writes src1 or c's output, or reads c's output memory (the allocator may
// reuse a dead tensor's buffer for c). Returns GGML_CUDA_FUSED_SELF when node i was computed, 0 otherwise.
static int ggml_cuda_try_fuse_gemv1_multi(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_GEMV1_MULTI"); return !e || atoi(e) != 0; }();
    ggml_tensor * mm = cgraph->nodes[i];
    if (!env || mm->op != GGML_OP_MUL_MAT || (mm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * x = mm->src[1];
    if (mm->src[0]->ne[2] != 1 || x->ne[2] != 1 || !ggml_cuda_gemv1_q2k_supported(cc, mm->src[0], x, mm)) {
        return 0;
    }
    const int64_t K = mm->src[0]->ne[0];
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr) {
            return false;
        }
        const char * pa = (const char *) a->data;
        const char * pb = (const char *) b->data;
        return pa < pb + ggml_nbytes(b) && pb < pa + ggml_nbytes(a);
    };
    const ggml_tensor * sel[GGML_CUDA_GEMV1_MULTI_MAX] = { mm };
    int n = 1;
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 128 && n < GGML_CUDA_GEMV1_MULTI_MAX; ++j) {
        ggml_tensor * c = cgraph->nodes[j];
        if (c->op != GGML_OP_MUL_MAT || c->src[1] != x || (c->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                c->src[0]->ne[0] != K || c->src[0]->ne[2] != 1 || c->src[0]->type != mm->src[0]->type ||
                !ggml_cuda_gemv1_q2k_supported(cc, c->src[0], x, c)) {
            continue;
        }
        bool ok = true;
        for (int k = i + 1; k < j && ok; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            if (std::find(cuda_ctx->early_done.begin(), cuda_ctx->early_done.end(), t) != cuda_ctx->early_done.end()) {
                continue; // already moved up (its output was checked when it was)
            }
            ok = !overlap(t, x) && !overlap(t, c);
            for (int q = 0; q < GGML_MAX_SRC && ok; ++q) {
                ok = !overlap(t->src[q], c);
            }
        }
        for (int k = 0; k < n && ok; ++k) {
            ok = !overlap(sel[k], c);
        }
        if (ok) {
            sel[n++] = c;
        }
    }
    if (n < 2) {
        return 0;
    }
    // epilogue: a sibling whose output (<= 512 values, one row) goes RMS_NORM -> MUL -> ROPE -> VIEW -> SET_ROWS (f16)
    // later in the graph (the decode kv projection) is normalized, rotated and stored into its cache row by the launch's
    // last workgroup of that segment; the five nodes are then skipped (GGML_CUDA_GEMV1_EPI=0 off). They may move up to i
    // only if no node in between touches the cache or writes their inputs.
    static const bool epi_env = [] { const char * e = getenv("GGML_CUDA_GEMV1_EPI"); return !e || atoi(e) != 0; }();
    ggml_cuda_gemv1_epi ep = {};
    ep.seg = -1;
    int epi_p = -1;
    for (int k = 0; k < n && epi_env && epi_p < 0; ++k) {
        const ggml_tensor * c = sel[k];
        if (c->src[0]->ne[1] > 512 || c->src[0]->ne[1] % 2 != 0 || c->ne[1] != 1 || !ggml_is_contiguous(c)) {
            continue;
        }
        for (int p = i + 1; p + 4 < cgraph->n_nodes && p < i + 160; ++p) {
            const ggml_tensor * rn = cgraph->nodes[p];
            if (rn->op != GGML_OP_RMS_NORM) {
                continue;
            }
            const ggml_tensor * s0 = rn->src[0];
            if (!(s0 == c || (s0->view_src == c && s0->view_offs == 0 && ggml_is_contiguous(s0) &&
                              ggml_nelements(s0) == ggml_nelements(c)))) {
                continue;
            }
            const ggml_tensor * mul  = cgraph->nodes[p + 1];
            const ggml_tensor * rope = cgraph->nodes[p + 2];
            const ggml_tensor * view = cgraph->nodes[p + 3];
            const ggml_tensor * sr   = cgraph->nodes[p + 4];
            const enum ggml_op ops[5] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };
            bool ok = true;
            for (int q = 0; q < 5 && ok; ++q) {
                ok = cgraph->nodes[p + q]->op == ops[q] && (cgraph->nodes[p + q]->flags & GGML_TENSOR_FLAG_COMPUTE);
            }
            const int out_idx = p + 4;
            const int fidx[5] = { p, p + 1, p + 2, p + 3, p + 4 };
            ok = ok && ggml_can_fuse_subgraph_ext(cgraph, fidx, 5, ops, &out_idx, 1) &&
                ggml_check_edges(cgraph, p, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
                ggml_cuda_should_fuse_rms_norm_mul_rope(rn, mul, rope) && ggml_cuda_should_fuse_rope_set_rows(rope, view, sr);
            if (!ok) {
                break;
            }
            const ggml_tensor * nw  = mul->src[0] == rn ? mul->src[1] : mul->src[0];
            const ggml_tensor * pos = rope->src[1];
            const ggml_tensor * ff  = rope->src[2];
            const ggml_tensor * idx = sr->src[1];
            const int mode = ((const int32_t *) rope->op_params)[2];
            ok = sr->type == GGML_TYPE_F16 && mode == GGML_ROPE_TYPE_NORMAL && rn->ne[0] == ggml_nelements(c) &&
                ggml_nrows(rn) == 1 && ggml_nelements(nw) == rn->ne[0] && ggml_is_contiguous(nw) &&
                pos->type == GGML_TYPE_I32 && ggml_nelements(pos) == 1 && ggml_nelements(idx) == 1 &&
                sr->nb[0] == sizeof(half) && sr->nb[1] % sizeof(half) == 0 && ((uintptr_t) sr->data) % 4 == 0 &&
                ff == nullptr;
            for (int q = i + 1; q < p && ok; ++q) {
                const ggml_tensor * t = cgraph->nodes[q];
                if (std::find(cuda_ctx->early_done.begin(), cuda_ctx->early_done.end(), t) != cuda_ctx->early_done.end() ||
                        std::find(sel, sel + n, t) != sel + n) {
                    continue;
                }
                ok = !overlap(t, sr) && !overlap(t, nw) && !overlap(t, pos) && !overlap(t, idx) && !overlap(t, ff);
                for (int s = 0; s < GGML_MAX_SRC && ok; ++s) {
                    ok = !overlap(t->src[s], sr);
                }
            }
            if (!ok) {
                break;
            }
            const int   n_dims     = ((const int32_t *) rope->op_params)[1];
            const int   n_ctx_orig = ((const int32_t *) rope->op_params)[4];
            float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
            memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
            memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
            memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
            memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
            memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
            memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));
            float corr[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr);
            ep.seg          = k;
            ep.nw           = (const float *) nw->data;
            memcpy(&ep.eps, rn->op_params, sizeof(float));
            ep.n_dims       = n_dims;
            ep.n_offs       = ((const int32_t *) rope->op_params)[15];
            ep.pos          = (const int32_t *) pos->data;
            ep.freq_scale   = freq_scale;
            ep.ext_factor   = ext_factor;
            ep.attn_factor  = attn_factor;
            ep.theta_scale  = powf(freq_base, -2.0f/n_dims);
            ep.corr0        = corr[0];
            ep.corr1        = corr[1];
            ep.freq_factors = ff ? (const float *) ff->data : nullptr;
            ep.dst          = (half *) sr->data;
            ep.row_idx      = (const int64_t *) idx->data;
            ep.stride       = sr->nb[1]/sizeof(half);
            epi_p = p;
            break;
        }
    }
    // the epilogue segment goes first: its workgroups are dispatched first, so its last-arriver tail overlaps the others
    const ggml_tensor * order[GGML_CUDA_GEMV1_MULTI_MAX];
    int no = 0;
    if (epi_p >= 0) {
        order[no++] = sel[ep.seg];
    }
    for (int k = 0; k < n; ++k) {
        if (epi_p < 0 || k != ep.seg) {
            order[no++] = sel[k];
        }
    }
    if (epi_p >= 0) {
        ep.seg = 0;
    }
    ggml_cuda_gemv1_q2k_multi(*cuda_ctx, order, n, epi_p >= 0 ? &ep : nullptr);
    for (int k = 1; k < n; ++k) {
        cuda_ctx->early_done.push_back(sel[k]);
    }
    if (epi_p >= 0) {
        for (int q = 0; q < 5; ++q) {
            cuda_ctx->early_done.push_back(cgraph->nodes[epi_p + q]);
        }
    }
    return GGML_CUDA_FUSED_SELF;
}

// single-token .. 8-token F16 matvecs (gcn_f16_mv) that share src1 and K, e.g. DeepSeek V4's compressor kv / gate, indexer
// compressor kv / gate and indexer proj (all of attn_norm, replicated F16): later siblings are computed with node i in one
// launch over their concatenated rows and skipped when reached (the gemv1_multi rules: a sibling c may move up to i only
// if no node in between writes src1 or c's output or reads c's output memory). GGML_CUDA_F16MV_MULTI=0 off.
static int ggml_cuda_try_fuse_f16mv_multi(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_F16MV_MULTI"); return !e || atoi(e) != 0; }();
    ggml_tensor * mm = cgraph->nodes[i];
    if (!env || mm->op != GGML_OP_MUL_MAT || (mm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || mm->src[0]->type != GGML_TYPE_F16 ||
            !ggml_cuda_gcn_f16_matvec_supported(cuda_ctx->device, mm->src[0], mm->src[1], mm)) {
        return 0;
    }
    const ggml_tensor * x = mm->src[1];
    const int64_t K = mm->src[0]->ne[0];
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr) {
            return false;
        }
        const char * pa = (const char *) a->data;
        const char * pb = (const char *) b->data;
        return pa < pb + ggml_nbytes(b) && pb < pa + ggml_nbytes(a);
    };
    const ggml_tensor * sel[GGML_CUDA_F16MV_MAXM] = { mm };
    int n = 1;
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 128 && n < GGML_CUDA_F16MV_MAXM; ++j) {
        ggml_tensor * c = cgraph->nodes[j];
        if (c->op != GGML_OP_MUL_MAT || c->src[1] != x || (c->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                c->src[0]->type != GGML_TYPE_F16 || c->src[0]->ne[0] != K ||
                !ggml_cuda_gcn_f16_matvec_supported(cuda_ctx->device, c->src[0], x, c)) {
            continue;
        }
        bool ok = true;
        for (int k = i + 1; k < j && ok; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            if (std::find(cuda_ctx->early_done.begin(), cuda_ctx->early_done.end(), t) != cuda_ctx->early_done.end()) {
                continue; // already moved up (its output was checked when it was)
            }
            ok = !overlap(t, x) && !overlap(t, c);
            for (int q = 0; q < GGML_MAX_SRC && ok; ++q) {
                ok = !overlap(t->src[q], c);
            }
        }
        for (int k = 0; k < n && ok; ++k) {
            ok = !overlap(sel[k], c);
        }
        if (ok) {
            sel[n++] = c;
        }
    }
    if (n < 2 || !ggml_cuda_gcn_f16_matvec_multi(*cuda_ctx, sel, n)) {
        return 0;
    }
    for (int k = 1; k < n; ++k) {
        cuda_ctx->early_done.push_back((ggml_tensor *) sel[k]);
    }
    return GGML_CUDA_FUSED_SELF;
}

// single-token RMS_NORM -> MUL (weight) -> MUL_MAT (q2_K, gemv1): the matvec normalizes the token in its prologue
// (GGML_CUDA_NORM_GEMV1=0 off). Returns the number of nodes consumed after i (2) or 0.
static int ggml_cuda_try_fuse_norm_gemv1(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_NORM_GEMV1"); return !e || atoi(e) != 0; }();
    if (!env || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * rn  = cgraph->nodes[i];
    ggml_tensor * mul = cgraph->nodes[i + 1];
    ggml_tensor * mm  = cgraph->nodes[i + 2];
    if (rn->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || mm->op != GGML_OP_MUL_MAT || mm->src[1] != mul ||
            (mul->src[0] != rn && mul->src[1] != rn) ||
            (rn->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (mul->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (rn->flags & GGML_TENSOR_FLAG_OUTPUT) || (mul->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            !ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i + 1, 1)) {
        return 0;
    }
    const ggml_tensor * x = rn->src[0];
    const ggml_tensor * w = mul->src[0] == rn ? mul->src[1] : mul->src[0];
    const int64_t K = x->ne[0];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nelements(x) != K || ((uintptr_t) x->data) % 16 != 0 ||
            w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != K || ((uintptr_t) w->data) % 16 != 0 ||
            !ggml_are_same_shape(rn, x) || !ggml_are_same_shape(mul, x) || K % 4 != 0) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ggml_cuda_gemv1_q2k_supported(cc, mm->src[0], mul, mm)) {
        return 0;
    }
    ggml_cuda_gemv1_q2k(*cuda_ctx, mm->src[0], x, mm, (const float *) w->data, ggml_get_op_params_f32(rn, 0));
    return 2;
}

// GLM-5-Next KDA conv block (decode / verify), from the q|k CONCAT: [MUL_MAT v ->] CONCAT(qk, v) -> RESHAPE ->
// TRANSPOSE -> CONCAT(conv state, transposed qkv) -> [VIEW tail, VIEW cache row, CPY] per rollback slot -> unrelated
// nodes (state gathers / clears) and the conv-weight RESHAPEs + CONCATs -> SSM_CONV -> SILU -> VIEW q -> L2_NORM ->
// VIEW k -> L2_NORM: one kernel (ggml_cuda_kda_conv_block), then the unrelated nodes in their order.
// GGML_KDA_CONV_BLOCK=0 off.
static int ggml_cuda_try_fuse_kda_conv(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const int nn = cgraph->n_nodes;
    ggml_tensor * qk = cgraph->nodes[i];
    const auto is_out = [](const ggml_tensor * t) { return (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0; };
    if (qk->op != GGML_OP_CONCAT || qk->type != GGML_TYPE_F32 || qk->src[1]->ne[0] <= 16 ||
            ggml_node_get_use_count(cgraph, i) != 1 || is_out(qk)) {
        return 0;
    }
    int j = i + 1;
    ggml_tensor * vmm = nullptr; // the v projection, when it runs between the two CONCATs
    if (j < nn && cgraph->nodes[j]->op == GGML_OP_MUL_MAT) {
        vmm = cgraph->nodes[j++];
    }
    if (j >= nn) {
        return 0;
    }
    ggml_tensor * qkv = cgraph->nodes[j];
    if (qkv->op != GGML_OP_CONCAT || qkv->src[0] != qk || (vmm && qkv->src[1] != vmm) || ggml_node_get_use_count(cgraph, j) != 1 ||
            is_out(qkv) || (vmm && (ggml_node_get_use_count(cgraph, j - 1) != 1 || is_out(vmm)))) {
        return 0;
    }
    // a v projection computed after the q|k concat may sit in q's or k's (then dead) memory: it would overwrite them
    // before the kernel reads them (glm5next expands all projections ahead of the concats, so this is the fallback)
    const auto ovl = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    if (vmm && (ovl(vmm, qk->src[0]) || ovl(vmm, qk->src[1]))) {
        return 0;
    }
    // the conv-state CONCAT, through the RESHAPE / TRANSPOSE views of qkv
    int jc = -1;
    for (int k = j + 1; k < std::min(nn, j + 6); ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (t->op == GGML_OP_CONCAT) {
            jc = k;
            break;
        }
        if (t->op != GGML_OP_RESHAPE && t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_VIEW) {
            return 0;
        }
    }
    if (jc < 0) {
        return 0;
    }
    ggml_tensor * cin = cgraph->nodes[jc];
    if (cin->src[1]->view_src != qkv || cin->src[1]->ne[0] > 16 || is_out(cin)) {
        return 0;
    }
    // the conv-state tails, one [VIEW of cin, VIEW of the cache, CPY] per rollback slot
    const ggml_tensor * tv[8], * tc[8];
    int n_slots = 0;
    int k = jc + 1;
    while (k + 2 < nn && cgraph->nodes[k]->op == GGML_OP_VIEW && cgraph->nodes[k]->src[0] == cin) {
        const ggml_tensor * v = cgraph->nodes[k], * dv = cgraph->nodes[k + 1], * cpy = cgraph->nodes[k + 2];
        if (n_slots == 8 || dv->op != GGML_OP_VIEW || cpy->op != GGML_OP_CPY || cpy->src[0] != v || cpy->src[1] != dv ||
                v->ne[0] != 3 || v->nb[1] != cin->nb[1] || ggml_node_get_use_count(cgraph, k) != 1) {
            return 0;
        }
        tv[n_slots] = v;
        tc[n_slots] = cpy;
        n_slots++;
        k += 3;
    }
    if (n_slots == 0 || ggml_node_get_use_count(cgraph, jc) != n_slots + 1) {
        return 0;
    }
    // up to the SSM_CONV of cin: the conv-weight nodes (absorbed) and unrelated nodes (run after the kernel)
    std::vector<int> middle;
    int is = -1;
    for (; k < std::min(nn, jc + 64); ++k) {
        if (cgraph->nodes[k]->op == GGML_OP_SSM_CONV && cgraph->nodes[k]->src[0] == cin) {
            is = k;
            break;
        }
        middle.push_back(k);
    }
    if (is < 0 || is + 5 >= nn) {
        return 0;
    }
    ggml_tensor * conv = cgraph->nodes[is];
    ggml_tensor * silu = cgraph->nodes[is + 1];
    ggml_tensor * qv = cgraph->nodes[is + 2], * ql = cgraph->nodes[is + 3];
    ggml_tensor * kv = cgraph->nodes[is + 4], * kl = cgraph->nodes[is + 5];
    if (silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != conv ||
            ggml_node_get_use_count(cgraph, is) != 1 || is_out(conv) || is_out(silu) ||
            qv->op != GGML_OP_VIEW || qv->src[0] != silu || ql->op != GGML_OP_L2_NORM || ql->src[0] != qv ||
            kv->op != GGML_OP_VIEW || kv->src[0] != silu || kl->op != GGML_OP_L2_NORM || kl->src[0] != kv ||
            ggml_node_get_use_count(cgraph, is + 2) != 1 || ggml_node_get_use_count(cgraph, is + 4) != 1) {
        return 0;
    }
    const ggml_tensor * w2 = conv->src[1];
    const ggml_tensor * w1 = w2->src[0];
    if (w2->op != GGML_OP_CONCAT || w1 == nullptr || w1->op != GGML_OP_CONCAT) {
        return 0;
    }
    // absorbed middle nodes: the weight CONCATs and the RESHAPEs feeding them, used nowhere else
    const ggml_tensor * absorbed[5] = { w1, w2, w1->src[0], w1->src[1], w2->src[1] };
    const auto is_absorbed = [&](const ggml_tensor * t) {
        for (const ggml_tensor * a : absorbed) {
            if (t == a) {
                return true;
            }
        }
        return false;
    };
    const ggml_tensor * fused[] = { qk, qkv, cin, conv, silu, qv, kv, w1, w2 };
    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    std::vector<const ggml_tensor *> outs = { silu, ql, kl };
    for (int s = 0; s < n_slots; ++s) {
        outs.push_back(tc[s]->src[1]);
    }
    for (int m : middle) {
        const ggml_tensor * t = cgraph->nodes[m];
        if (is_absorbed(t)) {
            if (ggml_node_get_use_count(cgraph, m) != 1 || is_out(t)) {
                return 0;
            }
            continue;
        }
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            for (const ggml_tensor * f : fused) {
                if (t->src[q] == f) {
                    return 0;
                }
            }
            for (int s = 0; s < n_slots; ++s) {
                if (t->src[q] == tv[s] || t->src[q] == tc[s]) {
                    return 0;
                }
            }
        }
        // the unrelated nodes run after the kernel: they may not read or write its outputs
        if (ggml_is_empty(t) || ggml_cuda_is_view_or_noop(t) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        const ggml_tensor * ts[1 + GGML_MAX_SRC] = { t };
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            ts[1 + q] = t->src[q];
        }
        for (const ggml_tensor * u : ts) {
            for (const ggml_tensor * o : outs) {
                if (u && u->data && !ggml_is_empty(u) && overlaps(u, o)) {
                    return 0;
                }
            }
        }
    }
    bool need_barrier = false;
    if (!ggml_cuda_kda_conv_block_ok(qk, qkv, cin, conv, silu, tc, n_slots, ql, kl, &need_barrier)) {
        return 0;
    }
    // a deferred conv-state gather is read by the kernel itself: its ids may not lie under any output, its state source
    // under none but the cache write-back (the in-place row update)
    for (const auto & e : cuda_ctx->conv_state_rows) {
        if (e.first == cin && e.second) {
            for (size_t q = 0; q < outs.size(); ++q) {
                if (overlaps(outs[q], e.second->src[1]) || (q < 3 && overlaps(outs[q], e.second->src[0]))) {
                    return 0;
                }
            }
        }
    }
    if (vmm) {
        GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, vmm));
    }
    const ggml_tensor * rows = ggml_cuda_conv_state_rows_take(cuda_ctx, cin, false);
    ggml_cuda_kda_conv_block(*cuda_ctx, qk, qkv, cin, conv, silu, tv, tc, n_slots, ql, kl, need_barrier, rows);
    for (int m : middle) {
        ggml_tensor * t = cgraph->nodes[m];
        if (is_absorbed(t) || ggml_is_empty(t) || ggml_cuda_is_view_or_noop(t) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                ggml_cuda_defer_gdn_state_rows(cuda_ctx, cgraph, m)) {
            continue;
        }
        GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, t));
    }
    return is + 5 - i;
}

// consecutive MUL_MATs of one quantized type and K that share src1 (1-4 columns: decode / MTP verify), e.g. the q, k,
// v, f_a, g_a and beta projections of a GLM-5-Next KDA layer: one MMVQ launch (ggml_cuda_mul_mat_vec_q_multi). The run
// stops before a member that the node after it consumes (that member keeps its own fusions: bias add, gate/up GLU).
// GGML_CUDA_MMVQ_MULTI=0 off.
static int ggml_cuda_try_fuse_mmvq_multi(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * mm = cgraph->nodes[i];
    const ggml_type type = mm->src[0] ? mm->src[0]->type : GGML_TYPE_COUNT;
    if (mm->op != GGML_OP_MUL_MAT || (mm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !ggml_cuda_mul_mat_vec_q_multi_type_ok(type) ||
            mm->src[1]->ne[1] > 4) {
        return 0;
    }
    // the run: consecutive matmuls of this input and weight type (GLM-5.3-Flash UD-Q2_K_XL: q5_K q, then q6_K k and v)
    const ggml_tensor * mms[MMVQ_MULTI_MAX];
    int n = 0;
    for (int j = i; j < cgraph->n_nodes && n < MMVQ_MULTI_MAX; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        if (c->op != GGML_OP_MUL_MAT || c->src[1] != mm->src[1] || c->src[0]->type != type ||
                (c->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            break;
        }
        mms[n++] = c;
    }
    while (n >= 2 && i + n < cgraph->n_nodes) {
        const ggml_tensor * nx = cgraph->nodes[i + n];
        bool uses = false;
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            uses = uses || nx->src[q] == mms[n - 1];
        }
        if (!uses) {
            break;
        }
        --n;
    }
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    if (n < 2 || !ggml_cuda_should_use_mmvq(type, cc, mm->src[1]->ne[1]) ||
            !ggml_cuda_mul_mat_vec_q_multi_supported(mms, n)) {
        return 0;
    }
    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    for (int k = 0; k < n; ++k) {
        if (overlaps(mms[k], mm->src[1])) {
            return 0;
        }
        for (int q = 0; q < k; ++q) {
            if (overlaps(mms[k], mms[q])) {
                return 0;
            }
        }
    }
    ggml_cuda_mul_mat_vec_q_multi(*cuda_ctx, mms, n);
    return n - 1;
}

// dense q4_K / q5_K gate/up + SwiGLU(-clamp) at 1-2 tokens on GCN (gcn_kq_mv1): checked before the multi-MMVQ sibling
// fusion, which otherwise takes the gate projection alone (GLM-5.3's shared expert at the 2-token MTP verify ran gate, up
// and the GLU as three launches). Returns 2 (nodes consumed after i) or 0.
static int ggml_cuda_try_fuse_kqmv1_glu(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    // structure only: the kernel reads the q8_1 copy of src1 made before it runs, so the GLU output may lie over the dead
    // src1 (the allocator does that for GLM-5.3's shared expert; ggml_cuda_can_fuse allows it for single-column MMVQ only)
    if (i + 2 >= cgraph->n_nodes || !ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, { i + 2 })) {
        return 0;
    }
    ggml_tensor * glu  = cgraph->nodes[i + 2];
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up   = glu->src[1];
    if (!((gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i])) ||
            ggml_get_op_params_i32(glu, 1) != 0 ||
            (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU && ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU_CLAMP) ||
            gate->src[1] != up->src[1]) {
        return 0;
    }
    const float limit = ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP ? ggml_get_op_params_f32(glu, 3) : INFINITY;
    return ggml_cuda_gcn_kq_matvec_glu(*cuda_ctx, up->src[0], gate->src[0], up->src[1], glu, limit) ? 2 : 0;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL) {
        const int n_hc = ggml_cuda_try_fuse_hc_gate(cuda_ctx, cgraph, i);
        if (n_hc > 0) {
            return n_hc;
        }
        const int n_kg = ggml_cuda_try_fuse_kda_gate(cuda_ctx, cgraph, i);
        if (n_kg > 0) {
            return n_kg;
        }
    }

    if (node->op == GGML_OP_ADD) {
        const int n_kg = ggml_cuda_try_fuse_kda_gate(cuda_ctx, cgraph, i);
        if (n_kg > 0) {
            return n_kg;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID) {
        const int n_moe1 = ggml_cuda_try_fuse_moe1(cuda_ctx, cgraph, i);
        if (n_moe1 > 0) {
            return n_moe1;
        }
    }

    if (node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_CONCAT) {
        const int n_rows = ggml_cuda_try_fuse_dsv4_rows(cuda_ctx, cgraph, i);
        if (n_rows > 0) {
            return n_rows;
        }
    }
    if (node->op == GGML_OP_GET_ROWS) {
        const int n_ape = ggml_cuda_try_fuse_dsv4_ape(cuda_ctx, cgraph, i);
        if (n_ape > 0) {
            return n_ape;
        }
    }
    if (node->op == GGML_OP_SET_ROWS) {
        const int n_sr = ggml_cuda_try_fuse_dsv4_set_rows(cuda_ctx, cgraph, i);
        if (n_sr > 0) {
            return n_sr;
        }
    }

    if (node->op == GGML_OP_DSV4_HC_POST) {
        const int n_step = ggml_cuda_try_fuse_hc_step(cuda_ctx, cgraph, i);
        if (n_step > 0) {
            return n_step;
        }
    }

    if (node->op == GGML_OP_MUL_MAT) {
        const int n_r1 = ggml_cuda_try_fuse_router1(cuda_ctx, cgraph, i);
        if (n_r1 > 0) {
            return n_r1;
        }
    }

    if (node->op == GGML_OP_RMS_NORM) {
        const int n_ng = ggml_cuda_try_fuse_norm_gemv1(cuda_ctx, cgraph, i);
        if (n_ng > 0) {
            return n_ng;
        }
    }

    if (node->op == GGML_OP_MUL_MAT) {
        const int n_gm = ggml_cuda_try_fuse_gemv1_multi(cuda_ctx, cgraph, i);
        if (n_gm != 0) {
            return n_gm;
        }
        const int n_fm = ggml_cuda_try_fuse_f16mv_multi(cuda_ctx, cgraph, i);
        if (n_fm != 0) {
            return n_fm;
        }
        const int n_kg = ggml_cuda_try_fuse_kqmv1_glu(cuda_ctx, cgraph, i);
        if (n_kg > 0) {
            return n_kg;
        }
        const int n_mm = ggml_cuda_try_fuse_mmvq_multi(cuda_ctx, cgraph, i);
        if (n_mm > 0) {
            return n_mm;
        }
    }

    if (node->op == GGML_OP_CONCAT) {
        const int n_kc = ggml_cuda_try_fuse_kda_conv(cuda_ctx, cgraph, i);
        if (n_kc > 0) {
            return n_kc;
        }
    }

    if (node->op == GGML_OP_CONCAT && !cuda_ctx->conv_state_rows.empty() && !(node->src[1]->ne[0] <= 16)) {
        ggml_cuda_conv_state_rows_take(cuda_ctx, node, true);
    }

    // decode HC pre as one persistent kernel: [shared expert ->] [inject ->] [combine ->] RMSNorm ... gated DSV4_HC_PRE
    // [-> router logits] (GGML_CUDA_HC_PERSIST=1..4)
    if (node->op == GGML_OP_MUL_MAT && node == cuda_ctx->hcp_shexp_node && i + 8 < cgraph->n_nodes &&
            cgraph->nodes[i + 8] == cuda_ctx->hcp_inject_node && ggml_cuda_hc_persist_shexp_match(cuda_ctx->device, cgraph, i)) {
        const int ext = ggml_cuda_hc_persist_ffn_ext(cgraph, i + 13);
        ggml_cuda_hc_persist_shexp(*cuda_ctx, cgraph, i);
        return 22 + ext;
    }
    if (node->op == GGML_OP_MUL_MAT && node == cuda_ctx->hcp_shexp_node && ggml_cuda_hc_persist_shexp_local_match(cgraph, i)) {
        ggml_cuda_hc_persist_shexp_local(*cuda_ctx, cgraph, i);
        cuda_ctx->hcp_shexp_node = nullptr;
        return 7;
    }
    if (node->op == GGML_OP_MUL_MAT && node == cuda_ctx->hcp_inject_node &&
            ggml_cuda_hc_persist_post_match(cuda_ctx->device, cgraph, i + 1, i)) {
        const int ext = ggml_cuda_hc_persist_ffn_ext(cgraph, i + 5);
        ggml_cuda_hc_persist_post(*cuda_ctx, cgraph, i + 1, true);
        return 14 + ext;
    }
    if (node->op == GGML_OP_MUL_MAT && node != cuda_ctx->hcp_inject_node && ggml_cuda_hc_persist_inject_other(*cuda_ctx, cgraph, i)) {
        const int ext = ggml_cuda_hc_persist_ffn_ext(cgraph, i + 5);
        ggml_cuda_hc_persist_post(*cuda_ctx, cgraph, i + 1, true);
        return 14 + ext;
    }
    if (node->op == GGML_OP_SCALE && ggml_cuda_hc_persist_post_match(cuda_ctx->device, cgraph, i, -1)) {
        const int ext = ggml_cuda_hc_persist_ffn_ext(cgraph, i + 4);
        ggml_cuda_hc_persist_post(*cuda_ctx, cgraph, i, false);
        return 13 + ext;
    }
    if (node->op == GGML_OP_RMS_NORM && ggml_cuda_hc_persist_match(cuda_ctx->device, cgraph, i)) {
        const int ext = ggml_cuda_hc_persist_ffn_ext(cgraph, i);
        ggml_cuda_hc_persist(*cuda_ctx, cgraph, i);
        return 9 + ext;
    }

    // indexer sparse mask from the top-k (long contexts): on gfx906 prefill the mask chain and the attention in one pass
    // (default on; GGML_CUDA_FA_QSA_GCN=0 off, fattn-qsa-gcn.cu), else the mask in one kernel for the dense attention
    if (node->op == GGML_OP_CONT && node->src[0]->op == GGML_OP_TOP_K) {
        int n = ggml_cuda_qsa_attn_gcn(*cuda_ctx, cgraph, i);
        if (n <= 0) {
            n = ggml_cuda_hc_qsa_mask(*cuda_ctx, cgraph, i);
        }
        if (n > 0) {
            return n;
        }
    }

    // PLE depthwise conv tail: the tap copies, multiplies and adds, SiLU and both residual adds (opt-in)
    if (node->op == GGML_OP_CONT && node->src[0]->op == GGML_OP_PERMUTE) {
        const int n = ggml_cuda_hc_ple_conv(*cuda_ctx, cgraph, i);
        if (n > 0) {
            return n;
        }
    }

    // attention output gate: CONT(gate view) -> SIGMOID -> MUL(attn) in one kernel that also writes the q8_1 input of the
    // output projection that follows
    if (node->op == GGML_OP_CONT) {
        int mm_idx = -1;
        const int n = ggml_cuda_hc_attn_gate_match(cgraph, i, &mm_idx);
        if (n > 0) {
            ggml_cuda_hc_attn_gate(*cuda_ctx, cgraph, i, mm_idx);
            return n - 1;
        }
    }

    // decode attention q/k chain: q/k norm + rope + k/v cache rows in one kernel
    if (node->op == GGML_OP_RMS_NORM && node->ne[0] == 256) {
        const int n = ggml_cuda_attn_qk_fused(*cuda_ctx, cgraph, i, [](ggml_backend_cuda_context & ctx, ggml_tensor * t) {
            // a projection the persistent kernel already computed, or a plain op now
            return ggml_cuda_hc_persist_done(ctx, t) > 0 || ggml_cuda_compute_forward(ctx, t);
        });
        if (n > 0) {
            return n;
        }
    }

    if (node->op == GGML_OP_RMS_NORM && node->ne[0] == 128) {
        int mm_idx = -1, z_mm = -1;
        const int n = ggml_cuda_hc_gdn_gated_norm_match(cgraph, i, &mm_idx, &z_mm);
        if (n > 0) {
            if (z_mm >= 0) {
                // the gate's up projection first (it does not read the norm), then the fused norm and gate
                ggml_cuda_compute_forward(*cuda_ctx, cgraph->nodes[z_mm]);
            }
            ggml_cuda_hc_gdn_gated_norm(*cuda_ctx, cgraph, i, mm_idx, z_mm >= 0 ? 1 : 0);
            return n - 1;
        }
    }

    // RMS_NORM -> SCALE and SCALE -> SILU pairs (per-head q/k norms, the HC low-rank path): one kernel each; mostly
    // saves kernel launches in decode
    static const bool no_small_fusions = [] { const char * e = getenv("GGML_CUDA_NO_SMALL_FUSIONS"); return e && atoi(e) != 0; }();
    // decode HC up projection: SCALE -> SILU -> MUL_MAT -> RESHAPE -> gated DSV4_HC_PRE in one kernel (GCN)
    if (node->op == GGML_OP_SCALE && i + 4 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[i + 1]) == GGML_UNARY_OP_SILU &&
            cgraph->nodes[i + 1]->src[0] == node && cgraph->nodes[i + 2]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[i + 3]->op == GGML_OP_RESHAPE && cgraph->nodes[i + 3]->src[0] == cgraph->nodes[i + 2] &&
            cgraph->nodes[i + 4]->op == GGML_OP_DSV4_HC_PRE && cgraph->nodes[i + 4]->src[1] == cgraph->nodes[i + 3] &&
            ggml_cuda_hc_up_pre_dec_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, node, cgraph->nodes[i + 1],
                                              cgraph->nodes[i + 2], cgraph->nodes[i + 4]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE },
                                   { i + 4 })) {
        ggml_cuda_hc_up_pre_dec(*cuda_ctx, node, cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    // qwen4exp indexer epilogue: RELU(scores) -> CONT(head 0) + ADD(heads 1..) -> ADD(block bias) -> CONT(PERMUTE) ->
    // GET_ROWS(cell_blk) -> CONT(PERMUTE) -> [CPY mask to F32 ->] ADD(mask), up to the TOP_K: one kernel
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_RELU && node->type == GGML_TYPE_F32) {
        static const bool enabled = [] { const char * e = getenv("GGML_CUDA_IDX_EXPAND_FUSE"); return !e || atoi(e) != 0; }();
        const ggml_tensor * relu = node;
        int i_exp = -1;
        for (int j = i + 1; enabled && j < std::min(cgraph->n_nodes, i + 48); ++j) {
            if (cgraph->nodes[j]->op == GGML_OP_TOP_K) {
                const ggml_tensor * e = cgraph->nodes[j]->src[0];
                for (int k = i + 1; k < j; ++k) {
                    if (cgraph->nodes[k] == e) {
                        i_exp = k;
                    }
                }
                break;
            }
        }
        const ggml_tensor * exp = i_exp > 0 ? cgraph->nodes[i_exp] : nullptr;
        std::vector<const ggml_tensor *> tree = { relu };
        const auto op_is = [](const ggml_tensor * t, ggml_op op) { return t && t->op == op; };
        const ggml_tensor * score = nullptr, * bias = nullptr, * cell_blk = nullptr, * mask = nullptr;
        bool ok = exp && exp->op == GGML_OP_ADD && exp->type == GGML_TYPE_F32 && relu->src[0]->type == GGML_TYPE_F32 &&
            relu->ne[3] == 1 && ggml_is_contiguous(relu) && ggml_is_contiguous(relu->src[0]);
        if (ok) {
            score = relu->src[0];
            tree.push_back(exp);
            const ggml_tensor * ce = exp->src[0], * mr = exp->src[1];
            ok = op_is(ce, GGML_OP_CONT) && op_is(ce->src[0], GGML_OP_PERMUTE) && op_is(ce->src[0]->src[0], GGML_OP_GET_ROWS);
            if (ok) {
                const ggml_tensor * p2 = ce->src[0], * g = p2->src[0];
                const ggml_tensor * c1 = g->src[0];
                cell_blk = g->src[1];
                tree.insert(tree.end(), { ce, p2, g });
                ok = op_is(c1, GGML_OP_CONT) && op_is(c1->src[0], GGML_OP_PERMUTE) && op_is(c1->src[0]->src[0], GGML_OP_ADD) &&
                     cell_blk->type == GGML_TYPE_I32 && cell_blk->ne[0] == exp->ne[0] && ggml_is_contiguous(cell_blk) &&
                     g->ne[2] == 1 && g->ne[3] == 1;
                if (ok) {
                    const ggml_tensor * p1 = c1->src[0], * sb = p1->src[0];
                    tree.insert(tree.end(), { c1, p1, sb });
                    bias = sb->src[1];
                    // head sum: ADD(...ADD(CONT(VIEW h0), VIEW h1)..., VIEW h_{H-1})
                    std::vector<const ggml_tensor *> heads;
                    const ggml_tensor * x = sb->src[0];
                    while (op_is(x, GGML_OP_ADD)) {
                        tree.push_back(x);
                        heads.push_back(x->src[1]);
                        x = x->src[0];
                    }
                    ok = op_is(x, GGML_OP_CONT) && bias->type == GGML_TYPE_F32 && bias->nb[0] == sizeof(float) &&
                         bias->ne[0] == relu->ne[0] && bias->ne[1] == relu->ne[2] && bias->ne[2] == 1 && bias->ne[3] == 1;
                    if (ok) {
                        tree.push_back(x);
                        heads.push_back(x->src[0]);
                        std::reverse(heads.begin(), heads.end());
                        ok = (int64_t) heads.size() == relu->ne[1];
                        for (size_t h = 0; ok && h < heads.size(); ++h) {
                            const ggml_tensor * v = heads[h];
                            ok = op_is(v, GGML_OP_VIEW) && v->src[0] == relu && v->view_offs == h*relu->nb[1] &&
                                 v->ne[0] == relu->ne[0] && v->ne[1] == relu->ne[2] && v->nb[1] == relu->nb[2];
                            tree.push_back(v);
                        }
                    }
                }
            }
            if (ok) {
                // mask: RESHAPE(CPY(f16 mask -> f32)) or RESHAPE(f32 mask)
                const ggml_tensor * m = mr;
                if (op_is(m, GGML_OP_RESHAPE)) {
                    tree.push_back(m);
                    m = m->src[0];
                }
                if (op_is(m, GGML_OP_CPY) && m->type == GGML_TYPE_F32 && m->src[0]->type == GGML_TYPE_F16 &&
                        ggml_are_same_shape(m, m->src[0])) {
                    tree.push_back(m);
                    m = m->src[0];
                }
                mask = m;
                ok = (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) && mask->ne[0] == exp->ne[0] &&
                     mask->ne[1] >= exp->ne[1] && mask->nb[0] == ggml_type_size(mask->type) && exp->ne[2] == 1 && exp->ne[3] == 1 &&
                     exp->nb[0] == sizeof(float);
            }
        }
        // every node from the RELU to the final ADD is part of the chain, used only inside it; the mask cast may also
        // feed the attention: then it still runs (as a normal node), AFTER the fused kernel.
        // ggml_cast sets src[1] = result (ggml.c), which ggml_visit_parents counts as a use: a single-use cast looked
        // shared, ran first, and its output (placed by the allocator over the score buffer, dead in the unfused order)
        // overwrote the scores the kernel then read -> 0/-inf scores, top-k = the first 2048 cells: every QSA prefill past
        // ~2K context attended to the wrong cells (MI50 16K PPL 4.13 vs 2.07). Count real uses; run a shared cast last.
        const auto real_uses = [&](int k) {
            const ggml_tensor * t = cgraph->nodes[k];
            return ggml_node_get_use_count(cgraph, k) - (t->op == GGML_OP_CPY && t->src[1] == t ? 1 : 0);
        };
        ggml_tensor * mask_cpy = nullptr;
        for (int k = i; ok && k <= i_exp; ++k) {
            ggml_tensor * t = cgraph->nodes[k];
            if (std::find(tree.begin(), tree.end(), t) == tree.end()) {
                ok = false;
                break;
            }
            if (k < i_exp && t->op == GGML_OP_CPY && real_uses(k) > 1) {
                mask_cpy = t;
                continue;
            }
            if (k < i_exp && ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || real_uses(k) != (t == relu ? (int) relu->ne[1] : 1))) {
                ok = false;
            }
        }
        if (ok) {
            // out may not overlap the inputs (other threads read score/bias/mask rows it would overwrite)
            const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
                const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
                return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
            };
            ok = !overlaps(exp, score) && !overlaps(exp, bias) && !overlaps(exp, cell_blk) && !overlaps(exp, mask) &&
                 (!mask_cpy || !overlaps(exp, mask_cpy));
        }
        if (ok) {
            ggml_cuda_idx_score_expand(*cuda_ctx, score, bias, cell_blk, mask, cgraph->nodes[i_exp]); // reads the f16 mask itself
            if (mask_cpy && (mask_cpy->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                // after the kernel: its buffer may lie over the (then dead) score
                GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, mask_cpy));
            }
            return i_exp - i;
        }
    }

    // decode / verify gated-DeltaNet conv block: CONCAT, [VIEW tail -> CONT -> VIEW cache -> CPY] per rollback slot,
    // unrelated nodes (the recurrent state gather), SSM_CONV -> SILU, VIEW q -> RMS_NORM -> SCALE, VIEW k -> RMS_NORM ->
    // SCALE: one kernel, then the unrelated nodes in their order (they may not touch the fused outputs)
    if (node->op == GGML_OP_CONCAT && node->src[1]->ne[0] <= 16) {
        const int W = std::min(cgraph->n_nodes, i + 64);
        const ggml_tensor * tv[8], * tc[8];
        int n_slots = 0;
        int j = i + 1;
        bool ok = true;
        while (ok && j + 3 < W && cgraph->nodes[j]->op == GGML_OP_VIEW && cgraph->nodes[j]->src[0] == node) {
            const ggml_tensor * cont = cgraph->nodes[j + 1], * dv = cgraph->nodes[j + 2], * cpy = cgraph->nodes[j + 3];
            ok = n_slots < 8 && cont->op == GGML_OP_CONT && cont->src[0] == cgraph->nodes[j] && dv->op == GGML_OP_VIEW &&
                 cpy->op == GGML_OP_CPY && cpy->src[0] == cont && cpy->src[1] == dv &&
                 cgraph->nodes[j]->ne[0] == 3 && cgraph->nodes[j]->nb[1] == node->nb[1] && ggml_is_contiguous(cont) &&
                 ggml_node_get_use_count(cgraph, j) == 1 && ggml_node_get_use_count(cgraph, j + 1) == 1;
            if (ok) {
                tv[n_slots] = cgraph->nodes[j];
                tc[n_slots] = cpy;
                n_slots++;
                j += 4;
            }
        }
        std::vector<int> middle;
        int is = -1;
        for (; ok && n_slots > 0 && j < W; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            if (t->op == GGML_OP_SSM_CONV && t->src[0] == node) {
                is = j;
                break;
            }
            for (int k = 0; k < GGML_MAX_SRC && ok; ++k) {
                if (t->src[k] == node) {
                    ok = false;
                }
                for (int q = 0; q < n_slots && ok; ++q) {
                    if (t->src[k] == tv[q] || t->src[k] == tc[q]->src[0]) {
                        ok = false;
                    }
                }
            }
            middle.push_back(j);
        }
        ok = ok && is > 0 && is + 7 < cgraph->n_nodes && ggml_node_get_use_count(cgraph, i) == n_slots + 1;
        bool need_barrier = false;
        if (ok) {
            ggml_tensor * silu = cgraph->nodes[is + 1];
            const ggml_tensor * qv = cgraph->nodes[is + 2], * qr = cgraph->nodes[is + 3], * kv = cgraph->nodes[is + 5], * kr = cgraph->nodes[is + 6];
            ggml_tensor * qs = cgraph->nodes[is + 4], * ks = cgraph->nodes[is + 7];
            ok = silu->op == GGML_OP_UNARY && silu->src[0] == cgraph->nodes[is] && ggml_node_get_use_count(cgraph, is) == 1 &&
                 qv->op == GGML_OP_VIEW && qv->src[0] == silu && qr->op == GGML_OP_RMS_NORM && qr->src[0] == qv &&
                 qs->op == GGML_OP_SCALE && qs->src[0] == qr &&
                 kv->op == GGML_OP_VIEW && kv->src[0] == silu && kr->op == GGML_OP_RMS_NORM && kr->src[0] == kv &&
                 ks->op == GGML_OP_SCALE && ks->src[0] == kr &&
                 ggml_node_get_use_count(cgraph, is + 2) == 1 && ggml_node_get_use_count(cgraph, is + 3) == 1 &&
                 ggml_node_get_use_count(cgraph, is + 5) == 1 && ggml_node_get_use_count(cgraph, is + 6) == 1 &&
                 !(qr->flags & GGML_TENSOR_FLAG_OUTPUT) && !(kr->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                 !(cgraph->nodes[is]->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                 ggml_cuda_gdn_conv_block_dec_ok(node, cgraph->nodes[is], silu, tc, n_slots, qr, qs, kr, ks, &need_barrier);
            // the unrelated nodes run after the kernel: they may not read or write its outputs
            const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
                const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
                return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
            };
            std::vector<const ggml_tensor *> outs = { silu, qs, ks };
            for (int q = 0; q < n_slots; ++q) {
                outs.push_back(tc[q]->src[1]);
            }
            // a deferred state gather is read by the kernel itself: its ids may not lie under any output, its state source
            // under none but the cache write-back (outs[3..], the in-place row update)
            for (const auto & e : cuda_ctx->conv_state_rows) {
                if (ok && e.first == node && e.second) {
                    for (size_t q = 0; q < outs.size(); ++q) {
                        if (overlaps(outs[q], e.second->src[1]) || (q < 3 && overlaps(outs[q], e.second->src[0]))) {
                            ok = false;
                        }
                    }
                }
            }
            for (int m : middle) {
                const ggml_tensor * t = cgraph->nodes[m];
                if (!ok || ggml_is_empty(t) || ggml_cuda_is_view_or_noop(t) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }
                const ggml_tensor * ts[1 + GGML_MAX_SRC] = { t };
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    ts[1 + k] = t->src[k];
                }
                for (const ggml_tensor * u : ts) {
                    for (const ggml_tensor * o : outs) {
                        if (u && u->data && !ggml_is_empty(u) && overlaps(u, o)) {
                            ok = false;
                        }
                    }
                }
            }
            if (ok) {
                const ggml_tensor * rows = ggml_cuda_conv_state_rows_take(cuda_ctx, node, false);
                ggml_cuda_gdn_conv_block_dec(*cuda_ctx, node, cgraph->nodes[is], silu, tv, tc, n_slots, qr, qs, kr, ks, need_barrier, rows);
                for (int m : middle) {
                    ggml_tensor * t = cgraph->nodes[m];
                    if (ggml_is_empty(t) || ggml_cuda_is_view_or_noop(t) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                            ggml_cuda_defer_gdn_state_rows(cuda_ctx, cgraph, m)) {
                        continue;
                    }
                    GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, t));
                }
                return is + 7 - i;
            }
        }
    }
    if (node->op == GGML_OP_CONCAT && !cuda_ctx->conv_state_rows.empty()) {
        ggml_cuda_conv_state_rows_take(cuda_ctx, node, true); // not fused by the conv block: gather now
    }

    // shared-expert gate tail: MUL_MAT(1 row) -> SIGMOID -> MUL -> ADD
    if (node->op == GGML_OP_MUL_MAT && i + 3 < cgraph->n_nodes && node->src[0]->ne[1] == 1 &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[i + 1]) == GGML_UNARY_OP_SIGMOID &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL && cgraph->nodes[i + 3]->op == GGML_OP_ADD &&
            ggml_cuda_shexp_gate_add_supported(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD }, { i + 3 })) {
        ggml_cuda_shexp_gate_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3]);
        return 3;
    }

    // decode GDN gates: MUL_MAT(alpha) -> RESHAPE -> ADD -> SOFTPLUS -> MUL -> RESHAPE and MUL_MAT(beta) -> RESHAPE -> SIGMOID
    if (node->op == GGML_OP_MUL_MAT && i + 8 < cgraph->n_nodes && cgraph->nodes[i + 3]->op == GGML_OP_UNARY &&
            cgraph->nodes[i + 8]->op == GGML_OP_UNARY &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
                                   GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY }, { i + 5, i + 8 }) &&
            ggml_cuda_gdn_ab_dec_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, node, cgraph->nodes[i + 2],
                cgraph->nodes[i + 3], cgraph->nodes[i + 4], cgraph->nodes[i + 6], cgraph->nodes[i + 8])) {
        ggml_cuda_gdn_ab_dec(*cuda_ctx, node, cgraph->nodes[i + 2], cgraph->nodes[i + 6], cgraph->nodes[i + 4], cgraph->nodes[i + 8]);
        return 8;
    }

    if (!no_small_fusions && i + 1 < cgraph->n_nodes) {
        ggml_tensor * nx = cgraph->nodes[i + 1];
        if (node->op == GGML_OP_RMS_NORM && nx->op == GGML_OP_SCALE && nx->src[0] == node &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, { i + 1 }) &&
                ggml_cuda_rms_norm_scale_narrow(*cuda_ctx, node, nx)) {
            return 1;
        }
        if (node->op == GGML_OP_SCALE && nx->op == GGML_OP_UNARY && ggml_get_unary_op(nx) == GGML_UNARY_OP_SILU &&
                nx->src[0] == node && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY }, { i + 1 }) &&
                ggml_cuda_scale_silu(*cuda_ctx, node, nx)) {
            return 1;
        }
    }

    // inject logits -> SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST [-> RMSNorm -> MUL -> 2x RESHAPE -> HC down]:
    // the combine computes its stream weights ps2*sigmoid(ps1*x) itself (3 fewer tiny kernels per module, which
    // matters for decode)
    if (node->op == GGML_OP_SCALE && i + 3 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[i + 1]) == GGML_UNARY_OP_SIGMOID &&
            cgraph->nodes[i + 1]->src[0] == node && cgraph->nodes[i + 2]->op == GGML_OP_SCALE &&
            cgraph->nodes[i + 2]->src[0] == cgraph->nodes[i + 1] && cgraph->nodes[i + 3]->op == GGML_OP_DSV4_HC_POST &&
            cgraph->nodes[i + 3]->src[2] == cgraph->nodes[i + 2] && cgraph->nodes[i + 3]->src[3] == nullptr &&
            ggml_get_op_params_f32(node, 1) == 0.0f && ggml_get_op_params_f32(cgraph->nodes[i + 2], 1) == 0.0f &&
            node->src[0]->type == GGML_TYPE_F32 && ggml_are_same_shape(node->src[0], cgraph->nodes[i + 2])) {
        static const bool no_rawpost = [] { const char * e = getenv("GGML_HC_NO_RAWPOST"); return e && atoi(e) != 0; }();
        const ggml_tensor * raw = node->src[0];
        const float ps1 = ggml_get_op_params_f32(node, 0);
        const float ps2 = ggml_get_op_params_f32(cgraph->nodes[i + 2], 0);
        const int p = i + 3, r = i + 4;
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        // the persistent HC pre kernel takes over from the RMSNorm: plain combine here
        if (!no_rawpost && ggml_cuda_hc_persist_match(cuda_ctx->device, cgraph, r) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { p })) {
            ggml_cuda_op_dsv4_hc_post_rawpost(*cuda_ctx, cgraph->nodes[p], raw, ps1, ps2);
            return 3;
        }
        if (!no_rawpost && r + 4 < cgraph->n_nodes && cgraph->nodes[r]->op == GGML_OP_RMS_NORM &&
                cgraph->nodes[r + 1]->op == GGML_OP_MUL && cgraph->nodes[r + 2]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[r + 3]->op == GGML_OP_RESHAPE && cgraph->nodes[r + 4]->op == GGML_OP_MUL_MAT &&
                cgraph->nodes[r]->src[0] == cgraph->nodes[p] &&
                (cgraph->nodes[r + 1]->src[0] == cgraph->nodes[r] || cgraph->nodes[r + 1]->src[1] == cgraph->nodes[r]) &&
                cgraph->nodes[r + 2]->src[0] == cgraph->nodes[r + 1] && cgraph->nodes[r + 4]->src[1] == cgraph->nodes[r + 2] &&
                ggml_cuda_hc_norm_down_supported(cc, cgraph->nodes[p], cgraph->nodes[r], cgraph->nodes[r + 1], cgraph->nodes[r + 4], raw) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST,
                    GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { p, r + 1, r + 2, r + 3, r + 4 })) {
            const bool f16 = ggml_cuda_xn_consumers_f16_ok(cc, cgraph, r + 1, cgraph->nodes[r + 4]) &&
                             ggml_cuda_hc_xn_f16_inplace_safe(cgraph->nodes[p], cgraph->nodes[r + 1], raw);
            ggml_cuda_hc_norm_down(*cuda_ctx, cgraph->nodes[p], cgraph->nodes[r], cgraph->nodes[r + 1], cgraph->nodes[r + 4],
                                   f16, raw, ps1, ps2);
            return r + 4 - i;
        }
        // decode / verify: combine + norm with one block per (stream, token)
        if (!no_rawpost && r + 1 < cgraph->n_nodes && cgraph->nodes[r]->op == GGML_OP_RMS_NORM &&
                cgraph->nodes[r + 1]->op == GGML_OP_MUL && cgraph->nodes[r]->src[0] == cgraph->nodes[p] &&
                (cgraph->nodes[r + 1]->src[0] == cgraph->nodes[r] || cgraph->nodes[r + 1]->src[1] == cgraph->nodes[r]) &&
                ggml_cuda_hc_combine_norm_dec_supported(cgraph->nodes[p], cgraph->nodes[r], cgraph->nodes[r + 1], raw) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST,
                    GGML_OP_RMS_NORM, GGML_OP_MUL }, { p, r + 1 })) {
            // the down projection (MUL -> RESHAPE -> q8_0 MUL_MAT on MMVQ) reads a q8_1 copy the kernel writes into the
            // q8 cache under the matmul's input tensor (no separate quantize launch)
            void * xq = nullptr;
            int64_t s_xq = 0;
            static const bool xq_env = [] { const char * e = getenv("GGML_HC_COMBINE_Q8"); return e && atoi(e) != 0; }(); // opt-in: 83 fewer quantize launches/token but tg@8K 46.8 -> 46.2 (the shuffle chain in the 1-block kernel)
            if (xq_env && r + 4 < cgraph->n_nodes && cgraph->nodes[r + 2]->op == GGML_OP_RESHAPE &&
                    cgraph->nodes[r + 2]->src[0] == cgraph->nodes[r + 1] && cgraph->nodes[r + 4]->op == GGML_OP_MUL_MAT &&
                    cgraph->nodes[r + 4]->src[1] == cgraph->nodes[r + 2] && cgraph->nodes[r + 4]->src[0]->type == GGML_TYPE_Q8_0) {
                const ggml_tensor * y = cgraph->nodes[r + 2];
                const int64_t K = y->ne[0];
                if (K == 4*cgraph->nodes[r + 1]->ne[0] && K % MATRIX_ROW_PADDING == 0 && y->ne[1] <= 8 && y->ne[2] == 1 && y->ne[3] == 1 &&
                        ggml_is_contiguous(y)) {
                    const size_t nbytes = (size_t) y->ne[1]*K/QK8_1*sizeof(block_q8_1);
                    cuda_ctx->q8_cache.push_back({ y, (int) GGML_TYPE_Q8_0, { K, y->ne[1], 1, 1 },
                                                   std::make_unique<ggml_cuda_pool_alloc<char>>(cuda_ctx->pool(), nbytes) });
                    xq   = cuda_ctx->q8_cache.back().buf->get();
                    s_xq = K/QK8_1;
                }
            }
            ggml_cuda_hc_combine_norm_dec(*cuda_ctx, cgraph->nodes[p], cgraph->nodes[r], cgraph->nodes[r + 1], raw, ps1, ps2, xq, s_xq);
            return r + 1 - i;
        }
        // no down fusion (decode, non-RDNA): combine + norm in one pass
        if (!no_rawpost && r + 1 < cgraph->n_nodes && cgraph->nodes[r]->op == GGML_OP_RMS_NORM &&
                cgraph->nodes[r + 1]->op == GGML_OP_MUL && cgraph->nodes[r]->src[0] == cgraph->nodes[p] &&
                (cgraph->nodes[r + 1]->src[0] == cgraph->nodes[r] || cgraph->nodes[r + 1]->src[1] == cgraph->nodes[r]) &&
                ggml_cuda_hc_combine_norm_supported(cgraph->nodes[p], cgraph->nodes[r], cgraph->nodes[r + 1], raw) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST,
                    GGML_OP_RMS_NORM, GGML_OP_MUL }, { p, r + 1 })) {
            ggml_cuda_hc_combine_norm(*cuda_ctx, cgraph->nodes[p], cgraph->nodes[r], cgraph->nodes[r + 1], raw, ps1, ps2);
            return r + 1 - i;
        }
        if (!no_rawpost && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { p })) {
            ggml_cuda_op_dsv4_hc_post_rawpost(*cuda_ctx, cgraph->nodes[p], raw, ps1, ps2);
            return 3;
        }
    }

    // [hyper-connection combine ->] grouped RMSNorm -> MUL -> 2x RESHAPE -> HC down projection
    if (node->op == GGML_OP_DSV4_HC_POST || node->op == GGML_OP_RMS_NORM) {
        const bool with_post = node->op == GGML_OP_DSV4_HC_POST;
        const int  r = with_post ? i + 1 : i;
        if (r + 4 < cgraph->n_nodes && cgraph->nodes[r]->op == GGML_OP_RMS_NORM && cgraph->nodes[r + 1]->op == GGML_OP_MUL &&
                cgraph->nodes[r + 2]->op == GGML_OP_RESHAPE && cgraph->nodes[r + 3]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[r + 4]->op == GGML_OP_MUL_MAT &&
                (!with_post || cgraph->nodes[r]->src[0] == node) &&
                (cgraph->nodes[r + 1]->src[0] == cgraph->nodes[r] || cgraph->nodes[r + 1]->src[1] == cgraph->nodes[r]) &&
                cgraph->nodes[r + 2]->src[0] == cgraph->nodes[r + 1] && cgraph->nodes[r + 4]->src[1] == cgraph->nodes[r + 2] &&
                ggml_cuda_hc_norm_down_supported(ggml_cuda_info().devices[cuda_ctx->device].cc,
                    with_post ? node : nullptr, cgraph->nodes[r], cgraph->nodes[r + 1], cgraph->nodes[r + 4])) {
            const bool ok = with_post ?
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_DSV4_HC_POST, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE,
                    GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { i, i + 2, i + 3, i + 4, i + 5 }) :
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_RESHAPE,
                    GGML_OP_MUL_MAT }, { i + 1, i + 2, i + 3, i + 4 });
            if (ok) {
                const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
                const bool f16 = ggml_cuda_xn_consumers_f16_ok(cc, cgraph, r + 1, cgraph->nodes[r + 4]) &&
                                 ggml_cuda_hc_xn_f16_inplace_safe(with_post ? node : nullptr, cgraph->nodes[r + 1], nullptr);
                ggml_cuda_hc_norm_down(*cuda_ctx, with_post ? node : nullptr, cgraph->nodes[r], cgraph->nodes[r + 1],
                                       cgraph->nodes[r + 4], f16);
                return r + 4 - i;
            }
        }
    }

    // SIGMOID(z) -> MUL(a, .) -> RESHAPE -> dense projection (GDN output gate -> ssm_out)
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID && i + 3 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            (cgraph->nodes[i + 1]->src[0] == node || cgraph->nodes[i + 1]->src[1] == node) &&
            cgraph->nodes[i + 2]->op == GGML_OP_RESHAPE && cgraph->nodes[i + 2]->src[0] == cgraph->nodes[i + 1] &&
            cgraph->nodes[i + 3]->op == GGML_OP_MUL_MAT && cgraph->nodes[i + 3]->src[1] == cgraph->nodes[i + 2] &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { i + 3 }) &&
            ggml_cuda_gate_proj_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, node, cgraph->nodes[i + 1],
                                          cgraph->nodes[i + 3])) {
        ggml_cuda_gate_proj(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 3]);
        return 3;
    }
    // same without the RESHAPE (attention output gate -> attn_output)
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            (cgraph->nodes[i + 1]->src[0] == node || cgraph->nodes[i + 1]->src[1] == node) &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL_MAT && cgraph->nodes[i + 2]->src[1] == cgraph->nodes[i + 1] &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_MUL_MAT }, { i + 2 }) &&
            ggml_cuda_gate_proj_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, node, cgraph->nodes[i + 1],
                                          cgraph->nodes[i + 2])) {
        ggml_cuda_gate_proj(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    // same on GCN at prefill: q8_0 GEMM with the gated mix as its epilogue (xn stays F32 there)
    if (node->op == GGML_OP_MUL_MAT && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE && cgraph->nodes[i + 1]->src[0] == node &&
            cgraph->nodes[i + 2]->op == GGML_OP_DSV4_HC_PRE && cgraph->nodes[i + 2]->src[1] == cgraph->nodes[i + 1] &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE }, { i + 2 }) &&
            ggml_cuda_gcn_hc_up_mix_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, node, cgraph->nodes[i + 2])) {
        ggml_cuda_gcn_hc_up_mix(*cuda_ctx, node, cgraph->nodes[i + 2]);
        return 2;
    }

    // hyper-connection up projection -> reshape -> gated DSV4_HC_PRE: one kernel, the gate is never stored
    if (node->op == GGML_OP_MUL_MAT && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE && cgraph->nodes[i + 1]->src[0] == node &&
            cgraph->nodes[i + 2]->op == GGML_OP_DSV4_HC_PRE && cgraph->nodes[i + 2]->src[1] == cgraph->nodes[i + 1] &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE }, { i + 2 }) &&
            ggml_cuda_hc_up_mix_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, node, cgraph->nodes[i + 2])) {
        ggml_cuda_hc_up_mix(*cuda_ctx, node, cgraph->nodes[i + 2]);
        return 2;
    }

    // F16 MoE down projection whose only consumer is the weighted expert reduction: the expert rows are stored as
    // F16 in place (half the write and the read), then the reduction runs right away
    if (node->op == GGML_OP_MUL_MAT_ID && i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(cgraph, i) == 1) {
        static const bool no_f16 = [] { const char * e = getenv("GGML_MOE_DOWN_NO_F16"); return e && atoi(e) != 0; }();
        ggml_cuda_moe_weighted_reduction_match match;
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        if (!no_f16 && ggml_cuda_match_moe_weighted_reduction(cgraph, i + 1, match) && match.experts == node &&
                ggml_is_contiguous(node) && node->ne[0] % 4 == 0 && (node->ne[1] == 8 || node->ne[1] == 10) &&
                ggml_cuda_moe_f16_supported(cc, node->src[0], node->src[1], node->src[2], node)) {
            const int output_idx = i + match.node_count;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i + 1, match.node_count, &output_idx, 1)) {
                ggml_cuda_moe_f16(*cuda_ctx, node->src[0], node->src[1], node->src[2], node, /*out16 =*/ true);
                ggml_cuda_op_moe_weighted_reduction(*cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count;
            }
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            // prefill MoE gate/up + SwiGLU(-clamp) on the expert-grouped GCN v4 GEMM: one preparation, the up launch applies
            // the GLU to the gate projection the first launch wrote into glu
            {
                const int cc_g = ggml_cuda_info().devices[cuda_ctx->device].cc;
                const bool glu_ok = glu->op == GGML_OP_GLU && ggml_get_op_params_i32(glu, 1) == 0 &&
                    (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU || ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP);
                if (glu_ok && gate->op == GGML_OP_MUL_MAT_ID && up->op == GGML_OP_MUL_MAT_ID && glu->src[0] == gate &&
                        glu->src[1] == up && gate->src[1] == up->src[1] && gate->src[2] == up->src[2] &&
                        ggml_are_same_shape(gate->src[0], up->src[0]) && ggml_are_same_shape(gate, glu) &&
                        ggml_is_contiguous(glu) &&
                        ggml_cuda_gcn_kq_moe_supported(cc_g, gate->src[0], gate->src[1], gate->src[2], glu) &&
                        ggml_cuda_gcn_kq_moe_supported(cc_g, up->src[0], up->src[1], up->src[2], glu)) {
                    const float limit = ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP ? ggml_get_op_params_f32(glu, 3) : INFINITY;
                    ggml_cuda_gcn_kq_moe(*cuda_ctx, gate->src[0], gate->src[1], gate->src[2], glu, up->src[0], limit);
                    fused_mul_mat_vec = true;
                    fused_node_count  = 3;
                    break;
                }
            }

            // GCN row-lane MoE: gate/up + SwiGLU share one preparation and the up kernel applies SwiGLU
            if (ggml_cuda_moe_vec_pair_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, gate, up, glu)) {
                ggml_cuda_moe_vec_pair(*cuda_ctx, gate, up, glu);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // large-batch MoE gate/up + SwiGLU on RDNA3: one F16-WMMA kernel for both projections
            if (ggml_cuda_moe_f16_pair_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, gate, up, glu)) {
                // SwiGLU output as F16 in place when its only consumer is the next node, a MUL_MAT_ID on the F16 path
                static const bool no_f16 = [] { const char * e = getenv("GGML_MOE_NO_F16_INPLACE"); return e && atoi(e) != 0; }();
                const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
                const bool out16 = !no_f16 && i + 3 < cgraph->n_nodes && !(glu->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    ggml_node_get_use_count(cgraph, i + 2) == 1 && cgraph->nodes[i + 3]->op == GGML_OP_MUL_MAT_ID &&
                    cgraph->nodes[i + 3]->src[1] == glu && ggml_is_contiguous(glu) &&
                    ggml_cuda_moe_f16_supported(cc, cgraph->nodes[i + 3]->src[0], glu, cgraph->nodes[i + 3]->src[2],
                                                cgraph->nodes[i + 3]);
                ggml_cuda_moe_f16_pair(*cuda_ctx, gate, up, glu, out16);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // GCN, 1-2 tokens, dense q4_K / q5_K gate/up + SwiGLU / SwiGLU-clamp (GGML_CUDA_GCN_KQ_MV1; GLM-5.3's shared
            // expert at the 2-token MTP verify, where MMVQ's fusion is single-column only)
            if (ids == nullptr && up->op == GGML_OP_MUL_MAT && gate->op == GGML_OP_MUL_MAT && glu->src[0] == gate &&
                    glu->src[1] == up && ggml_get_op_params_i32(glu, 1) == 0 &&
                    (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU || ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP) &&
                    gate->src[1] == src1 && ggml_cuda_gcn_kq_matvec_glu(*cuda_ctx, src0, gate->src[0], src1, glu,
                        ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP ? ggml_get_op_params_f32(glu, 3) : INFINITY)) {
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up) || ggml_cuda_should_fuse_mmvq_glu_cols(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        // GCN dense K-quant matvec at 2..4 tokens: the addend added as it writes (GGML_CUDA_KQMV_ADD=0 off)
        if (bias_op == GGML_OP_ADD &&
                ggml_cuda_gcn_kq_matvec_add_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, mm_node, bias_node, bias_tensor)) {
            ggml_cuda_gcn_kq_matvec(*cuda_ctx, src0, src1, bias_node, bias_tensor);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node) || (ggml_cuda_mmvq_add2_ok(mm_node, bias_node, bias_tensor) &&
                !ggml_cuda_gcn_kq_matvec_supported(ggml_cuda_info().devices[cuda_ctx->device].cc, src0, src1, mm_node))) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    return 0;
}

// GGML_CUDA_OP_PROFILE=1 (with GGML_CUDA_DISABLE_GRAPHS=1): time every node (fused groups under their first op)
// with events and print the per-op totals at exit. Synchronizes after every node, so only the shares are meaningful.
// Mode 5: like 1, but a spin kernel runs ahead of every node so all of its kernels are queued before the first one
// starts: the times are GPU execution only, without host launch gaps (decode-sized ops).
static __global__ void ggml_cuda_op_profile_spin(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}
static __global__ void ggml_cuda_op_profile_empty() {}

// GGML_CUDA_FIX_MOE_IDS=1 (measurement only): overwrite the expert ids of every MUL_MAT_ID with a fixed pattern of
// distinct experts per token. Skip tests (GGML_CUDA_SKIP_OPS) leave garbage activations that route every token of a
// batch to the same experts, which makes the MoE faster and charges the saving to the skipped op.
static __global__ void ggml_cuda_fix_moe_ids(int32_t * ids, const int n_used, const int n_tok, const int64_t s1,
                                             const int n_expert) {
    const int k = threadIdx.x;
    const int t = blockIdx.x;
    if (k < n_used && t < n_tok) {
        ids[t*s1 + k] = (t*389 + k*97) % n_expert; // 97 is odd: distinct for k < n_expert
    }
}

// skip tests of index-producing ops (GGML_CUDA_SKIP_OPS): valid stand-in indices
static __global__ void ggml_cuda_skip_iota(int32_t * dst, const int64_t n, const int64_t row) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (int32_t) (i % row);
    }
}

struct ggml_cuda_op_profile {
    struct entry { double ms = 0; int64_t n = 0; };
    std::map<std::string, entry> totals;
    double dev_ms[GGML_CUDA_MAX_DEVICES] = {};
    double cal_us[GGML_CUDA_MAX_DEVICES] = {}; // mode 5: median time of an empty kernel through the same spin + events
    bool enabled = false;
    cudaEvent_t ev0[GGML_CUDA_MAX_DEVICES] = {}, ev1[GGML_CUDA_MAX_DEVICES] = {};
    // mode 3: no sync per op; event pairs are resolved once per graph (true GPU time while the queue stays full)
    bool async = false;
    bool head_spin = false;
    std::vector<cudaEvent_t> pool[GGML_CUDA_MAX_DEVICES];
    size_t pool_used[GGML_CUDA_MAX_DEVICES] = {};
    struct pend { cudaEvent_t a, b; std::string key; std::string name; };
    // GGML_CUDA_OP_PROFILE_TRACE=d (async modes): print device d's nodes in execution order with their GPU time,
    // skipping the first GGML_CUDA_OP_PROFILE_TRACE_SKIP nodes, then GGML_CUDA_OP_PROFILE_TRACE_N lines (default 400)
    int trace_dev = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE_TRACE"); return e ? atoi(e) : -1; }();
    int64_t trace_skip = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE_TRACE_SKIP"); return e ? (int64_t) atoll(e) : (int64_t) 0; }();
    int64_t trace_left = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE_TRACE_N"); return e ? (int64_t) atoll(e) : (int64_t) 400; }();
    std::vector<pend> pending[GGML_CUDA_MAX_DEVICES];
    cudaEvent_t take(int dev) {
        if (pool_used[dev] == pool[dev].size()) {
            cudaEvent_t e;
            CUDA_CHECK(cudaEventCreate(&e));
            pool[dev].push_back(e);
        }
        return pool[dev][pool_used[dev]++];
    }
    void flush(int dev) {
        if (pending[dev].empty()) {
            return;
        }
        CUDA_CHECK(cudaEventSynchronize(pending[dev].back().b));
        for (auto & p : pending[dev]) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, p.a, p.b));
            dev_ms[dev] += ms;
            if (dev == trace_dev && trace_left > 0) {
                if (trace_skip > 0) {
                    trace_skip--;
                } else {
                    trace_left--;
                    fprintf(stderr, "TR %8.1f us  %-28s %s\n", 1000*ms, p.name.c_str(), p.key.c_str());
                }
            }
            auto & e = totals[p.key];
            e.ms += ms;
            e.n++;
        }
        pending[dev].clear();
        pool_used[dev] = 0;
    }
    ggml_cuda_op_profile() {
        const char * e = getenv("GGML_CUDA_OP_PROFILE");
        enabled = e && atoi(e) != 0;
        async   = e && (atoi(e) == 3 || atoi(e) == 4 || atoi(e) == 6);
        // mode 6: like 3, but one spin kernel at the start of each graph (GGML_CUDA_OP_PROFILE_SPIN_US per node, default
        // 20) lets the host queue every node before the GPU starts: back-to-back execution with warm caches and no
        // per-node sync, the closest to a CUDA graph (mode 5 measures each node alone after a sync: cold L2)
        head_spin = e && atoi(e) == 6;
    }
    ~ggml_cuda_op_profile() {
        if (totals.empty()) {
            return;
        }
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
            if (dev_ms[d] > 0) {
                fprintf(stderr, "ggml_cuda_op_profile: device %d busy %.1f ms\n", d, dev_ms[d]);
            }
            if (cal_us[d] > 0) {
                fprintf(stderr, "ggml_cuda_op_profile: device %d calibration %.2f us\n", d, cal_us[d]);
            }
        }
        std::vector<std::pair<std::string, entry>> v(totals.begin(), totals.end());
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.ms > b.second.ms; });
        double sum = 0;
        for (auto & kv : v) {
            sum += kv.second.ms;
        }
        fprintf(stderr, "ggml_cuda_op_profile: total %.1f ms\n", sum);
        for (auto & kv : v) {
            fprintf(stderr, "  %6.2f%%  %9.2f ms  %7lld  %s\n", 100*kv.second.ms/sum, kv.second.ms, (long long) kv.second.n, kv.first.c_str());
        }
    }
    static std::string key(const ggml_tensor * node, int n_fused) {
        std::string k = ggml_op_name(node->op);
        if (node->op == GGML_OP_UNARY) {
            k += std::string(".") + ggml_unary_op_name(ggml_get_unary_op(node));
        } else if (node->op == GGML_OP_GLU) {
            k += std::string(".") + ggml_glu_op_name(ggml_get_glu_op(node));
        }
        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0]) {
            k += std::string(" ") + ggml_type_name(node->src[0]->type) + " " + std::to_string(node->src[0]->ne[1]) + "x" +
                 std::to_string(node->src[0]->ne[0]);
        }
        if (node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_ADD || node->op == GGML_OP_MUL || node->op == GGML_OP_CONT ||
                node->op == GGML_OP_SCALE || node->op == GGML_OP_UNARY) {
            k += " [" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "]";
        }
        if (n_fused > 1) {
            k += " [fused " + std::to_string(n_fused) + "]";
        }
        // GGML_CUDA_OP_PROFILE_SRC=1: also the name, dst shape and every source's type/shape (c = contiguous)
        static const bool src_env = getenv("GGML_CUDA_OP_PROFILE_SRC") != nullptr;
        if (src_env) {
            auto sh = [](const ggml_tensor * t) {
                return std::string(ggml_type_name(t->type)) + "[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
                    std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) + "]" + (ggml_is_contiguous(t) ? "c" : "");
            };
            k += std::string(" '") + node->name + "' -> " + sh(node);
            for (int i = 0; i < GGML_MAX_SRC && node->src[i]; ++i) {
                k += " " + sh(node->src[i]);
            }
        }
        return k;
    }
};
static ggml_cuda_op_profile g_op_profile;

// GGML_CUDA_OP_PROFILE_SRC=1, fused groups: the key of the group's first node shows only that node's tensors, so the
// group's memory traffic is invisible to tools. Appended: " || in <external inputs> out <external outputs>
// mm <matmul/attention nodes: op src0 src1>". Inputs are sources not produced inside the group (views of group nodes
// count as inside); outputs are group nodes read by a later node or flagged as graph outputs, plus the last node.
// Cached per first node (graphs are reused, so their nodes keep their addresses).
static std::string ggml_cuda_op_profile_group_io(const ggml_cgraph * cgraph, int i0, int n) {
    static std::unordered_map<const ggml_tensor *, std::string> cache;
    const ggml_tensor * first = cgraph->nodes[i0];
    const auto it = cache.find(first);
    if (it != cache.end()) {
        return it->second;
    }
    auto sh = [](const ggml_tensor * t) {
        return std::string(ggml_type_name(t->type)) + "[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
            std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) + "]";
    };
    const int i1 = std::min(i0 + n, cgraph->n_nodes);
    auto in_group = [&](const ggml_tensor * t) {
        for (int j = i0; j < i1; ++j) {
            const ggml_tensor * g = cgraph->nodes[j];
            if (t == g || (t->view_src && t->view_src == g)) {
                return true;
            }
        }
        return false;
    };
    std::vector<const ggml_tensor *> ins, outs;
    auto add_unique = [](std::vector<const ggml_tensor *> & v, const ggml_tensor * t) {
        const ggml_tensor * base = t->view_src ? t->view_src : t;
        for (const ggml_tensor * u : v) {
            const ggml_tensor * ub = u->view_src ? u->view_src : u;
            if (ub == base && u->data == t->data && ggml_nbytes(u) == ggml_nbytes(t)) {
                return;
            }
        }
        v.push_back(t);
    };
    std::string mm;
    for (int j = i0; j < i1; ++j) {
        const ggml_tensor * g = cgraph->nodes[j];
        for (int k = 0; k < GGML_MAX_SRC && g->src[k]; ++k) {
            if (!in_group(g->src[k])) {
                add_unique(ins, g->src[k]);
            }
        }
        if (g->op == GGML_OP_MUL_MAT || g->op == GGML_OP_MUL_MAT_ID || g->op == GGML_OP_FLASH_ATTN_EXT) {
            mm += std::string(" ") + ggml_op_name(g->op) + ":" + sh(g->src[0]) + ":" + sh(g->src[1]) +
                (g->op == GGML_OP_MUL_MAT_ID ? ":" + sh(g->src[2]) : std::string());
        }
        if (ggml_op_is_empty(g->op)) {
            continue;
        }
        bool used_later = j == i1 - 1 || (g->flags & GGML_TENSOR_FLAG_OUTPUT);
        for (int m = i1; m < cgraph->n_nodes && !used_later; ++m) {
            const ggml_tensor * c = cgraph->nodes[m];
            for (int k = 0; k < GGML_MAX_SRC && c->src[k] && !used_later; ++k) {
                used_later = c->src[k] == g || c->src[k]->view_src == g;
            }
        }
        if (used_later) {
            add_unique(outs, g);
        }
    }
    std::string r = " || in";
    for (const ggml_tensor * t : ins) {
        r += " " + sh(t);
    }
    r += " out";
    for (const ggml_tensor * t : outs) {
        r += " " + sh(t);
    }
    if (!mm.empty()) {
        r += " mm" + mm;
    }
    cache.emplace(first, r);
    return r;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // GGML_CUDA_OP_PROFILE_MAX_NT=K: profile only graphs whose matmuls see <= K tokens (e.g. decode after a deep prefill)
    bool op_prof = g_op_profile.enabled;
    if (op_prof) {
        static const int64_t max_nt = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE_MAX_NT"); return e ? (int64_t) atoll(e) : (int64_t) 0; }();
        if (max_nt > 0) {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                // tokens: MUL_MAT_ID src1 ne[2]; MUL_MAT against a weight: src1 ne[1]
                const bool big_id = n->op == GGML_OP_MUL_MAT_ID && n->src[1]->ne[2] > max_nt;
                const bool big_mm = n->op == GGML_OP_MUL_MAT && n->src[0]->buffer &&
                    ggml_backend_buffer_get_usage(n->src[0]->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    n->src[1]->ne[1]*n->src[1]->ne[2] > max_nt;
                if (big_id || big_mm) {
                    op_prof = false;
                    break;
                }
            }
        }
    }

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    const ggml_backend_cuda_context::shexp_conc * conc_cur = nullptr;
    bool        conc_side_open = false;
    int         conc_n = 0;
    cudaEvent_t conc_fork = nullptr, conc_done = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    // GGML_CUDA_DUMP_NODES=1: print the nodes of the first single-token MoE graph per device (op, name, shapes, sources)
    {
        static const int dump = [] { const char * e = getenv("GGML_CUDA_DUMP_NODES"); return e ? atoi(e) : 0; }();
        static bool dumped[GGML_CUDA_MAX_DEVICES] = {};
        // =2: every small graph (< 20 nodes) of device 0, once per distinct first node
        if (dump == 2 && cuda_ctx->device == 0 && cgraph->n_nodes < 20) {
            static std::set<std::string> seen;
            const std::string key = std::string(cgraph->nodes[0]->name) + "/" + std::to_string(cgraph->n_nodes);
            if (seen.insert(key).second) {
                fprintf(stderr, "SMALLGRAPH dev0 n=%d:", cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    fprintf(stderr, " %s(%s)", ggml_op_desc(cgraph->nodes[i]), cgraph->nodes[i]->name);
                }
                fprintf(stderr, "\n");
            }
        }
        // =3: the first decode attention graph of device 0 (one with a DSV4_SPARSE_ATTN node), full node list
        if (dump == 3 && cuda_ctx->device == 0) {
            static bool done = false;
            bool has_sa = false;
            for (int i = 0; i < cgraph->n_nodes && !done; ++i) {
                has_sa = has_sa || cgraph->nodes[i]->op == GGML_OP_DSV4_SPARSE_ATTN;
            }
            if (has_sa && !done) {
                done = true;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    const ggml_tensor * n = cgraph->nodes[i];
                    fprintf(stderr, "ANODE %4d %-14s %-28s [%lld,%lld,%lld,%lld] %s", i, ggml_op_desc(n), n->name,
                        (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3], ggml_type_name(n->type));
                    for (int j = 0; j < GGML_MAX_SRC && n->src[j]; ++j) {
                        fprintf(stderr, " | %s", n->src[j]->name);
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
        if (dump == 1 && !dumped[cuda_ctx->device]) {
            bool one_tok_moe = false;
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                if (n->op == GGML_OP_MUL_MAT_ID && n->src[1]->ne[2] == 1) {
                    one_tok_moe = true;
                    break;
                }
            }
            if (one_tok_moe) {
                dumped[cuda_ctx->device] = true;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    const ggml_tensor * n = cgraph->nodes[i];
                    fprintf(stderr, "NODE dev%d %4d %-14s %-28s [%lld,%lld,%lld,%lld] %s", cuda_ctx->device, i, ggml_op_desc(n), n->name,
                        (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3], ggml_type_name(n->type));
                    for (int j = 0; j < GGML_MAX_SRC && n->src[j]; ++j) {
                        fprintf(stderr, " | %s %s[%lld,%lld,%lld]", n->src[j]->name, ggml_type_name(n->src[j]->type),
                            (long long) n->src[j]->ne[0], (long long) n->src[j]->ne[1], (long long) n->src[j]->ne[2]);
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
    }

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            struct deferred_concat { const ggml_tensor * concat; const ggml_tensor * tail; const ggml_tensor * conv; };
            std::vector<deferred_concat> deferred_concats;
            cuda_ctx->f16_inplace.clear();
            cuda_ctx->gdn_qknorms.clear();
            cuda_ctx->hcp_inject_node = nullptr;
            cuda_ctx->hcp_shexp_node  = nullptr;
            cuda_ctx->hcp_done.clear();
            cuda_ctx->early_done.clear();
            cuda_ctx->conv_state_rows.clear();
            cuda_ctx->gdn_state_rows.clear();
            {
                // GGML_CUDA_DUMP_GRAPH=N: print device 0's first N graphs node by node (index, op, name, shape, sources)
                static int dump_left = [] { const char * e = getenv("GGML_CUDA_DUMP_GRAPH"); return e ? atoi(e) : 0; }();
                if (dump_left > 0 && cuda_ctx->device == 0) {
                    --dump_left;
                    for (int i = 0; i < cgraph->n_nodes; i++) {
                        const ggml_tensor * t = cgraph->nodes[i];
                        fprintf(stderr, "G %4d %-16s %-44s [%lld,%lld,%lld,%lld] use %d", i, ggml_op_desc(t), t->name,
                                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                                ggml_node_get_use_count(cgraph, i));
                        for (int k = 0; k < GGML_MAX_SRC && t->src[k]; ++k) {
                            fprintf(stderr, " | %s '%s'", ggml_op_desc(t->src[k]), t->src[k]->name);
                        }
                        fprintf(stderr, "\n");
                    }
                }
            }
            {
                static const bool q8_dbg = getenv("GGML_CUDA_Q8_CACHE_DBG") != nullptr;
                if (q8_dbg && (cuda_ctx->q8_cache_hits || cuda_ctx->q8_cache_misses)) {
                    fprintf(stderr, "q8_cache dev %d: %lld hits %lld misses\n", cuda_ctx->device,
                        (long long) cuda_ctx->q8_cache_hits, (long long) cuda_ctx->q8_cache_misses);
                }
                cuda_ctx->q8_cache_hits = cuda_ctx->q8_cache_misses = 0;
            }
            cuda_ctx->q8_cache.clear();
            if (op_prof && g_op_profile.head_spin) {
                static const double spin_us = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE_SPIN_US"); return e ? atof(e) : 20.0; }();
                const long long cycles = (long long) (spin_us * cgraph->n_nodes * 1.7e3); // ~1.7 GHz
                ggml_cuda_op_profile_spin<<<1, 1, 0, cuda_ctx->stream()>>>(cycles);
            }
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // CONCAT(conv_state, x^T) -> ... -> SSM_CONV -> SILU of the gated-DeltaNet short convolution, plus the
                // conv-state tail VIEW -> CONT: computed here, while x and the state are still live, straight into the
                // SILU and CONT outputs; the concat is never materialized. Bails out if those outputs overlap memory
                // that any node in between reads or writes (the allocator may place them in memory freed after here).
                static const bool conv_block_dec = [] { const char * e = getenv("GGML_GDN_CONV_BLOCK_DEC"); return !(e && atoi(e) == 0); }();
                if (node->op == GGML_OP_CONCAT && ggml_node_get_use_count(cgraph, i) == 2 && (!conv_block_dec || node->src[1]->ne[0] > 16)) {
                    int iv = -1, ic = -1, is = -1;
                    bool other = false;
                    for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 64) && !(ic >= 0 && is >= 0); ++j) {
                        const ggml_tensor * t = cgraph->nodes[j];
                        const bool uses = t->src[0] == node || t->src[1] == node || t->src[2] == node;
                        if (t->op == GGML_OP_VIEW && t->src[0] == node && iv < 0) {
                            iv = j;
                        } else if (t->op == GGML_OP_SSM_CONV && t->src[0] == node && is < 0) {
                            is = j;
                        } else if (iv >= 0 && t->op == GGML_OP_CONT && t->src[0] == cgraph->nodes[iv] && ic < 0) {
                            ic = j;
                        } else if (uses) {
                            other = true;
                            break;
                        }
                    }
                    bool ok = !other && iv >= 0 && ic >= 0 && is >= 0 && is + 1 < cgraph->n_nodes && ic < is &&
                            ggml_node_get_use_count(cgraph, iv) == 1 && ggml_node_get_use_count(cgraph, is) == 1 &&
                            cgraph->nodes[is + 1]->op == GGML_OP_UNARY &&
                            cgraph->nodes[iv]->ne[0] == 3 && cgraph->nodes[iv]->view_offs == node->src[1]->ne[0]*sizeof(float) &&
                            cgraph->nodes[iv]->nb[1] == node->nb[1] && ggml_is_contiguous(cgraph->nodes[ic]) &&
                            ggml_are_same_shape(cgraph->nodes[ic], node->src[0]) &&
                            ggml_cuda_ssm_conv_deferred_ok(node, cgraph->nodes[is], cgraph->nodes[is + 1]);
                    if (ok) {
                        ggml_tensor * silu = cgraph->nodes[is + 1];
                        ggml_tensor * tail = cgraph->nodes[ic];
                        const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
                            const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
                            return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
                        };
                        // silu may alias x (same layout, handled by the kernel) but not the state or the tail
                        ok = !overlaps(silu, node->src[0]) && !overlaps(silu, tail) && !overlaps(tail, node->src[1]) &&
                             (!overlaps(silu, node->src[1]) || (silu->data == node->src[1]->data &&
                              silu->nb[1] == node->src[1]->nb[0] && node->src[1]->nb[1] == sizeof(float)));
                        for (int j = i + 1; ok && j <= is + 1; ++j) {
                            const ggml_tensor * t = cgraph->nodes[j];
                            if (j == iv || j == ic || j == is || j == is + 1 || ggml_is_empty(t)) {
                                continue;
                            }
                            const ggml_tensor * ts[1 + GGML_MAX_SRC] = { t };
                            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                                ts[1 + k] = t->src[k];
                            }
                            for (const ggml_tensor * u : ts) {
                                if (u && u != node && u->data && (overlaps(u, silu) || (j < ic && overlaps(u, tail)))) {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if (ok) {
                            ggml_cuda_ssm_conv_deferred(*cuda_ctx, node, cgraph->nodes[is], silu, tail);
                            deferred_concats.push_back({node, tail, cgraph->nodes[is]});
                            continue;
                        }
                    }
                }
                // gated-DeltaNet q/k: RMS_NORM -> SCALE feeding only the GDN node, whose prefill kernel can normalize
                // q/k itself; the raw inputs are views of the conv output, which the GDN keeps live through v
                if (node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_SCALE &&
                        cgraph->nodes[i + 1]->src[0] == node) {
                    bool skip = false;
                    for (const auto & e : cuda_ctx->gdn_qknorms) {
                        if (e.q_out == cgraph->nodes[i + 1] || e.k_out == cgraph->nodes[i + 1]) {
                            skip = true; // registered with its partner
                            break;
                        }
                    }
                    if (!skip) {
                        // find the GDN that reads this SCALE and check its other q/k operand has the same form
                        for (int j = i + 2; j < std::min(cgraph->n_nodes, i + 40); ++j) {
                            ggml_tensor * g = cgraph->nodes[j];
                            if (g->op != GGML_OP_GATED_DELTA_NET || (g->src[0] != cgraph->nodes[i + 1] && g->src[1] != cgraph->nodes[i + 1])) {
                                continue;
                            }
                            const ggml_tensor * qs = g->src[0], * ks = g->src[1];
                            const auto norm_of = [](const ggml_tensor * sc) -> const ggml_tensor * {
                                return sc->op == GGML_OP_SCALE && sc->src[0]->op == GGML_OP_RMS_NORM &&
                                       ggml_get_op_params_f32(sc, 1) == 0.0f ? sc->src[0] : nullptr;
                            };
                            const ggml_tensor * qn = norm_of(qs), * kn = norm_of(ks);
                            if (!qn || !kn) {
                                break;
                            }
                            const ggml_tensor * qr = qn->src[0], * kr = kn->src[0];
                            const ggml_tensor * vb = ggml_cuda_view_base(g->src[2]);
                            // indices of the four nodes, for use counts (they are adjacent pairs before the GDN)
                            int iqn = -1, ikn = -1;
                            for (int k = i; k < j; ++k) {
                                if (cgraph->nodes[k] == qn) { iqn = k; }
                                if (cgraph->nodes[k] == kn) { ikn = k; }
                            }
                            if (iqn < 0 || ikn < 0 || cgraph->nodes[iqn + 1] != qs || cgraph->nodes[ikn + 1] != ks ||
                                    ggml_node_get_use_count(cgraph, iqn) != 1 || ggml_node_get_use_count(cgraph, iqn + 1) != 1 ||
                                    ggml_node_get_use_count(cgraph, ikn) != 1 || ggml_node_get_use_count(cgraph, ikn + 1) != 1 ||
                                    (qs->flags & GGML_TENSOR_FLAG_OUTPUT) || (ks->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                                    ggml_cuda_view_base(qr) != vb || ggml_cuda_view_base(kr) != vb ||
                                    !ggml_cuda_gdn_colthread_qknorm_ok(g, qr, kr)) {
                                break;
                            }
                            cuda_ctx->gdn_qknorms.push_back({qs, ks, qr, kr, ggml_get_op_params_f32(qn, 0), ggml_get_op_params_f32(kn, 0),
                                                             ggml_get_op_params_f32(qs, 0), ggml_get_op_params_f32(ks, 0)});
                            skip = true;
                            break;
                        }
                    }
                    if (skip) {
                        i += 1; // RMS_NORM and SCALE are applied inside the GDN kernel
                        continue;
                    }
                }
                if (!deferred_concats.empty() && (node->op == GGML_OP_CONT || node->op == GGML_OP_SSM_CONV)) {
                    bool handled = false;
                    for (const auto & d : deferred_concats) {
                        if (node == d.tail) {
                            handled = true; // written at the CONCAT node
                            break;
                        }
                        if (node == d.conv) {
                            i += 1;         // SSM_CONV and SILU were computed at the CONCAT node
                            handled = true;
                            break;
                        }
                    }
                    if (handled) {
                        continue;
                    }
                }

                cudaEvent_t op_prof_ev0 = nullptr;
                if (op_prof && g_op_profile.async) {
                    op_prof_ev0 = g_op_profile.take(cuda_ctx->device);
                    CUDA_CHECK(cudaEventRecord(op_prof_ev0, cuda_ctx->stream()));
                } else if (op_prof) {
                    if (!g_op_profile.ev0[cuda_ctx->device]) {
                        CUDA_CHECK(cudaEventCreate(&g_op_profile.ev0[cuda_ctx->device]));
                        CUDA_CHECK(cudaEventCreate(&g_op_profile.ev1[cuda_ctx->device]));
                    }
                    static const bool spin = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE"); return e && atoi(e) == 5; }();
                    if (spin && g_op_profile.cal_us[cuda_ctx->device] == 0.0) {
                        // the fixed cost of one measurement (spin, event, launch, event) with an empty kernel
                        std::vector<float> t;
                        for (int it = 0; it < 32; ++it) {
                            ggml_cuda_op_profile_spin<<<1, 1, 0, cuda_ctx->stream()>>>(400000);
                            CUDA_CHECK(cudaEventRecord(g_op_profile.ev0[cuda_ctx->device], cuda_ctx->stream()));
                            ggml_cuda_op_profile_empty<<<1, 64, 0, cuda_ctx->stream()>>>();
                            CUDA_CHECK(cudaEventRecord(g_op_profile.ev1[cuda_ctx->device], cuda_ctx->stream()));
                            CUDA_CHECK(cudaEventSynchronize(g_op_profile.ev1[cuda_ctx->device]));
                            float ms = 0;
                            CUDA_CHECK(cudaEventElapsedTime(&ms, g_op_profile.ev0[cuda_ctx->device], g_op_profile.ev1[cuda_ctx->device]));
                            t.push_back(ms);
                        }
                        std::sort(t.begin(), t.end());
                        g_op_profile.cal_us[cuda_ctx->device] = 1e3*t[t.size()/2];
                    }
                    if (spin) {
                        ggml_cuda_op_profile_spin<<<1, 1, 0, cuda_ctx->stream()>>>(400000);
                    }
                    CUDA_CHECK(cudaEventRecord(g_op_profile.ev0[cuda_ctx->device], cuda_ctx->stream()));
                }
                const int op_prof_i = i;
                const auto op_profile_end = [&](const ggml_tensor * first, int n_fused) {
                    if (!op_prof) {
                        return;
                    }
                    static const bool prof_src = getenv("GGML_CUDA_OP_PROFILE_SRC") != nullptr;
                    const std::string io = prof_src && n_fused > 1 ? ggml_cuda_op_profile_group_io(cgraph, op_prof_i, n_fused) : std::string();
                    // GGML_CUDA_OP_PROFILE_DUMP=N: also print the first N nodes as they run (with GGML_CUDA_OP_PROFILE_SRC=1: shapes)
                    static int dump_left = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE_DUMP"); return e ? atoi(e) : 0; }();
                    // GGML_CUDA_OP_PROFILE_DUMP_SKIP=M: start the dump after M profiled nodes (past warm-up / first-token graphs)
                    static int dump_skip = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE_DUMP_SKIP"); return e ? atoi(e) : 0; }();
                    if (dump_left > 0 && dump_skip > 0) {
                        dump_skip--;
                    } else if (dump_left > 0) {
                        dump_left--;
                        fprintf(stderr, "NODE d%d %s\n", cuda_ctx->device, ggml_cuda_op_profile::key(first, n_fused).c_str());
                    }
                    if (g_op_profile.async) {
                        cudaEvent_t ev1 = g_op_profile.take(cuda_ctx->device);
                        CUDA_CHECK(cudaEventRecord(ev1, cuda_ctx->stream()));
                        static const bool per_dev4 = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE"); return e && atoi(e) == 4; }();
                        g_op_profile.pending[cuda_ctx->device].push_back({op_prof_ev0, ev1,
                            (per_dev4 ? "d" + std::to_string(cuda_ctx->device) + " " : std::string()) + ggml_cuda_op_profile::key(first, n_fused) + io,
                            std::string(first->name)});
                        return;
                    }
                    CUDA_CHECK(cudaEventRecord(g_op_profile.ev1[cuda_ctx->device], cuda_ctx->stream()));
                    CUDA_CHECK(cudaEventSynchronize(g_op_profile.ev1[cuda_ctx->device]));
                    float ms = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, g_op_profile.ev0[cuda_ctx->device], g_op_profile.ev1[cuda_ctx->device]));
                    g_op_profile.dev_ms[cuda_ctx->device] += ms;
                    static const bool per_dev = [] { const char * e = getenv("GGML_CUDA_OP_PROFILE"); return e && atoi(e) == 2; }();
                    auto & e = g_op_profile.totals[(per_dev ? "d" + std::to_string(cuda_ctx->device) + " " : std::string()) +
                                                   ggml_cuda_op_profile::key(first, n_fused) + io];
                    e.ms += ms;
                    e.n++;
                };

                // GGML_CUDA_SKIP_OPS="key1;key2": do not run nodes whose profile key contains one of the keys (results are
                // garbage): the throughput change is the real in-graph cost of those ops
                static const std::vector<std::string> skip_ops = [] {
                    std::vector<std::string> v;
                    const char * e = getenv("GGML_CUDA_SKIP_OPS");
                    std::string cur;
                    for (const char * c = e ? e : ""; ; ++c) {
                        if (*c == ';' || *c == 0) { if (!cur.empty()) { v.push_back(cur); } cur.clear(); if (!*c) { break; } } else { cur += *c; }
                    }
                    return v;
                }();
                // a skipped node that starts a fused group skips the whole group: its length is recorded the first
                // time the node runs (graphs are reused, so only the first evaluation of each graph runs it)
                static std::unordered_map<const ggml_tensor *, int> skip_fused_len;
                bool skip_this = false;
                if (!skip_ops.empty()) {
                    const std::string k = ggml_cuda_op_profile::key(node, 1);
                    for (const auto & so : skip_ops) {
                        skip_this = skip_this || k.find(so) != std::string::npos;
                    }
                    if (skip_this) {
                        const auto it = skip_fused_len.find(node);
                        if (it != skip_fused_len.end()) {
                            // index producers: write valid indices (0, 1, ... per row) so the consumers do not fault
                            if ((node->op == GGML_OP_TOP_K || node->op == GGML_OP_ARGSORT) && node->type == GGML_TYPE_I32 &&
                                    ggml_is_contiguous(node)) {
                                const int64_t ne = ggml_nelements(node);
                                ggml_cuda_skip_iota<<<(unsigned) ((ne + 255)/256), 256, 0, cuda_ctx->stream()>>>(
                                    (int32_t *) node->data, ne, node->ne[0]);
                            }
                            i += it->second - 1;
                            continue;
                        }
                    }
                }

                static const bool fix_moe_ids = [] { const char * e = getenv("GGML_CUDA_FIX_MOE_IDS"); return e && atoi(e) != 0; }();
                if (fix_moe_ids && node->op == GGML_OP_MUL_MAT_ID && node->src[2] && node->src[2]->type == GGML_TYPE_I32 &&
                        node->src[2]->nb[0] == sizeof(int32_t) && node->src[2]->ne[0] <= 1024) {
                    const ggml_tensor * ids = node->src[2];
                    ggml_cuda_fix_moe_ids<<<(unsigned) ids->ne[1], (unsigned) ids->ne[0], 0, cuda_ctx->stream()>>>(
                        (int32_t *) ids->data, (int) ids->ne[0], (int) ids->ne[1], ids->nb[1]/sizeof(int32_t), (int) node->src[0]->ne[2]);
                }

                if (node->op == GGML_OP_GET_ROWS && (ggml_cuda_defer_gdn_state_rows(cuda_ctx, cgraph, i) ||
                                                     ggml_cuda_defer_conv_state_rows(cuda_ctx, cgraph, i))) {
                    continue;
                }

                // shared-expert branch on stream 1 (see graph_optimize)
                if (!cuda_ctx->shexp_conc_map.empty()) {
                    const auto cit = cuda_ctx->shexp_conc_map.find(node);
                    if (cit != cuda_ctx->shexp_conc_map.end() && !conc_cur) {
                        if ((int) cuda_ctx->conc_events.size() < 2*(conc_n + 1)) {
                            for (int e = 0; e < 2; ++e) {
                                cudaEvent_t ev;
                                CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                                cuda_ctx->conc_events.push_back(ev);
                            }
                        }
                        conc_fork = cuda_ctx->conc_events[2*conc_n];
                        conc_done = cuda_ctx->conc_events[2*conc_n + 1];
                        conc_n++;
                        CUDA_CHECK(cudaEventRecord(conc_fork, cuda_ctx->stream(cuda_ctx->device, 0)));
                        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(cuda_ctx->device, 1), conc_fork));
                        cuda_ctx->curr_stream_no = 1;
                        conc_cur = &cit->second;
                        conc_side_open = true;
                    } else if (conc_side_open && !conc_cur->side.count(node)) {
                        CUDA_CHECK(cudaEventRecord(conc_done, cuda_ctx->stream(cuda_ctx->device, 1)));
                        cuda_ctx->curr_stream_no = 0;
                        conc_side_open = false;
                    }
                    if (conc_cur && !conc_side_open && node == conc_cur->join) {
                        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(cuda_ctx->device, 0), conc_done));
                        conc_cur = nullptr;
                    }
                }

                if (!cuda_ctx->hcp_done.empty()) {
                    const int done_len = ggml_cuda_hc_persist_done(*cuda_ctx, node);
                    if (done_len > 0) {
                        i += done_len - 1; // computed early by a persistent kernel
                        continue;
                    }
                }
                if (!cuda_ctx->early_done.empty()) {
                    const auto it = std::find(cuda_ctx->early_done.begin(), cuda_ctx->early_done.end(), node);
                    if (it != cuda_ctx->early_done.end()) {
                        cuda_ctx->early_done.erase(it); // computed early with a sibling (multi-matrix gemv1)
                        continue;
                    }
                }
                const int64_t tl_f0 = g_timeline.enabled ? ggml_cuda_timeline::now() : 0;
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                if (nodes_to_skip == GGML_CUDA_FUSED_SELF) {
                    continue; // the node was computed by a fused launch that consumed only itself (+ early siblings)
                }
                if (skip_this) {
                    skip_fused_len[node] = nodes_to_skip + 1;
                }
                if (g_timeline.enabled && ggml_cuda_timeline::now() - tl_f0 > 5000) {
                    fprintf(stderr, "TLSLOW dev %d fused node %d %s (%s) host %.2f ms\n", cuda_ctx->device, i, ggml_op_name(node->op), node->name, (ggml_cuda_timeline::now() - tl_f0)/1e3);
                }

                if (nodes_to_skip != 0) {
                    op_profile_end(node, nodes_to_skip + 1);
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                const int64_t tl_h0 = g_timeline.enabled ? ggml_cuda_timeline::now() : 0;
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (g_timeline.enabled) {
                    const int64_t dt = ggml_cuda_timeline::now() - tl_h0;
                    if (dt > 5000) {
                        fprintf(stderr, "TLSLOW dev %d node %d %s (%s) host %.2f ms\n", cuda_ctx->device, i, ggml_op_name(node->op), node->name, dt/1e3);
                    }
                }
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                op_profile_end(node, 1);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

        if (op_prof && g_op_profile.async) {
            g_op_profile.flush(cuda_ctx->device);
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH


static enum ggml_status ggml_backend_cuda_graph_compute_impl(ggml_backend_t backend, ggml_cgraph * cgraph);

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    if (!g_timeline.enabled) {
        return ggml_backend_cuda_graph_compute_impl(backend, cgraph);
    }
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_timeline::entry * e;
    {
        std::lock_guard<std::mutex> lock(g_timeline.mtx);
        e = &g_timeline.entries.emplace_back();
    }
    e->dev = cuda_ctx->device;
    e->n_nodes = cgraph->n_nodes;
    e->h0 = ggml_cuda_timeline::now();
    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), ggml_cuda_timeline::cb_start, e));
    const enum ggml_status st = ggml_backend_cuda_graph_compute_impl(backend, cgraph);
    CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), ggml_cuda_timeline::cb_end, e));
    e->h1 = ggml_cuda_timeline::now();
    return st;
}

// GGML_CUDA_GRAPH_STATS=1 (debug): per device, how often a cgraph ran as a captured graph replay, was (re)captured, ran
// directly (warm-up, incompatible or disabled) or reset its warm-up, plus the first few changed-property node names
struct ggml_cuda_graph_stats {
    bool on = getenv("GGML_CUDA_GRAPH_STATS") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_STATS")) != 0;
    std::atomic<int64_t> replay[GGML_CUDA_MAX_DEVICES] = {}, capture[GGML_CUDA_MAX_DEVICES] = {},
        direct[GGML_CUDA_MAX_DEVICES] = {}, reset[GGML_CUDA_MAX_DEVICES] = {}, incompat[GGML_CUDA_MAX_DEVICES] = {};
    ~ggml_cuda_graph_stats() {
        if (!on) {
            return;
        }
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
            if (replay[d] + capture[d] + direct[d] > 0) {
                fprintf(stderr, "graph_stats dev %d: replay %lld capture %lld direct %lld (incompatible %lld) resets %lld\n", d,
                    (long long) replay[d], (long long) capture[d], (long long) direct[d], (long long) incompat[d],
                    (long long) reset[d]);
            }
        }
    }
};
static ggml_cuda_graph_stats g_graph_stats;

static enum ggml_status ggml_backend_cuda_graph_compute_impl(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    ggml_cuda_hcp_ar_graph_begin(*cuda_ctx, cgraph); // before any capture: a fallback exchange runs live

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (!graph_compatible && g_graph_stats.on) {
            g_graph_stats.incompat[cuda_ctx->device]++;
        }
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    if (g_graph_stats.on) {
                        g_graph_stats.reset[cuda_ctx->device]++;
                    }
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (g_graph_stats.on) {
        (use_cuda_graph ? (cuda_graph_update_required ? g_graph_stats.capture : g_graph_stats.replay) : g_graph_stats.direct)[cuda_ctx->device]++;
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (!disable_fusion) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL) {
                continue;
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                continue;
            }

            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
            if (match.expert_scale != nullptr) {
                params->add_alloc_dep(
                    params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
            }
            i += match.node_count - 1;
        }
    }

    // decode: the shared-expert chain (gate/up -> GLU -> down, "ffn_shexp-<il>") does not depend on the routed experts, so
    // it can run on a second stream while the router and the experts run on the main one (a decode kernel leaves most of
    // the GPU idle). Its nodes move right after the FFN input and its outputs stay allocated until ffn_out, so the two
    // branches never share memory; graph_compute forks at the first chain node and joins before shared_expert_gate.
    // One token only: there the gate/up pair always fuses (its intermediates are never written). Opt-in
    // (GGML_CUDA_SHEXP_CONC=1): bit-identical but tg 49.4 -> 44 on gfx906: the stream fork/join inside the captured HIP
    // graph costs far more than the overlap gains (cross-queue waits).
    {
        static const bool conc_env = [] { const char * e = getenv("GGML_CUDA_SHEXP_CONC"); return e && atoi(e) != 0; }();
        cuda_ctx->shexp_conc_map.clear();
        if (conc_env && GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
            std::unordered_map<const ggml_tensor *, int> idx;
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                idx[cgraph->nodes[i]] = i;
            }
            const auto find_named = [&](int from, int to, const std::string & nm) -> int {
                for (int j = from; j < std::min(to, cgraph->n_nodes); ++j) {
                    if (nm == cgraph->nodes[j]->name) {
                        return j;
                    }
                }
                return -1;
            };
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                ggml_tensor * lg = cgraph->nodes[i];
                if (lg->op != GGML_OP_MUL_MAT || strncmp(lg->name, "ffn_moe_logits-", 15) != 0 || lg->src[1]->ne[1] != 1) {
                    continue;
                }
                const std::string il = lg->name + 15;
                const ggml_tensor * root = lg->src[1];
                const auto ir = idx.find(root);
                const int i_sh = find_named(i, i + 256, "ffn_shexp-" + il);
                const int i_g  = find_named(i, i + 256, "shared_expert_gate-" + il);
                const int i_o  = find_named(i, i + 256, "ffn_out-" + il);
                if (ir == idx.end() || i_sh < 0 || i_g < 0 || i_o < 0 || !(i_sh < i_g && i_g < i_o)) {
                    continue;
                }
                const int r = ir->second;
                // the chain: ancestors of ffn_shexp inside (r, i_sh]
                std::unordered_set<const ggml_tensor *> side;
                std::vector<const ggml_tensor *> stack = { cgraph->nodes[i_sh] };
                bool ok = true;
                while (!stack.empty() && ok) {
                    const ggml_tensor * t = stack.back();
                    stack.pop_back();
                    if (side.count(t)) {
                        continue;
                    }
                    side.insert(t);
                    for (int k = 0; k < GGML_MAX_SRC; ++k) {
                        const ggml_tensor * sn = t->src[k];
                        if (!sn || sn == root) {
                            continue;
                        }
                        const auto it = idx.find(sn);
                        if (it == idx.end() || it->second <= r) {
                            continue; // weights, inputs, earlier results
                        }
                        if (it->second > i_sh) {
                            ok = false;
                            break;
                        }
                        stack.push_back(sn);
                    }
                }
                // nothing else before the join may read the chain, and the chain may not read routed-expert nodes
                for (int j = r + 1; ok && j < i_g; ++j) {
                    const ggml_tensor * t = cgraph->nodes[j];
                    if (side.count(t)) {
                        continue;
                    }
                    for (int k = 0; k < GGML_MAX_SRC; ++k) {
                        if (t->src[k] && side.count(t->src[k])) {
                            ok = false;
                        }
                    }
                }
                if (!ok || side.size() < 2) {
                    continue;
                }
                // reorder (r, i_sh]: chain first, then the rest, each in its original order
                std::vector<ggml_tensor *> first, rest;
                for (int j = r + 1; j <= i_sh; ++j) {
                    (side.count(cgraph->nodes[j]) ? first : rest).push_back(cgraph->nodes[j]);
                }
                int w = r + 1;
                for (ggml_tensor * t : first) { cgraph->nodes[w++] = t; }
                for (ggml_tensor * t : rest)  { cgraph->nodes[w++] = t; }
                for (int j = r + 1; j <= i_sh; ++j) {
                    idx[cgraph->nodes[j]] = j;
                }
                // keep the chain's results until ffn_out: the GLU output (read by the down projection on the side stream)
                // and ffn_shexp (read at the join)
                for (const ggml_tensor * t : side) {
                    if (t->op == GGML_OP_GLU || t == cgraph->nodes[idx[first.back()]]) {
                        params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(t), cgraph->nodes[i_o]);
                    }
                }
                ggml_backend_cuda_context::shexp_conc c;
                c.side = std::move(side);
                c.join = cgraph->nodes[i_g];
                cuda_ctx->shexp_conc_map.emplace(first.front(), std::move(c));
                i = i_o;
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
            // only the shape of the kernel (src0) is used, so a strided dummy kernel view is fine (pixtral, deepseek4v)
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 &&
                   (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_MIX:
            return ggml_cuda_dsv4_hc_mix_supported(op);
        case GGML_OP_DSV4_SPARSE_ATTN:
            return ggml_cuda_dsv4_sparse_attn_supported(op);
        case GGML_OP_DSV4_COMP_POOL:
            return ggml_cuda_dsv4_comp_pool_supported(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_GET_ROWS_MEAN:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->ne[1] <= INT_MAX && op->ne[2] <= 65535 && op->ne[3] <= 65535;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// true if this backend's cross-device copies (and host-staged input uploads) are ordered on the destination
// device's stream (the staged path), which lets the scheduler re-plan buffers without a full stop
static bool ggml_backend_cuda_copies_stream_ordered(ggml_backend_t backend) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    return ggml_cuda_use_staged_copy(ctx->device);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_copies_stream_ordered") == 0) {
        return (void *)ggml_backend_cuda_copies_stream_ordered;
    }
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
