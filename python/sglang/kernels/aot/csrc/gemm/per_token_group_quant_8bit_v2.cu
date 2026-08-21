#ifdef TORCH_TARGET_VERSION
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Half.h>

#include <cstdlib>
#include <mutex>
#include <optional>
#include <type_traits>

#include "gemm/gemm_ops.h"
#include "sgl_kernel_cuda_device.h"
#include "sgl_kernel_cuda_stream.h"

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

#define SGL_CHECK(...) STD_TORCH_CHECK(__VA_ARGS__)
#define SGL_CHECK_NO_MSG(condition)                                                                   \
  STD_TORCH_CHECK(                                                                                    \
      condition,                                                                                      \
      "Expected " #condition                                                                          \
      " to be true, but got false.  (Could this error message be improved?  If so, please report an " \
      "enhancement request to PyTorch.)")
#define CHECK_CUDA(x) STD_TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_EQ(a, b) STD_TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)
#define SGL_CURRENT_CUDA_STREAM() sgl_kernel::stable::get_current_cuda_stream()
#define SGL_GET_ENV_ENABLE_PDL() stable_get_env_enable_pdl()
#define SGL_INPUT_PTR(tensor, type) stable_input_ptr<type>(tensor)
#define SGL_OUTPUT_Q_PTR(tensor, type) stable_output_q_ptr<type>(tensor)
#define SGL_OUTPUT_S_PTR(tensor, type) stable_output_s_ptr<type>(tensor)
#define SGL_MASKED_M_PTR(optional_tensor) stable_masked_m_ptr(optional_tensor)
#define SGL_OPTIONAL_TENSOR_ARG const std::optional<Tensor>& masked_m
#ifdef FLASHINFER_ENABLE_BF16
#define SGL_DISPATCH_CASE_BF16(c_type, ...) \
  case ScalarType::BFloat16: {              \
    using c_type = nv_bfloat16;             \
    return __VA_ARGS__();                   \
  }
#else
#define SGL_DISPATCH_CASE_BF16(c_type, ...)
#endif
#define SGL_DISPATCH_FLOAT_FP16(TYPE, NAME, ...)                                           \
  [&]() -> bool {                                                                          \
    const auto scalar_type = TYPE;                                                         \
    switch (scalar_type) {                                                                 \
      case ScalarType::Half: {                                                             \
        using scalar_t = nv_half;                                                          \
        return __VA_ARGS__();                                                              \
      }                                                                                    \
        SGL_DISPATCH_CASE_BF16(scalar_t, __VA_ARGS__)                                      \
      default:                                                                             \
        STD_TORCH_CHECK(                                                                   \
            false,                                                                         \
            NAME                                                                           \
            "(at::Tensor, at::Tensor, at::Tensor, int64_t, double, double, double, bool, " \
            "bool, const std::optional<at::Tensor>&)::<lambda()> failed to dispatch data " \
            "type ",                                                                       \
            torch::headeronly::toString(scalar_type));                                     \
        return false;                                                                      \
    }                                                                                      \
  }()

inline bool stable_get_env_enable_pdl() {
  static std::once_flag flag;
  static bool enable_pdl = false;
  std::call_once(flag, []() {
    const auto& prop = sgl_kernel::stable::get_cached_device_properties();
    if (prop.major * 10 + prop.minor >= 90) {
      const char* env = std::getenv("TRTLLM_ENABLE_PDL");
      enable_pdl = env != nullptr && env[0] == '1' && env[1] == '\0';
    }
  });
  return enable_pdl;
}

template <typename T>
const T* stable_input_ptr(const Tensor& tensor) {
  if constexpr (std::is_same_v<T, nv_half>) {
    return reinterpret_cast<const T*>(tensor.const_data_ptr<torch::headeronly::Half>());
  } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return reinterpret_cast<const T*>(tensor.const_data_ptr<torch::headeronly::BFloat16>());
  } else {
    return tensor.const_data_ptr<T>();
  }
}

template <typename T>
T* stable_output_q_ptr(Tensor& tensor) {
  if constexpr (std::is_same_v<T, int8_t>) {
    return tensor.mutable_data_ptr<int8_t>();
  } else {
    static_assert(std::is_same_v<T, torch::headeronly::Float8_e4m3fn>);
    return tensor.mutable_data_ptr<torch::headeronly::Float8_e4m3fn>();
  }
}

template <typename T>
T* stable_output_s_ptr(Tensor& tensor) {
  if constexpr (std::is_same_v<T, uint32_t>) {
    return reinterpret_cast<T*>(tensor.mutable_data_ptr<int32_t>());
  } else {
    static_assert(std::is_same_v<T, float>);
    return tensor.mutable_data_ptr<float>();
  }
}

const int32_t* stable_masked_m_ptr(const std::optional<Tensor>& masked_m) {
  return masked_m.has_value() ? masked_m->const_data_ptr<int32_t>() : nullptr;
}
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include "utils.h"

using Tensor = torch::Tensor;
using ScalarType = at::ScalarType;

#define SGL_CHECK(...) TORCH_CHECK(__VA_ARGS__)
#define SGL_CHECK_NO_MSG(condition) TORCH_CHECK(condition)
#define SGL_CURRENT_CUDA_STREAM() at::cuda::getCurrentCUDAStream()
#define SGL_GET_ENV_ENABLE_PDL() getEnvEnablePDL()
#define SGL_INPUT_PTR(tensor, type) static_cast<type*>(tensor.data_ptr())
#define SGL_OUTPUT_Q_PTR(tensor, type) static_cast<type*>(tensor.data_ptr())
#define SGL_OUTPUT_S_PTR(tensor, type) static_cast<type*>(tensor.data_ptr())
#define SGL_MASKED_M_PTR(optional_tensor) \
  static_cast<int32_t*>(optional_tensor.has_value() ? optional_tensor->data_ptr() : 0)
#define SGL_OPTIONAL_TENSOR_ARG const std::optional<Tensor>& masked_m
#define SGL_DISPATCH_FLOAT_FP16(TYPE, NAME, ...) DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(TYPE, scalar_t, __VA_ARGS__)
#endif

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

template <int THREADS_PER_SUBWARP>
__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffffffff;

  static_assert(
      (THREADS_PER_SUBWARP & (THREADS_PER_SUBWARP - 1)) == 0 && THREADS_PER_SUBWARP <= 16 && THREADS_PER_SUBWARP >= 1,
      "THREADS_PER_SUBWARP must be 1, 2, 4, 8, or 16");

  if constexpr (THREADS_PER_SUBWARP >= 16) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  }
  if constexpr (THREADS_PER_SUBWARP >= 8) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  }
  if constexpr (THREADS_PER_SUBWARP >= 4) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  }
  if constexpr (THREADS_PER_SUBWARP >= 2) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  }
  return val;
}

__device__ __forceinline__ float silu(const float& val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  float half = 0.5f * val;
  float t = __tanhf(half);
  return half * (1.0f + t);
#else
  return val / (1.0f + __expf(-val));
#endif
}

__device__ float2 fmul2_rn(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(a, b);
#else
  float2 result;
  result.x = a.x * b.x;
  result.y = a.y * b.y;
  return result;
#endif
}

// Copied and modified from DeepEP
__forceinline__ __device__ float fast_pow2(int x) {
  // We can ensure `-126 <= x and x <= 127`
  uint32_t bits_x = (x + 127) << 23;
  return *reinterpret_cast<float*>(&bits_x);
}

// Copied and modified from DeepEP
__forceinline__ __device__ int fast_log2_ceil(float x) {
  auto bits_x = *reinterpret_cast<uint32_t*>(&x);
  auto exp_x = (bits_x >> 23) & 0xff;
  auto man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

// Copied and modified from DeepEP
template <bool ROUND_SCALE, typename dtype_info>
__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv) {
  constexpr float MAX_8BIT_INV = 1.0f / dtype_info::MAX;
  if constexpr (ROUND_SCALE) {
    auto exp_scale_inv = fast_log2_ceil(amax * MAX_8BIT_INV);
    scale = fast_pow2(-exp_scale_inv);
    scale_inv = fast_pow2(exp_scale_inv);
  } else {
    scale_inv = amax * MAX_8BIT_INV;
    scale = dtype_info::MAX / amax;
  }
}

// Copied and modified from DeepEP
template <bool SCALE_UE8M0, typename OUT_DTYPE_T = std::conditional_t<SCALE_UE8M0, uint8_t, float>>
__forceinline__ __device__ OUT_DTYPE_T extract_required_scale_format(float value) {
  if constexpr (SCALE_UE8M0) {
    return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
  } else {
    return value;
  }
}

__device__ __forceinline__ void st_global(const int4* ptr, const int4& value) {
#ifndef USE_MUSA
  asm volatile(
      "st.global.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
#else
  int4* p = const_cast<int4*>(ptr);
  *p = value;
#endif
}

__device__ __forceinline__ int4 ld_global_nc(const int4* ptr) {
#ifndef USE_MUSA
  int4 ret;
  asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
#else
  return *ptr;
#endif
}

template <typename T>
struct DtypeInfo;

template <>
struct DtypeInfo<int8_t> {
  static constexpr float MIN = -128;
  static constexpr float MAX = 127;
};

template <>
struct DtypeInfo<torch::headeronly::Float8_e4m3fn> {
  static constexpr float MIN = -448;
  static constexpr float MAX = 448;
};

template <bool FUSE_SILU_AND_MUL>
__device__ __forceinline__ int compute_input_group_start_offset(
    int expert_idx,
    int token_idx,
    int hidden_dim_group_idx,
    int hidden_size,
    int num_tokens_per_expert,
    int group_size) {
  return expert_idx * num_tokens_per_expert * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) +
         token_idx * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) + hidden_dim_group_idx * group_size;
}

constexpr float LOCAL_ABSMAX_ABS = 1e-10;
constexpr uint32_t INPUT_PRIMARY_VEC_NUM_BYTES = 32;

struct NaiveScheduler {
  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = ([=]() -> int {
      if (num_groups % 16 == 0) {
        return 16;
      } else if (num_groups % 8 == 0) {
        return 8;
      } else if (num_groups % 4 == 0) {
        return 4;
      } else if (num_groups % 2 == 0) {
        return 2;
      }
      return 1;
    })();
    grid = dim3(num_groups / subwarps_per_block);
    block = dim3(subwarps_per_block * threads_per_subwarp);
  }

  template <bool FUSE_SILU_AND_MUL, int GROUP_SIZE, int THREADS_PER_SUBWARP, typename FUNC>
  __device__ __forceinline__ static void execute(
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int32_t* masked_m,
      const int num_tokens_per_expert,
      FUNC fn) {
    constexpr int expert_idx = 0;

    const int64_t subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;

    const int64_t block_group_id = blockIdx.x * subwarps_per_block;
    const int64_t group_id = block_group_id + subwarp_id;

    int64_t input_group_start_offset;
    if constexpr (!FUSE_SILU_AND_MUL) {
      input_group_start_offset = group_id * GROUP_SIZE;
    }

    const int token_idx = group_id / hidden_dim_num_groups;
    // At the hidden_size dimension, we are handling idx-th group
    const int hidden_dim_group_idx = group_id % hidden_dim_num_groups;

    if constexpr (FUSE_SILU_AND_MUL) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
      input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
    }

    fn(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset);
  }
};

struct MaskedLayoutScheduler {
  // TODO can be dynamically determined (which may be good when num rank is small)
  static constexpr int TOKEN_DIM_BLOCK_NUM_PER_EXPERT = 1024;
  static constexpr int SUBWARPS_PER_BLOCK = 16;

  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = SUBWARPS_PER_BLOCK;
    SGL_CHECK_NO_MSG(hidden_dim_num_groups % subwarps_per_block == 0);
    grid = dim3(hidden_dim_num_groups / subwarps_per_block, TOKEN_DIM_BLOCK_NUM_PER_EXPERT, num_local_experts);
    block = dim3(subwarps_per_block * threads_per_subwarp);
  }

  template <bool FUSE_SILU_AND_MUL, int GROUP_SIZE, int THREADS_PER_SUBWARP, typename FUNC>
  __device__ __forceinline__ static void execute(
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int32_t* masked_m,
      const int num_tokens_per_expert,
      FUNC fn) {
    const int64_t subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;

    const int expert_idx = blockIdx.z;
    const int token_idx_start = blockIdx.y;

    const int64_t hidden_dim_group_idx = blockIdx.x * SUBWARPS_PER_BLOCK + subwarp_id;

    const int curr_expert_token_num = masked_m[expert_idx];

    for (int token_idx = token_idx_start; token_idx < curr_expert_token_num;
         token_idx += TOKEN_DIM_BLOCK_NUM_PER_EXPERT) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
      const int64_t input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
      fn(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset);
    }
  }
};

template <
    typename SCHEDULER,
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    bool FUSE_SILU_AND_MUL = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s,
    const int32_t* __restrict__ masked_m,
    const int subwarps_per_block,
    const int hidden_dim_num_groups,
    // TODO can this be removed?
    const int scale_expert_stride,
    const int scale_hidden_stride,
    const int num_tokens_per_expert) {
  using dst_dtype_info = DtypeInfo<DST_DTYPE>;
  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif

  SCHEDULER::execute<FUSE_SILU_AND_MUL, GROUP_SIZE, THREADS_PER_SUBWARP>(
      subwarps_per_block,
      hidden_dim_num_groups,
      masked_m,
      num_tokens_per_expert,
      [&](const int expert_idx,
          const int token_idx,
          const int hidden_dim_group_idx,
          const int lane_id,
          const int input_group_start_offset) {
        constexpr uint32_t INPUT_PRIMARY_VEC_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(T);
        constexpr uint32_t INPUT_PRIMARY_INT4_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(int4);

        const int offset_num_groups = expert_idx * num_tokens_per_expert * hidden_dim_num_groups +
                                      token_idx * hidden_dim_num_groups + hidden_dim_group_idx;

        int4 input_primary_int4[INPUT_PRIMARY_INT4_SIZE];
        T* input_primary_vec = reinterpret_cast<T*>(input_primary_int4);
        static_assert(sizeof(input_primary_vec[0]) * INPUT_PRIMARY_VEC_SIZE == sizeof(input_primary_int4));

        int4 input_secondary_int4[INPUT_PRIMARY_INT4_SIZE];
        T* input_secondary_vec = reinterpret_cast<T*>(input_secondary_int4);
        static_assert(sizeof(input_secondary_vec[0]) * INPUT_PRIMARY_VEC_SIZE == sizeof(input_secondary_int4));

#pragma unroll
        for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
          input_primary_int4[j] = ld_global_nc(
              reinterpret_cast<const int4*>(input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE) + j);
        }
        if constexpr (FUSE_SILU_AND_MUL) {
          const int secondary_offset = hidden_dim_num_groups * GROUP_SIZE;
#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
            input_secondary_int4[j] = ld_global_nc(
                reinterpret_cast<const int4*>(
                    input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE + secondary_offset) +
                j);
          }
        }

        constexpr int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
        scale_element_t* scale_output;
        if constexpr (IS_COLUMN_MAJOR) {
          constexpr int scale_token_stride = 1;

          const int hidden_idx_packed = hidden_dim_group_idx / num_elems_per_pack;
          const int pack_idx = hidden_dim_group_idx % num_elems_per_pack;
          scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                         (expert_idx * scale_expert_stride * num_elems_per_pack +
                          hidden_idx_packed * scale_hidden_stride * num_elems_per_pack +
                          token_idx * scale_token_stride * num_elems_per_pack + pack_idx);
        } else {
          static_assert(!SCALE_UE8M0);
          scale_output = output_s + offset_num_groups;
        }

        // can speed up if too slow
        if constexpr (IS_COLUMN_MAJOR and SCALE_UE8M0) {
          const int remainder_num_groups = hidden_dim_num_groups % num_elems_per_pack;
          if ((remainder_num_groups != 0) and (hidden_dim_group_idx == hidden_dim_num_groups - 1) and
              (lane_id < num_elems_per_pack - remainder_num_groups)) {
            const int shift = 1 + lane_id;
            *(scale_output + shift) = 0;
          }
        }

        float local_absmax = LOCAL_ABSMAX_ABS;

#pragma unroll
        for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
          float val;
          if constexpr (FUSE_SILU_AND_MUL) {
            // TODO maybe vectorize
            T val_lowprec = static_cast<T>(silu(static_cast<float>(input_primary_vec[j]))) * input_secondary_vec[j];
            val = static_cast<float>(val_lowprec);
            input_primary_vec[j] = val_lowprec;
          } else {
            val = static_cast<float>(input_primary_vec[j]);
          }

          float abs_val = fabsf(val);
          local_absmax = fmaxf(local_absmax, abs_val);
        }

        local_absmax = GroupReduceMax<THREADS_PER_SUBWARP>(local_absmax, lane_id);

        float y_scale, y_scale_inv;
        calculate_fp8_scales<SCALE_UE8M0, dst_dtype_info>(local_absmax, y_scale, y_scale_inv);
        float2 y_scale_repeated = {y_scale, y_scale};

        if (lane_id == 0) {
          *scale_output = extract_required_scale_format<SCALE_UE8M0>(y_scale_inv);
        }

        int4 output_buf;
        static_assert(sizeof(output_buf) == INPUT_PRIMARY_VEC_SIZE * sizeof(DST_DTYPE));

        if constexpr (std::is_same_v<DST_DTYPE, torch::headeronly::Float8_e4m3fn>) {
          const auto output_buf_ptr = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
          static_assert(sizeof(output_buf) == INPUT_PRIMARY_VEC_SIZE / 2 * sizeof(__nv_fp8x2_storage_t));
          static_assert(INPUT_PRIMARY_VEC_SIZE % 2 == 0);

#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; j += 2) {
            float2 inputx2 = {static_cast<float>(input_primary_vec[j]), static_cast<float>(input_primary_vec[j + 1])};
            float2 outputx2 = fmul2_rn(inputx2, y_scale_repeated);

            outputx2.x = fminf(fmaxf(outputx2.x, dst_dtype_info::MIN), dst_dtype_info::MAX);
            outputx2.y = fminf(fmaxf(outputx2.y, dst_dtype_info::MIN), dst_dtype_info::MAX);

            output_buf_ptr[j / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
          }
        } else {
          const auto output_buf_ptr = reinterpret_cast<DST_DTYPE*>(&output_buf);

#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
            float val = static_cast<float>(input_primary_vec[j]);
            float q_val = fminf(fmaxf(val * y_scale, dst_dtype_info::MIN), dst_dtype_info::MAX);
            output_buf_ptr[j] = DST_DTYPE(q_val);
          }
        }

        st_global(
            reinterpret_cast<int4*>(output_q + offset_num_groups * GROUP_SIZE + lane_id * INPUT_PRIMARY_VEC_SIZE),
            output_buf);
      });

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void sgl_per_token_group_quant_8bit_v2(
    // vanilla: (num_tokens, hidden_size)
    // fuse_silu_and_mul: (num_tokens, hidden_size * 2)
    // fuse_silu_and_mul + masked_layout: (num_experts, num_tokens-with-padding, hidden_size * 2)
    Tensor input,
    Tensor output_q,
    Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    SGL_OPTIONAL_TENSOR_ARG) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  SGL_CHECK_NO_MSG(input.numel() > 0);

  SGL_CHECK_NO_MSG(std::abs(LOCAL_ABSMAX_ABS - eps) < 1e-13);

  CHECK_EQ(input.numel() % group_size, 0);
  const int num_groups = static_cast<int>(input.numel()) / group_size / (fuse_silu_and_mul ? 2 : 1);

  const bool masked_layout = masked_m.has_value();
  SGL_CHECK_NO_MSG(output_s.dim() == (masked_layout ? 3 : 2));

  const int num_local_experts = masked_layout ? input.size(0) : 1;

  cudaStream_t stream = SGL_CURRENT_CUDA_STREAM();

  auto dst_type = output_q.scalar_type();

  const bool is_column_major = output_s.stride(-2) < output_s.stride(-1);
  const int hidden_dim_num_groups = static_cast<int>(output_q.size(-1)) / group_size;
  const int num_tokens_per_expert = static_cast<int>(output_q.size(-2));
  const int scale_expert_stride = masked_layout ? static_cast<int>(output_s.stride(0)) : 0;
  const int scale_hidden_stride = static_cast<int>(output_s.stride(-1));

#define LAUNCH_KERNEL_INNER(SCHEDULER, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, output_s_dtype, ...)           \
  do {                                                                                                               \
    int subwarps_per_block;                                                                                          \
    dim3 grid, block;                                                                                                \
    SCHEDULER::compute_exec_config(                                                                                  \
        THREADS_PER_SUBWARP, num_local_experts, hidden_dim_num_groups, num_groups, subwarps_per_block, grid, block); \
                                                                                                                     \
    cudaLaunchConfig_t config;                                                                                       \
    config.gridDim = grid;                                                                                           \
    config.blockDim = block;                                                                                         \
    config.dynamicSmemBytes = 0;                                                                                     \
    config.stream = stream;                                                                                          \
    cudaLaunchAttribute attrs[1];                                                                                    \
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                                \
    attrs[0].val.programmaticStreamSerializationAllowed = SGL_GET_ENV_ENABLE_PDL();                                  \
    config.numAttrs = 1;                                                                                             \
    config.attrs = attrs;                                                                                            \
    cudaLaunchKernelEx(                                                                                              \
        &config,                                                                                                     \
        per_token_group_quant_8bit_kernel<SCHEDULER, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, __VA_ARGS__>,    \
        SGL_INPUT_PTR(input, T),                                                                                     \
        SGL_OUTPUT_Q_PTR(output_q, DST_DTYPE),                                                                       \
        SGL_OUTPUT_S_PTR(output_s, output_s_dtype),                                                                  \
        SGL_MASKED_M_PTR(masked_m),                                                                                  \
        subwarps_per_block,                                                                                          \
        hidden_dim_num_groups,                                                                                       \
        scale_expert_stride,                                                                                         \
        scale_hidden_stride,                                                                                         \
        num_tokens_per_expert);                                                                                      \
  } while (0)

#define LAUNCH_KERNEL(GROUP_SIZE, T, DST_DTYPE)                                                                     \
  do {                                                                                                              \
    constexpr int THREADS_PER_SUBWARP = GROUP_SIZE / 16;                                                            \
    SGL_CHECK_NO_MSG(THREADS_PER_SUBWARP * INPUT_PRIMARY_VEC_NUM_BYTES == group_size * sizeof(T));                  \
                                                                                                                    \
    using dst_dtype_info = DtypeInfo<DST_DTYPE>;                                                                    \
    CHECK_EQ(dst_dtype_info::MIN, min_8bit);                                                                        \
    CHECK_EQ(dst_dtype_info::MAX, max_8bit);                                                                        \
                                                                                                                    \
    if (is_column_major) {                                                                                          \
      if (scale_ue8m0) {                                                                                            \
        if (fuse_silu_and_mul) {                                                                                    \
          if (masked_layout) {                                                                                      \
            LAUNCH_KERNEL_INNER(                                                                                    \
                MaskedLayoutScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, true);  \
          } else {                                                                                                  \
            LAUNCH_KERNEL_INNER(                                                                                    \
                NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true, true);         \
          }                                                                                                         \
        } else {                                                                                                    \
          LAUNCH_KERNEL_INNER(NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, uint32_t, true, true); \
        }                                                                                                           \
      } else {                                                                                                      \
        LAUNCH_KERNEL_INNER(NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, true);            \
      }                                                                                                             \
    } else {                                                                                                        \
      LAUNCH_KERNEL_INNER(NaiveScheduler, GROUP_SIZE, THREADS_PER_SUBWARP, T, DST_DTYPE, float, false);             \
    }                                                                                                               \
  } while (0)

#define LAUNCH_KERNEL_OUTER(...)                  \
  switch (group_size) {                           \
    case 16:                                      \
      LAUNCH_KERNEL(16, __VA_ARGS__);             \
      break;                                      \
    case 32:                                      \
      LAUNCH_KERNEL(32, __VA_ARGS__);             \
      break;                                      \
    case 64:                                      \
      LAUNCH_KERNEL(64, __VA_ARGS__);             \
      break;                                      \
    case 128:                                     \
      LAUNCH_KERNEL(128, __VA_ARGS__);            \
      break;                                      \
    default:                                      \
      SGL_CHECK(false, "Unsupported group_size"); \
  }                                               \
  while (0)

  SGL_DISPATCH_FLOAT_FP16(input.scalar_type(), "sgl_per_token_group_quant_8bit_v2", [&] {
    if (dst_type == ScalarType::Char) {
      LAUNCH_KERNEL_OUTER(scalar_t, int8_t);
      return true;
    } else if (dst_type == ScalarType::Float8_e4m3fn) {
      LAUNCH_KERNEL_OUTER(scalar_t, torch::headeronly::Float8_e4m3fn);
      return true;
    }
    return false;
  });

#undef LAUNCH_KERNEL
#undef LAUNCH_KERNEL_INNER
}

#undef SGL_CHECK
#undef SGL_CHECK_NO_MSG
#undef SGL_CURRENT_CUDA_STREAM
#undef SGL_GET_ENV_ENABLE_PDL
#undef SGL_INPUT_PTR
#undef SGL_OUTPUT_Q_PTR
#undef SGL_OUTPUT_S_PTR
#undef SGL_MASKED_M_PTR
#undef SGL_OPTIONAL_TENSOR_ARG
#undef SGL_DISPATCH_FLOAT_FP16
#ifdef TORCH_TARGET_VERSION
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
#undef CHECK_EQ
#undef SGL_DISPATCH_CASE_BF16
#endif
