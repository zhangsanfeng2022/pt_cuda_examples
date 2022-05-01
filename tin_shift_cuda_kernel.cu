#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#define THREADS_PER_BLOCK 512
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
	  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
			         i += blockDim.x * gridDim.x)
namespace {
template <typename T>
__global__ void tin_shift_forward_cuda_kernel(
    const int nthreads, const T* input, const int* shift, T* output,
    const int batch_size, const int channels, const int t_size,
    const int hw_size, const int group_size, const int group_channel) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int hw_index = index % hw_size;
    const int j = (index / hw_size) % channels;

    const int n_index = (index / hw_size / channels) % batch_size;
    int group_id = j / group_channel;
    int t_shift = shift[n_index * group_size + group_id];
    int offset = n_index * t_size * hw_size * channels + hw_size * j + hw_index;
    for (int i = 0; i < t_size; i++) {
      int now_t = i + t_shift;
      int data_id = i * hw_size * channels + offset;
      if (now_t < 0 || now_t >= t_size) {
        continue;
      }
      int out_id = now_t * hw_size * channels + offset;
      output[out_id] = input[data_id];
    }
  }
}

template <typename T>
__global__ void tin_shift_backward_cuda_kernel(
    const int nthreads, const T* input, const int* shift, T* output,
    const int batch_size, const int channels, const int t_size,
    const int hw_size, const int group_size, const int group_channel) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int hw_index = index % hw_size;
    const int j = (index / hw_size) % channels;

    const int n_index = (index / hw_size / channels) % batch_size;
    int group_id = j / group_channel;
    int t_shift = shift[n_index * group_size + group_id];
    int offset = n_index * t_size * hw_size * channels + hw_size * j + hw_index;
    for (int i = 0; i < t_size; i++) {
      int now_t = i + t_shift;
      int data_id = i * hw_size * channels + offset;
      if (now_t < 0 || now_t >= t_size) {
        continue;
      }
      int out_id = now_t * hw_size * channels + offset;
      output[out_id] = input[data_id];
    }
  }
}

} // namespace

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

void TINShiftForwardCUDAKernelLauncher(torch::Tensor input, torch::Tensor shift,
                                       torch::Tensor output) {
  int output_size = output.numel();
  int batch_size = input.size(0);
  int t_size = input.size(1);
  int channels = input.size(2);
  int hw_size = input.size(3);
  int group_size = shift.size(1);
  int group_channel = channels / group_size;
  int num_kernels = batch_size * hw_size * channels;

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "tin_shift_forward_cuda_kernel", [&] {
        tin_shift_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(), shift.data_ptr<int>(),
                output.data_ptr<scalar_t>(), batch_size, channels, t_size,
                hw_size, group_size, group_channel);
      });
}

void TINShiftBackwardCUDAKernelLauncher(torch::Tensor grad_output, torch::Tensor shift,
                                        torch::Tensor grad_input) {
  int output_size = grad_output.numel();
  int batch_size = grad_output.size(0);
  int t_size = grad_output.size(1);
  int channels = grad_output.size(2);
  int hw_size = grad_output.size(3);
  int group_size = shift.size(1);
  int group_channel = channels / group_size;
  int num_kernels = batch_size * hw_size * channels;

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "tin_shift_backward_cuda_kernel", [&] {
        tin_shift_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                shift.data_ptr<int>(), grad_input.data_ptr<scalar_t>(),
                batch_size, channels, t_size, hw_size, group_size,
                group_channel);
      });
}
