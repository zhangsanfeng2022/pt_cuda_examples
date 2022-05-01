#include <torch/extension.h>

void TINShiftForwardCUDAKernelLauncher(torch::Tensor input,
		                       torch::Tensor shift,
		                       torch::Tensor output);

void TINShiftBackwardCUDAKernelLauncher(torch::Tensor grad_output,
		                        torch::Tensor shift,
		                        torch::Tensor grad_input);

void tin_shift_forward(torch::Tensor input, torch::Tensor shift, torch::Tensor output){
  TINShiftForwardCUDAKernelLauncher(input, shift, output);
}

void tin_shift_backward(torch::Tensor grad_output, torch::Tensor shift, torch::Tensor grad_input) {
  TINShiftBackwardCUDAKernelLauncher(grad_output, shift, grad_input);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tin_shift_forward, "LLTM forward (CUDA)");
  m.def("backward", &tin_shift_backward, "LLTM backward (CUDA)");
}
