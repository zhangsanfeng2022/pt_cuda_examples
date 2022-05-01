from torch.utils.cpp_extension import load
lltm_cuda = load(
    'tin_shift_cuda', ['tin_shift_cuda.cpp', 'tin_shift_cuda_kernel.cu'], verbose=True)
help(tin_shift_cuda)
