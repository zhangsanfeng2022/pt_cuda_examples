from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tin_shift_cuda',
    ext_modules=[
        CUDAExtension('tin_shift_cuda', [
            'tin_shift.cpp',
            'tin_shift_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
