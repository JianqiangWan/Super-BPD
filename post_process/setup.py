from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bpd_cuda',
    ext_modules=[
        CUDAExtension('bpd_cuda', [
            'bpd_cuda.cpp',
            'bpd_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })