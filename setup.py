import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src_dir = os.path.join(os.path.dirname(__file__), "src")

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-std=c++17",
]

cxx_flags = ["-O3", "-std=c++17"]

ext = CUDAExtension(
    name="bitcache._C",
    sources=[
        "src/torch_binding.cpp",
        "src/binary_encode.cu",
        "src/residual.cu",
        "src/binary_attention.cu",
        "src/compensated_attention.cu",
        "src/online_encode.cu",
        "src/adaptive_precision.cu",
    ],
    include_dirs=[src_dir],
    extra_compile_args={
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    },
)

setup(
    name="bitcache",
    version="0.1.0",
    description="Binary KV-cache attention via XNOR-popcount with precision-preserving residual correction",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
