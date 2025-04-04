from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="diffstreamlines",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=[
        CUDAExtension(
            name="diffstreamlines._C",
            sources=["src/kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
