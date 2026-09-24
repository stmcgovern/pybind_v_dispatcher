import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

srcs = ["bindings.cpp"]
ver = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
if ver >= (2, 10):                       # Arm C needs the stable ABI headers
    srcs.append("stable.cpp")
else:
    print(f"torch {torch.__version__} < 2.10: skipping the STABLE_TORCH_LIBRARY arm")

setup(name="pybind_v_dispatcher",
      ext_modules=[CppExtension("pvd", srcs, extra_compile_args={"cxx": ["-O3"]})],
      cmdclass={"build_ext": BuildExtension})
