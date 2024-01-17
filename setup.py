import glob

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def get_extensions():
    extensions = []

    op_files = glob.glob('./src/fnnfunctor/ops/csrc/*.c*')
    extension = CUDAExtension
    ext_name = 'fnnfunctor.ops._ext'

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
    )

    extensions.append(ext_ops)

    return extensions

if __name__ == "__main__":
    setup(
        # ext_modules=get_extensions(),
        # cmdclass={'build_ext': BuildExtension},
    )
