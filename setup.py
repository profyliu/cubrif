from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


setup(name='cubrif',
        version="1.4.2",
        description = "Build random forests using CUDA GPU.",
        author = 'Yanchao Liu',
        author_email = 'yanchaoliu@wayne.edu',
        url = 'https://pypi.org/project/cubrif/',
        packages = ['cubrif'],
        ext_modules=[
            Extension('cubrifc',
                        sources = ['pycubrif.cpp'],
                        include_dirs = ['.'],
                        library_dirs=["."],
                        libraries=['cubrif']
                        )
            ]
      )
