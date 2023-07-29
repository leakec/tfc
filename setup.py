import sys
from os import path, name
import numpy
from setuptools import setup, Extension, find_packages
from setuptools.command.build_py import build_py as _build_py

# Get long description
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
long_description = long_description.replace('<img src="https://github.com/leakec/tfc/blob/main/docs/Univariate_TFC_Animation.gif" width="600" height="467">',"",1)

# Get numpy directory
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# Get version info
version_dict = {}
with open('src/tfc/version.py') as f:
  exec(f.read(), version_dict)
  version = version_dict["__version__"]

# In the future, can add -DHAS_CUDA to this to enable GPU support
if name == 'nt':
    # Windows compile flags
    cxxFlags = ["\O3", "\std=c++17", "\Wall", "\fPIC"]
else:
    cxxFlags = ["-O3", "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter", "-fPIC"]

# Create basis function c++ extension
BF = Extension(
    "tfc.utils.BF._BF",
    sources=["src/tfc/utils/BF/BF.i","src/tfc/utils/BF/BF.cxx"],
    include_dirs=["src/tfc/utils/BF", numpy_include],
    swig_opts=["-c++", "-doxygen", "-O", "-olddefs"],
    extra_compile_args=cxxFlags,
    extra_link_args=cxxFlags,
)

# Custom build options to include swig Python files
class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        super(build_py, self).run()

if sys.version_info >= (3, 8):
    numpy_version = "numpy>=1.23.0"
else:
    numpy_version = "numpy>=1.21.0"

# Setup
setup(
    name="tfc",
    version=version,
    author="Carl Leake and Hunter Johnston",
    author_email="leakec57@gmail.com",
    description="Theory of Functional Connections (TFC): A functional interpolation framework with applications in differential equations.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/leakec/tfc.git",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["src/tfc/py.typed"]},
    include_package_data=True,
    ext_modules=[BF],
    install_requires=[
        numpy_version,
        "jax",
        "jaxlib",
        "jaxtyping",
        "annotated-types",
        "matplotlib",
        "colorama",
        "graphviz",
        "yattag",
        "sympy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Education",
    ],
    cmdclass={
        "build_py": build_py,
    },
)
