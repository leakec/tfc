import sys
import os
from pathlib import Path
import numpy
from setuptools import setup, Extension, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext
from subprocess import check_call

# Get long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
long_description = long_description.replace(
    '<img src="https://github.com/leakec/tfc/blob/main/docs/Univariate_TFC_Animation.gif" width="600" height="467">',
    "",
    1,
)

# Get version info
version_dict = {}
with open("src/tfc/version.py") as f:
    exec(f.read(), version_dict)
    version = version_dict["__version__"]

# In the future, can add -DHAS_CUDA to this to enable GPU support
if os.name == "nt":
    # Windows compile flags
    cxxFlags = ["/O2", "/std:c++17", "/Wall", "/DWINDOWS_MSVC"]
else:
    cxxFlags = ["-O3", "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter", "-fPIC"]

if sys.version_info >= (3, 10):
    numpy_version = "numpy>=2.1.0"
else:
    numpy_version = "numpy>=1.21.0"

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = str((Path(sourcedir) / "src" / "tfc" / "utils" / "BF").absolute())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parents[0].absolute()
        bf_dir = extdir / "tfc" / "utils" / "BF"

        import pybind11
        dark = Path(pybind11.__file__).parents[0]
        pybind11_dir = dark / "share" / "cmake" / "pybind11"


        cfg = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_INSTALL_PREFIX={bf_dir}",
            f"-Dpybind11_DIR={pybind11_dir}"
        ]

        # Optional: use Ninja if available
        if "CMAKE_GENERATOR" not in os.environ:
            cmake_args += ["-G", "Ninja"]

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Run CMake configuration
        check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)

        # Run CMake build
        check_call(["cmake", "--build", ".", "--config", cfg], cwd=build_temp)

        # Run CMake install
        check_call(["cmake", "--install", "."], cwd=build_temp)


# Setup
setup(
    name="tfc",
    version=version,
    author="Carl Leake and Hunter Johnston",
    author_email="leakec57@gmail.com",
    description="Theory of Functional Connections (TFC): A functional interpolation framework with applications in differential equations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leakec/tfc.git",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["src/tfc/py.typed"]},
    python_requires=">=3.10",
    include_package_data=True,
    ext_modules=[CMakeExtension("BF")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        numpy_version,
        "jax ~= 0.6.0",
        "jaxlib ~= 0.6.0",
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
)
