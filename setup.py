from cmaketools import setup

# read the contents your README
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="tfc",
    version="0.0.10",
    author="Carl Leake and Hunter Johnston",
    author_email="leakec57@gmail.com",
    description="Theory of Functional Connections (TFC): A functional interpolation framework with applications in differential equations.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/leakec/tfc.git",
    license="MIT",
    classifiers = ["Development Status :: 4 - Beta",
                   "License :: OSI Approved :: MIT License",
                   "Natural Language :: English",
                   "Programming Language :: C++",
                   "Programming Language :: Python :: 3 :: Only",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Education"],
    src_dir="src",
    test_dir="tests",
    ext_module_dirs=["cxx","swig","python"],
    has_package_data=False,
    install_requires=["numpy",
                      "jax",
                      "jaxlib",
                      "matplotlib",
                      "colorama",
                      "graphviz",
                      "yattag"],
)
