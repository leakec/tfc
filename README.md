# Theory of Functional Connections (TFC)
**A functional interpolation method with applications in solving differential equations.**

This repository is currently under development. 

## Installation:
The following instructions can be used to install a source distribution via pip or build TFC directly from source.

### pip installation:
```bash
pip install --upgrade pip
pip install --upgrade tfc
```

### Build from source:
1. Create a dist directory in the main directory.
2. Run python setup.py bdist\_wheel from the main directory.
3. Navigate to dist, and run pip3 install "wheel" where "wheel" is the name of the wheel created in the previous step.

Dependencies:
* System Packages:
  * swig
  * cmake
  * graphviz
* Python Packages:
  * matplotlib
  * jax
  * jaxlib
  * colorama
  * graphviz
  * yattag

## Build Code Documentation from Source:
Currently, the documentation and source code are built seperately. To build the documentation follow these instructions.
1. Change into the docs directory.
2. Run:
```bash
sphinx-build . _build/html
```
The code documentation will appear under \_build/html and the main file is index.html. This file can also be accessed using SphinxDocumentation.html symbolic link.

Dependencies:
* System Packages:
  * cmake
  * graphviz
  * doxygen
  * python3-sphinx
* Python Packages:
  * sphinx
  * sphinx\_rtd\_theme
  * nbsphinx
  * breathe
  * exhale

## Testing instructions:
After following the build instructions:
1. Navigate to the tests directory.
2. Run py.test or python -m pytest
These serve as simple unit tests that test basic functionality of the code. These include tests for individual TFC functions, as well as full ODE and PDE tests.

## Mathematical Documentation:
Any users interested in the theory behind this method should start [here](https://www.mdpi.com/2227-7390/8/8/1303); note that this journal article is open access, so you should be able to download it for free. The curious user can continue their study of the theory by visiting this [link](https://www.researchgate.net/project/Theory-of-Functional-Connections) for a complete list of TFC publications with free downloadable PDFs.

## Citing this repository:
The authors of this repsitory and the associated theory have gone to lengths to ensure that both are publicy available at no cost to the user. All that we ask in return is that if you use them, please add a reference to this GitHub and following journal article. Thank you.
```
@misc{tfc2020github,
    author = {Carl Leake and Hunter Johnston},
    title = {{TFC: A Functional Interpolation Framework}},
    url = {https://github.com/leakec/tfc},
    version = {0.0.1},
    year = {2020},
}
@article{TFC, 
    title={The Multivariate Theory of Functional Connections: Theory, Proofs, and Application in Partial Differential Equations}, 
    volume={8}, 
    ISSN={2227-7390}, 
    url={http://dx.doi.org/10.3390/math8081303}, 
    DOI={10.3390/math8081303},
    number={8}, 
    journal={Mathematics},
    publisher={MDPI AG},
    author={Leake, Carl and Johnston, Hunter and Mortari, Daniele}, 
    year={2020}, 
    month={Aug}, 
    pages={1303}
}
```
