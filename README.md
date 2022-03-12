
# Theory of Functional Connections (TFC)
**A functional interpolation framework with applications in solving differential equations.**

![Continuous integration](https://github.com/leakec/tfc/actions/workflows/ci.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/tfc)

[**Installation guide**](#installation)
| [**Reference documentation**](https://tfc-documentation.readthedocs.io/en/latest/)
| [**Mathematical documentation**](#mathematical-documentation)

<img src="https://github.com/leakec/tfc/blob/main/docs/Univariate_TFC_Animation.gif" width="600" height="467">

## Summary:
The tfc Python module is designed to help you quickly and easily apply the Theory of Functional Connections (TFC) to optimization problems. For more information on the code itself and code-based tutorials, see the [Reference documentation](https://tfc-documentation.readthedocs.io/en/latest/). What follows is a brief description of TFC.

TFC is a functional interpolation framework centered around the concept of the constrained expression. A constrained expression is a mathematical functional that expresses all possible functions that satisfy a set of user-defined constraints. For example, suppose you are interested in all possible functions that satisfy the constraint y(0) = 3. The TFC constrained expression for this constraint is,

<p align="center">
y(x,g(x)) = g(x) + 3 - g(0),
</p>

where g(x) is any function defined at the constraint point; by changing g(x) you get different outputs, but all of these outputs satisfy the constraint regardless of how you choose g(x). In this way, you maintain a completly unrestricted functiton, g(x), which we will call the free function, but always satsify the constraint. Neat huh?

While developing the constrained expression for our example above was trivial, as you introduce more complex constraints in *n*-dimensions, trying to derive these constrained expression by-eye, i.e., without a step-by-step framework, becomes extremely difficult. Luckily, TFC comes equiped with a straightfowrward, step-by-step process for developing constrained expressions. For more information on this process, see the [Mathematical documentation](#mathematical-documentation).

Since the constrained expressions effectively translate the set of all functions defined at the constraints&mdash;this set is represented by g(x), the domain of the functional&mdash;to the set of all functions that satisfy the constraints&mdash;this is the output or co-domain of the constrained exppression&mdash;the constrained expresssion can be used to transform constrained optimization problems into unconstrained optimization problems. For example, consider the following differential equation,

<p align="center">
y<sub>x</sub> = 2y, &nbsp;&nbsp; where &nbsp;&nbsp; y(0) = 3.
</p>

This differential equation can be viewed as an optimization problem where we seek to minimize the residual of the differential equation, i.e., minimize J where J = y<sub>x</sub> - 2y. Classicly, we would have to minimize J using y(x) subject to the constraint y(0) = 3. However, with TFC we can minimize J using g(x) where y(x,g(x)) = g(x) + 3 - 0, and g(x) is not subject to any constraints. Thus, TFC has translated our differential equation from a constrained optimization problem to an unconstrained optimization problem! The benefits doing so include:

* More accurate solutions
* Faster solutions
* Robustness to initial guess

For more information on the appliation of TFC to differential equations and its benefits see the [Mathematical documentation](#mathematical-documentation).

## Installation:
The following instructions can be used to install a source distribution via pip or build TFC directly from source. Currently, we support building TFC navitely on Linux or macOS, but Windows users can still use TFC via the Windows Subsystem for Linux.

To install via pip run:
```bash
pip install --upgrade pip setuptools wheel numpy
pip install --upgrade tfc
```
The above will install a binary TFC wheel. The developers have found that installing a source distribution leads to code that is slightly faster on some machines, as the code is compiled using potentially newer versions of compilers and swig. If you would like the source distribution, then you can use the following:
```bash
pip install tfc --no-binary tfc
```
Note that you may need to first install the system package dependencies listed in the [**Building from source**](#building-from-source) section if they are not already installed.

## Reference Documentation:
For tutorials on how to use this package as well as information about the tfc API, see the [reference documentation](https://tfc-documentation.readthedocs.io/en/latest/).

## Mathematical Documentation:
Any users interested in the process for developing constrained expressions, the mathematical theory behind TFC, and the application of TFC to differential equations should start with this [journal article](https://www.mdpi.com/2227-7390/8/8/1303); note that the article is open access, so you can download it for free. The curious user can continue their study of the mathematical theory by visiting the [TFC article repository](https://www.researchgate.net/project/Theory-of-Functional-Connections) on ResearchGate for a complete list of TFC publications with free downloadable PDFs.

## Citing this repository:
The authors of this repsitory and the associated theory have gone to lengths to ensure that both are publicy available at no cost to the user. All that we ask in return is that if you use them, please add a reference to this GitHub and following journal article. Thank you.
```
@misc{tfc2021github,
    author = {Carl Leake and Hunter Johnston},
    title = {{TFC: A Functional Interpolation Framework}},
    url = {https://github.com/leakec/tfc},
    version = {0.1.6},
    year = {2021},
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

## For developers:

### Building from source:
1. Create a dist directory in the main directory.
2. Run python setup.py bdist\_wheel from the main directory.
3. Navigate to dist, and run pip3 install "wheel" where "wheel" is the name of the wheel created in the previous step.

Dependencies:
* System Packages:
  * swig
  * graphviz
* Python Packages:
  * matplotlib
  * jax
  * jaxlib
  * colorama
  * graphviz
  * yattag

### Testing instructions:
1. Navigate to the tests directory.
2. Run py.test or python -m pytest.
These serve as simple unit tests that test basic functionality of the code. These include tests for individual TFC functions, as well as full ODE and PDE tests.

### Building Reference Documentation from Source:
If for some reason you want to build the reference documentation from source, you can do so using these two steps:
1. Change into the docs directory.
2. Run:
```bash
sphinx-build . _build/html
```
The code documentation will appear under \_build/html and the main file is index.html. This file can also be accessed using the SphinxDocumentation.html symbolic link in the docs directory.

Dependencies:
* System Packages:
  * graphviz
  * doxygen
  * python3-sphinx
* Python Packages:
  * sphinx
  * sphinx\_rtd\_theme
  * nbsphinx
  * breathe
  * exhale
  
