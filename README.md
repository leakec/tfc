# Theory of Functional Connections (TFC)
**A functional interpolation method with applications in solving differential equations.**

This repository is currently under development. A public release to PyPi is planned for sometime during the spring semester. For now, if you are interested, feel free to download this GitHub and build from the source directly. 

## Build Instructions:
#. Create a dist directory in the main directory.
#. Run python setup.py bdist\_wheel from the main directory.
#. Navigate to dist, and run pip3 install "wheel" where "wheel" is the name of the wheel created in the previous step.

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

## Build Documentation Instructions:
Currently, the documentation and source code are built seperately. To build the documentation follow these instructions.
#. Create a build directory in the main directory if it does not already exists.
#. Clear the build directory if there is anything in it. (The documentation and code build are under 2 different cmake projects that will conflict.)
#. From the build directory run cmake ../doc
#. From the build director run make docs
The code documentation will appear under build/docs/sphinx and the main file is index.html. 

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

## Testing instructions:
After following the build instructions:
#. Navigate to the tests directory.
#. Run py.test or python -m pytest
These serve as simple unit tests that test basic functionality of the code. These include tests for individual TFC functions, as well as full ODE and PDE tests.
