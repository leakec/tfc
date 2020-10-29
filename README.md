## Theory of Functional Connections (TFC)
**A functional interpolation method with applications in solving differential equations.**

This repository is currently under development. A public release to PyPi is planned for sometime during the spring semester. For now, if you are interested, feel free to download this GitHub and build from the source directly. 

# Build Instructions:
1) Create a dist directory in the main folder.
1) Run python setup.py bdist\_wheel from the main folder.
2) Navigate to dist, and run pip3 install "wheel" where "wheel" is the name of the wheel created in the previous step.

Dependencies:
	System Packages:
		-doxygen
		-graphviz
		-swig
		-cmake
		-python3-sphinx

	Python Packages:
		-matplotlib
		-graphviz
		-yattag
		-sphinx
		-sphinx\_rtd\_theme
		-nbsphinx
		-breathe
		-jax
		-jaxlib
		-colorama

# Build Documentation Instructions:
Currently, the documentation and source code are built seperately. To build the documentation follow these instructions.
1) Create a build directory in the main directory if it does not already exists.
2) From the build directory run cmake ..
3) From the build director run make docs
The code documentation will appear under build/src/doc/docs/sphinx and the main file is index.html. 

# Testing instructions:
After following the build instructions:
1) Navigate to the tests directory.
2) Run py.test or python -m pytest
These serve as simple unit tests that test basic functionality of the code. These include tests for individual TFC functions, as well as full ODE and PDE tests.
