# Solution to the BVP Equation
# This file describes each file in this folder and its function

1. mainBVP.py
    This file is used to produce the following plots:
    Example 4.4: Figures 4.12 and 4.13
** Note: The section TEST PARAMETERS is used to vary problem or tfc specific parameters. Additionally, the runtime of this file is slower since, in order to provide modular solutions files for tfc, spectral, and ode, the code is forced to reinitalize the tfc class and re-jit the function each function call. This can be avoided and optimized when not doing a bassis function sweep.

2. timeBVP.py
    This file is used to produce the following plots:
        Example 4.4: Figure 4.14
** Note: The structure of this code mirrors mainBVP.py with the associated disclaimer as well. Also note, the run times will be dependent on computer hardware so they my note match those presented in the dissertation.

3. CALLED FUNCTIONS
    - BVP_tfc.py       --> solves with TFC
    - BVP_spectral.py  --> solves with spectral method
    - BVP_ode.py       --> solve with numpy solver (RK45)
