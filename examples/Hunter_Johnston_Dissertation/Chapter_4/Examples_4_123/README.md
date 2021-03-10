# Solution to the Lane Emden Equation
# This file describes each file in this folder and its function

1. mainLaneEmden.py
    - This file is used to produce the following plots:
    - Example 4.1: Figures 4.3 and 4.4
    - Example 4.2: Figures 4.6 and 4.7
    - Example 4.3: Figures 4.9 and 4.10
** Note: The section TEST PARAMETERS is used to vary problem or tfc specific parameters. For example, type is specified as either 0, 1,or 5 according to the variable 'a' in the dissertation. Additionally, the runtime of this file is slower since, in order to provide modular solutions files for tfc, spectral, and ode, the code is forced to reinitalize the tfc class and re-jit the function each function call. This can be avoided and optimized when not doing a bassis function sweep.

2. timeLaneEmden.py
    - This file is used to produce the following plots:
        - Example 4.1: Figure 4.5
        - Example 4.2: Figure 4.8
        - Example 4.3: Figure 4.11
** Note: The structure of this code mirrors mainLaneEmden.py with the associated disclaimer as well. Also note, the run times will be dependent on computer hardware so they my note match those presented in the dissertation.

3. CALLED FUNCTIONS
    - laneEmden_tfc.py       --> solves with TFC
    - laneEmden_spectral.py  --> solves with spectral method
    - laneEmden_ode.py       --> solve with numpy solver (RK45)
