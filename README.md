# LAVD_RCLV-Orthogonal-Parallel-Architecture
Orthogonal parallel detection of global rotationally coherent Lagrangian vortices

## How to Use
1. Define the LAVD & RCLV parameters first
2. Start the process with LAVD_start.py
3. Calculate GPU-based parallel LAVD in Algorithm 1: LAVD_start.py and LAVD_SLA.py
4. Extract CPU-based parallel RCLV in Algorithm 2: RCLV_start.py and ***i4_eddy_detect_core.py (non-open source)**

## References
When using this code, please cite the following source for the underlying theory: 

Tian, Fenglin, Mengjiao Wang, Xiao Liu, Qiu He, and Ge Chen. (2022). SLA-based orthogonal parallel detection of global rotationally coherent Lagrangian vortices. Journal of Atmospheric and Oceanic Technology, 39(6), 823–836, https://doi.org/10.1175/JTECH-D-21-0103.1.

## NOTE
It is mostly being used internally by our group and is still under development.

---------------------------------------
@author: Mengjiao Wang and Xiao Liu

Environment: Anaconda 3/Python 3.8/CUDA v11.0

Created on 2020.9.11

Modified on 2022.7.6
