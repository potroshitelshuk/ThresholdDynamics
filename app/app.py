import copy
import matplotlib.pyplot as plt
from math import cos
import warnings
warnings.filterwarnings('ignore')

import parser
import numpy as np

import os
import sys

project_path = os.path.abspath(os.path.join('..'))

if project_path not in sys.path:
    sys.path.append(project_path)

from src.solver import *





# if __name__ == "__main__":
# 	z = lambda t: 4 * t + np.cos(t)
# 	ro = lambda x: 6 * x * (1 - x)
# 	S = lambda t: 3 * t + np.sin(t)
# 	beta = 0.01
# 	cor_funcs = CorrectionFunc("beta * (S - x)", beta, z, S)
# 	y_0 = 0
# 	x_0 = 0
# 	T = 1
# 	solutions = search_beta(ro, z, S, cor_funcs, [x_0, y_0], T, 150, auto_=False)


if __name__ == "__main__":
	z = lambda t: 4 * t + np.cos(t)
	ro = lambda x: 6 * x * (1 - x)
	S = lambda t: 3 * t + np.sin(t)
	bgrid = np.array([10 ** x for x in range(-10, 2)])
	cor_funcs = [CorrectionFunc("beta * (S - x)", beta, z, S) for beta in bgrid]
	y_0 = 0
	x_0 = 0
	T = 1
	solutions = search_beta(ro, z, S, cor_funcs, [x_0, y_0], T, 150, auto_=True)

	for i, solution in enumerate(solutions):
	    tGrid, F2_Grid, F1_Grid, x_0, y_0, best_beta, error_c1, error_c2, phi = solutions[i]
	    print(f"Solution {i}: x_0 = {x_0} \ny_0 = {y_0}\nbeta = {best_beta}\nLoss(C1, C2) = {round(phi, 2)}" + "\n")