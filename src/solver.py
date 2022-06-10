import copy
from math import cos
import parser
import numpy as np

import os
import sys

project_path = os.path.abspath(os.path.join('..'))

if project_path not in sys.path:
    sys.path.append(project_path)


from src.modules import * 


import warnings
warnings.filterwarnings('ignore')

def search_beta(ro, z, S, cor_funcs, start_points, T, n_knots=150, auto_=False):
    """
    ro - распределение аудитории
    z - нарастающий трафик
    S - плановое значение
    cor_funcs - функции коррекции
    start_points - стартовые точеи
    T - длительность размещения
    n_knots - количество точек
    auto_ - флаг режима
    """
    
    def manual():
        print("Manual")
        func1 = lambda t, y: z_derivative_interpol.Compute(t) * integral_interpol.Compute(y)
        F1_Grid, F2_Grid, xGrid, _ = cauchy_solver([func1, cor_funcs], [x_0, y_0], tGrid)
        func_x = CubicSplineInterpolator(xGrid, F1_Grid)
        func_y = CubicSplineInterpolator(xGrid, F2_Grid)
        error_c1 = C1(func_x, func_y, ro, n_knots, tGrid, T)
        error_c2 = C2(func_x, S, T)
        phi = PHI(error_c1, error_c2)
        print(f"Beta = {cor_funcs.beta}")
        print(f"Ф = {phi}")
        return [(tGrid, F2_Grid, F1_Grid, x_0, y_0, cor_funcs.beta, error_c1, error_c2, phi)]
    
    def auto(x_0, y_0):
        print("Auto")
        results = []
        for cor_func in cor_funcs:

            F1_Grid, F2_Grid, xGrid, fl = cauchy_solver([func1, cor_func], [x_0, y_0], tGrid)
            if fl: 
                continue

            func_x = CubicSplineInterpolator(xGrid, F1_Grid)
            func_y = CubicSplineInterpolator(xGrid, F2_Grid)            
            error_c1 = C1(func_x, func_y, ro, n_knots, tGrid, T)
            error_c2 = C2(func_x, S, T)
            phi = PHI(error_c1, error_c2)
            results.append(phi)
            
        results = np.array(results)
        if not len(results):
            print("No beta satisfies: y in [0, 1]")
            return
        best_score = np.min(results)
        best_f = cor_funcs[np.argmin(results)]
        best_beta = best_f.beta
        print(f"Best beta = {best_beta}")
        print(f"Ф = {best_score}")
        x0_ = np.linspace(0, 5, 5)
        y0_ = np.linspace(0, 0.5, 5)
        solutions = []
        for x_0, y_0 in zip(x0_, y0_):
            F1_Grid, F2_Grid, xGrid, fl = cauchy_solver([func1, best_f], [x_0, y_0], tGrid)
            if fl:
                continue

            func_x = CubicSplineInterpolator(xGrid, F1_Grid)
            func_y = CubicSplineInterpolator(xGrid, F2_Grid)
            error_c1 = C1(func_x, func_y, ro, n_knots, tGrid, T)
            error_c2 = C2(func_x, S, T)
            phi = PHI(error_c1, error_c2)
            solutions.append((tGrid, F2_Grid, F1_Grid, x_0, y_0, best_beta, error_c1, error_c2, phi))
            
        return solutions
        
    x_0, y_0 = start_points
    tGrid = np.linspace(0, T, n_knots)
    fGrid = np.linspace(0, 1, n_knots)
    zGrid = z(tGrid)
    z_interpol = CubicSplineInterpolator(tGrid, zGrid)
    z_derivative_interpol = spline_deriv(z_interpol)
    integral = lambda y: simpson_method(func=ro, n_segments=n_knots, segment=[y, 1])
    integralGrid = integral(tGrid)
    integral_interpol = CubicSplineInterpolator(fGrid, integralGrid)
    func1 = lambda t, y: z_derivative_interpol.Compute(t) * integral_interpol.Compute(y)    
    
    if auto_:
        solutions = auto(x_0, y_0)
        return solutions
    else:
        solution = manual()
        return solution