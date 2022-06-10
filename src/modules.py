import copy
from math import cos
import parser
import numpy as np


import os
import sys

import warnings
warnings.filterwarnings('ignore')

class CubicSplineInterpolator:
    def __init__(self, xGrid, fGrid, sppa=0, sppb=0):

        self.sppa = sppa
        self.sppb = sppb
        self.xGrid = xGrid
        self.fGrid = fGrid
        self.N = len(xGrid)
        self.coeffs = self.ComputeCoefficients(xGrid, fGrid)

    def create_matrix(self, hGrid, yGrid):
        A = np.zeros((self.N, self.N))
        b = np.zeros((self.N, 1))
        b[0][0] = self.sppa
        A[0, 0] = 1
        A[-1, -1] = 1

        for i in range(1, self.N - 1):
            A[i, i - 1] = hGrid[i - 1]
            A[i, i + 1] = hGrid[i]
            A[i, i] = 2 * (hGrid[i - 1] + hGrid[i])
            b[i, 0] = 3 * (yGrid[i] / hGrid[i] - yGrid[i - 1] / hGrid[i - 1])

        return A, b

    def solve_matrix(self, A, b):
        alphas = np.zeros(self.N)
        betas = np.zeros(self.N)

        alphas[0] = A[0, 1] / A[0, 0]
        betas[0] = b[0][0] / A[0, 0]

        for i in range(1, self.N - 1):
            alphas[i] = A[i, i + 1] / (A[i, i] - A[i, i - 1] * alphas[i - 1])
            betas[i] = (b[i][0] - A[i, i - 1] * betas[i - 1]) / (
                A[i, i] - A[i, i - 1] * alphas[i - 1])

        coefs = np.zeros(self.N)
        coefs[self.N - 1] = self.sppb

        for i in range(self.N - 2, -1, -1):
            coefs[i] = betas[i] - alphas[i] * coefs[i + 1]

        return coefs

    def ComputeCoefficients(self, xGrid, fGrid):
        coefs = np.array([])

        hGrid = np.array([])
        for i in range(1, len(xGrid)):
            hGrid = np.append(hGrid, xGrid[i] - xGrid[i - 1])

        coefs_A = copy.deepcopy(fGrid)[:-1]

        yGrid = np.array([])
        for i in range(1, len(fGrid)):
            yGrid = np.append(yGrid, fGrid[i] - fGrid[i - 1])

        A, c = self.create_matrix(hGrid, yGrid)

        coefs_C = self.solve_matrix(A, c)
        coefs_B = np.zeros((self.N - 1))
        coefs_D = np.zeros((self.N - 1))

        for i in range(0, self.N - 1):
            coefs_D[i] = (coefs_C[i + 1] - coefs_C[i]) / (3 * hGrid[i])
            coefs_B[i] = (yGrid[i] / hGrid[i]) - (hGrid[i] / 3) * (
                2 * coefs_C[i] + coefs_C[i + 1])

        coefs_C = coefs_C[:-1]

        return np.vstack((coefs_A, coefs_B, coefs_C, coefs_D)).T

    def Compute(self, x_):
        res = []
        try:
            len(x_)
            for x in x_:
                ans = np.inf
                for i, point in enumerate(self.xGrid[1:]):
                    if x <= point:
                        a, b, c, d = self.coeffs[i]
                        f = a + b * (x - self.xGrid[i]) + c * (
                            x - self.xGrid[i])**2 + d * (x - self.xGrid[i])**3
                        ans = f
                        break

                res.append(ans)
            return res
        except Exception:
            ans = np.inf
            x = x_
            for i, point in enumerate(self.xGrid[1:]):
                if x <= point:
                    a, b, c, d = self.coeffs[i]
                    f = a + b * (x - self.xGrid[i]) + c * (
                        x - self.xGrid[i])**2 + d * (x - self.xGrid[i])**3
                    ans = f
                    break

            return ans


## возвращает интерполяцию производной

def spline_deriv(CubicSpline):
    """
    CubicSpline - Построенная интерполяция сплайнами
    """
    
    derivative_splines = []
    
    for spline in CubicSpline.coeffs:
        size = len(spline)
        new_coefs = []
        for i in range(1, size):
            new_coefs.append(i * spline[i])
        new_coefs.append(0)
        derivative_splines.append(new_coefs)
        
    new_CubicSpline = CubicSplineInterpolator(CubicSpline.xGrid, CubicSpline.fGrid)
    new_CubicSpline.coeffs = derivative_splines
    return new_CubicSpline


def simpson_method(func, n_segments, segment):
    """
    func - интегрируемая функция
    n_segments - количество делений
    segment - интервал интегрирования
    """
    
    segments = np.linspace(segment[0], segment[1], n_segments)
    sum_ = 0
    for i in range(n_segments - 1):
        sum_ += ((segments[i + 1] - segments[i]).astype(float) / 6) * \
        (func(segments[i]) + 4 * func(0.5 * (segments[i] + segments[i + 1])) + func(segments[i + 1]))\
        .astype(float)
        
    return sum_



## Решение задачи Коши методом Эйлера

def cauchy_solver(funcs, start_points, tGrid):
    """
    f1 - z'(t) * integral_y^1(ro(w) dw)
    f2 - функция корреции параметра y
    x_0 - аргументы f1
    y_0 - аргументы f2
    tGrid - сетка
    """
    
    f1, f2 = funcs
    x_0, y_0 = start_points
    x = np.array([x_0])
    y = np.array([y_0])
    
    h = tGrid[1] - tGrid[0]
    fl = 0
    
    for i in range(1, len(tGrid)):
        x_prev = x[i - 1]
        y_prev = y[i - 1]
        t_prev = tGrid[i - 1]
        
        x_new = x_prev + h * f1(t_prev, y_prev)
        y_new = y_prev + h * f2(t_prev, x_prev)
        
        if y_new > 1 or y_new < 0:
            fl = 1
            break
            
        x = np.append(x, x_new)
        y = np.append(y, y_new)
        
    return x, y, tGrid, fl

def C1(func_x, func_y, ro, n_knots, tGrid, T):
    x_deriv_interpol = spline_deriv(func_x)
    dw = lambda w: w * ro(w)
    integral_dw = lambda t: simpson_method(func=dw, n_segments=n_knots, segment=[func_y.Compute(t), 1])
    fGrid = integral_dw(tGrid)
    integral_interpol = CubicSplineInterpolator(tGrid, fGrid)
    
    dt = lambda t: x_deriv_interpol.Compute(t) * integral_dw(t)
    integral_dt = simpson_method(func=dt, n_segments=n_knots, segment=[0, T])
    error = 1 - (integral_dt / (func_x.Compute(T) - func_x.Compute(0)))
    return error

def C2(func_x, S, T):
    return abs(func_x.Compute(T) - S(T)) / S(T)

def PHI(C1, C2):
    return C1 + 10 * C2

## Оберточный класс для функции коррекции
class CorrectionFunc:
    def __init__(self, expr, b, z, S):
        self.beta = b
        self.z = z
        self.S = S
        self.form = parser.expr(expr).compile()
        
    def __call__(self, t, x):
        z = self.z(t)
        S = self.S(t)
        beta = self.beta
        return eval(self.form)