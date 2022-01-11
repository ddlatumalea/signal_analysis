'''This module provides the figures for the accompanying Jupyter notebook'''

import matplotlib.pyplot as plt, numpy as np, pandas as pd
from IPython.display import display
from scipy.interpolate import lagrange, interp1d, Akima1DInterpolator, CubicSpline, PchipInterpolator, CubicHermiteSpline


def figure1():
    plt.figure(figsize=(6.0, 9.0))
    xi = np.arange(0, 10, 2) + np.random.random(5) * 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    x = np.linspace(0, 10, 51)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.subplot(4, 2, (1, 4))
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k', xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('original')
    plt.xticks([], [])
    for figure, kind in enumerate(('zero', 'linear', 'quadratic', 'cubic')):
        plt.subplot(4, 2, figure + 5)
        plt.axhline(0.0, color='k', lw=0.5)
        spline = interp1d(xi, yi, kind=kind)
        for i in range(4):
            x = np.linspace(xi[i], xi[i + 1], 51)
            plt.plot(x, spline(x), ':')
        if figure > 1:
            plt.xlabel('$x$')
        else:
            plt.xticks([], [])
        if figure % 2 == 0: plt.ylabel('$y$')
        plt.plot(xi, yi, 'ok')
        plt.title(kind + ' spline')
    return 'Types of splines'

def figure2():
    xi = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0])
    yi = np.array([1.0, 1.0, 2.0, -1.0, 1.0, 1.0])
    spline = Akima1DInterpolator(xi, yi)
    x = np.linspace(-5.5, 5.5, 111)
    y = spline(x)
    plt.axhline(0., color='k', lw=.5); plt.axvline(0., color='k', lw=.5)
    plt.plot(x, y, '-')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Akima1DInterpolator')

def figure3():
    xi = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0])
    yi = np.array([1.0, 1.0, 2.0, -1.0, 1.0, 1.0])
    x = np.linspace(-5.5, 5.5, 111)
    plt.axhline(0., color='k', lw=.5); plt.axvline(0., color='k', lw=.5)
    for bc_type in ('not-a-knot', 'periodic', 'clamped', 'natural'):
        spline = CubicSpline(xi, yi, bc_type=bc_type)
        y = spline(x)
        plt.plot(x, y, '-', label=bc_type)
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('CubicSpline')
    plt.legend()

def figure4():
    xi = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0])
    yi = np.array([1.0, 1.0, 2.0, -1.0, 1.0, 1.0])
    spline = PchipInterpolator(xi, yi)
    x = np.linspace(-5.5, 5.5, 111)
    y = spline(x)
    plt.axhline(0., color='k', lw=.5); plt.axvline(0., color='k', lw=.5)
    plt.plot(x, y, '-')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('PchipInterpolator')

def figure5():
    xi = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0])
    yi = np.array([1.0, 1.0, 2.0, -1.0, 1.0, 1.0])
    dyi = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    spline = CubicHermiteSpline(xi, yi, dyi)
    x = np.linspace(-5.5, 5.5, 111)
    y = spline(x)
    plt.axhline(0., color='k', lw=.5); plt.axvline(0., color='k', lw=.5)
    plt.plot(x, y, '-')
    plt.plot(xi, yi, 'ok')
    plt.plot(xi[np.newaxis, :] + np.array([[-.25], [.25]]), yi[np.newaxis, :] + np.array([[-.25], [.25]]), '-k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('CubicHermiteSpline')

def figure6():
    xi = np.arange(0, 10, 2) + np.random.random(5) * 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    spline = interp1d(xi, yi, kind='nearest', fill_value='extrapolate')
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k')
    plt.plot(x, spline(x), '-')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Nearest-neighbor spline')

def figure7():
    from scipy.interpolate import interp1d
    xi, yi = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.5, 2.0, 0.5])
    spline = interp1d(xi, yi, kind='nearest', fill_value='extrapolate')
    x = np.linspace(-0.2, 3.2, 103)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, spline(x), ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True); plt.ylim(-3.0, 3.0)

def figure8():
    xi = np.arange(0, 10, 2) + np.random.random(5) * 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    spline = interp1d(xi, yi, kind='linear', fill_value='extrapolate')
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k')
    plt.plot(x, spline(x), '-')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Linear spline')

def figure9():
    xi = np.array([3.0, 7.0])
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k')
    plt.plot([0.0, 10.0], [yi[0], yi[0]], '-', label='$q_i(x)$')
    plt.plot([0.0, 10.0], [yi[1], yi[1]], '-', label='$q_{i+1}(x)$')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Weighted averaging')
    plt.legend()

def figure10():
    xi = np.array([3.0, 7.0])
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.figure(figsize=(6.0, 6.0))
    plt.subplot(3, 1, (1, 2))
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k')
    plt.plot([0.0, 10.0], [yi[0], yi[0]], '-', label='$q_i(x)$')
    plt.plot([0.0, 10.0], [yi[1], yi[1]], '-', label='$q_{i+1}(x)$')
    plt.plot(xi, yi, 'ok')
    plt.ylabel('$y$')
    plt.title('Weighted averaging')
    plt.legend(); plt.xticks([], [])
    plt.subplot(3, 1, 3)
    plt.plot([0.0, xi[0], xi[1], 10.0], [1.0, 1.0, 0.0, 0.0], '-', label='$w_i(x)$')
    plt.plot([0.0, xi[0], xi[1], 10.0], [0.0, 0.0, 1.0, 1.0], '-', label='$1-w_i(x)$')
    plt.xlabel('$x$'); plt.ylabel('$w$')
    plt.legend()
    return 'Weighted averaging'

def figure11():
    xi = np.array([3.0, 7.0])
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    spline = interp1d(xi, yi, kind='linear')
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.figure(figsize=(6.0, 6.0))
    plt.subplot(3, 1, (1, 2))
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k')
    plt.plot([0.0, 10.0], [yi[0], yi[0]], '-', label='$q_i(x)$')
    plt.plot([0.0, 10.0], [yi[1], yi[1]], '-', label='$q_{i+1}(x)$')
    plt.plot(xi, yi, '--', label='$p_i(x)$')
    plt.plot(xi, yi, 'ok')
    plt.ylabel('$y$')
    plt.title('Weighted averaging')
    plt.legend(); plt.xticks([], [])
    plt.subplot(3, 1, 3)
    plt.plot([0.0, xi[0], xi[1], 10.0], [1.0, 1.0, 0.0, 0.0], '-', label='$w_i(x)$')
    plt.plot([0.0, xi[0], xi[1], 10.0], [0.0, 0.0, 1.0, 1.0], '-', label='$w_{i+1}(x)$')
    plt.xlabel('$x$'); plt.ylabel('$w$')
    plt.legend()
    return 'Weighted averaging'
    
def figure12():
    xi, yi = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.5, 2.0, 0.5])
    spline = interp1d(xi, yi, kind='linear', fill_value='extrapolate')
    x = np.linspace(-0.2, 3.2, 103)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, spline(x), ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True); plt.ylim(-3.0, 3.0)
    
def figure13():
    xi = np.arange(1, 9, 2) + np.random.random(4) * 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    p1 = lagrange(xi[:-1], yi[:-1])(x)
    p2 = lagrange(xi[1:], yi[1:])(x)
    plt.figure(figsize=(6.0, 6.0))
    plt.subplot(3, 1, (1, 2))
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k')
    plt.plot(x, p1, '-', label='$q_i(x)$')
    plt.plot(x, p2, '-', label='$q_{i+1}(x)$')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Weighted averaging')
    plt.legend(); plt.xticks([], []); plt.ylim((-0.3, 3.0))
    plt.subplot(3, 1, 3)
    plt.plot([0.0, xi[1], xi[2], 10.0], [1.0, 1.0, 0.0, 0.0], '-', label='$w_i(x)$')
    plt.plot([0.0, xi[1], xi[2], 10.0], [0.0, 0.0, 1.0, 1.0], '-', label='$1-w_i(x)$')
    plt.xlabel('$x$'); plt.ylabel('$w$')
    plt.legend()
    return 'Weighted averaging'

def figure14():
    xi = np.arange(1, 9, 2) + np.random.random(4) * 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    p1 = lagrange(xi[:-1], yi[:-1])(x)
    p2 = lagrange(xi[1:], yi[1:])(x)
    w = np.clip((xi[2]-x)/(xi[2]-xi[1]), 0.0, 1.0)
    subset = np.logical_and(x >= xi[1], x <= xi[2])
    plt.figure(figsize=(6.0, 6.0))
    plt.subplot(3, 1, (1, 2))
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k')
    plt.plot(x, p1, '-', label='$q_i(x)$')
    plt.plot(x, p2, '-', label='$q_{i+1}(x)$')
    plt.plot(x[subset], (p1 * w + p2 * (1.0 - w))[subset], '--', label='$p_i(x)$')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Weighted averaging')
    plt.legend(); plt.xticks([], []); plt.ylim((-0.2, 2.8))
    plt.subplot(3, 1, 3)
    plt.plot(x, w, '-', label='$w_i(x)$')
    plt.plot(x, 1.0 - w, '-', label='$1-w_i(x)$')
    plt.xlabel('$x$'); plt.ylabel('$w$')
    plt.legend()
    return 'Weighted averaging'

def figure15():
    from scipy.interpolate import interp1d
    xi, yi = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.5, 2.0, 0.5])
    spline = interp1d(xi, yi, kind='cubic', fill_value='extrapolate')
    x = np.linspace(-0.2, 3.2, 103)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, spline(x), ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True); plt.ylim(-3.0, 3.0)
    
    
def figure(fignum=0):
    fignum = int(fignum)
    caption = eval(f'figure{fignum}()')
    if caption is None:
        caption = plt.gca().get_title()
    plt.show()
    print(f'Figure {fignum}: {caption}')

    
if __name__ == '__main__':
    figure()