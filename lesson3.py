'''This module provides the figures for the accompanying Jupyter notebook'''

import matplotlib.pyplot as plt, numpy as np, pandas as pd
from IPython.display import display
from scipy.interpolate import lagrange, interp1d
from scipy.stats import t
from scipy.signal import savgol_filter

def figure1():
    xi = np.arange(0, 10) + np.random.random(10)
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    polynomial = lagrange(xi, yi)
    spline = interp1d(xi, yi, kind='cubic', fill_value='extrapolate')
    curvefit = np.poly1d(np.polyfit(xi, yi, 5))
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k', label='original')
    plt.plot(x, polynomial(x), '--b', label='polynomial')
    plt.plot(x, spline(x), '--r', label='spline')
    plt.plot(x, curvefit(x), '-g', label='curve fit')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Interpolation versus Curve fitting')
    plt.legend(); plt.ylim((-0.2, 3.0))
    
def figure2():
    xi = (np.arange(0, 20) + np.random.random(20)) / 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0
    curvefit = np.poly1d(np.polyfit(xi, yi, 5))
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k', label='original')
    plt.plot(x, curvefit(x), '-g', label='curve fit')
    plt.plot([xi[0], xi[0]], [yi[0], curvefit(xi[0])], '-r', label='residual')
    plt.plot(np.array([xi, xi]), np.array([yi, curvefit(xi)]), '-r')
    plt.plot(xi, curvefit(xi), '.g')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Residuals')
    plt.legend(); plt.ylim((-0.2, 3.0))
    
def figure3():
    xi, yi = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]), np.array([2.0, 2.0, -1.0, -2.0, -1.0])
    curvefit = np.poly1d(np.polyfit(xi, yi, deg=1))
    x = np.linspace(-1.5, 3.5, 101)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, curvefit(x), ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True)
        
def figure4():
    xi, yi = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]), np.array([-4.0, -1.0, 3.0, 1.0, 7.0, 7.0, 8.0])
    curvefit = np.poly1d(np.polyfit(xi, yi, deg=1))
    x = np.linspace(-3.5, 3.5, 141)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, curvefit(x), ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Exercise 6')
    plt.grid(True)
        
def figure5():
    xi = (np.arange(0, 20) + np.random.random(20)) / 2.0
    yi1 = 1.5 + np.cos(xi)
    yi2 = yi1 + np.random.randn(20) / 4.0
    x = np.linspace(0, 10, 201)
    y = 1.5 + np.cos(x)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k', label='original')
    plt.plot([xi[0], xi[0]], [yi1[0], yi2[0]], '-r', label='residual')
    plt.plot(np.array([xi, xi]), np.array([yi1, yi2]), '-r')
    plt.plot(xi, yi2, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('IID residuals')
    plt.legend(); plt.ylim((-0.2, 3.0))
    
def figure6():
    xi, yi = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]), np.array([2.0, 2.0, -1.0, -2.0, -1.0])
    design = np.vstack((xi, np.ones_like(xi))).T
    coef = np.linalg.solve(design.T @ design, design.T @ yi)
    sigma2 = np.sum((yi - design @ coef) ** 2) / (design.shape[0] - design.shape[1])
    cov = sigma2 * np.linalg.inv(design.T @ design)
    x = np.linspace(-1.5, 3.5, 101)
    X = np.vstack((x, np.ones_like(x))).T
    y = X @ coef
    band = np.sqrt(np.sum((X @ cov) * X, axis=1))
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.fill_between(x, y + t.ppf(0.025, 3) * band, y + t.ppf(0.975, 3) * band, color='#cccccc')
    plt.plot(x, y, ':k')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True)
    
def figure7():
    xi, yi = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]), np.array([-4.0, -1.0, 3.0, 1.0, 7.0, 7.0, 8.0])
    design = np.vstack((xi, np.ones_like(xi))).T
    coef = np.linalg.solve(design.T @ design, design.T @ yi)
    sigma2 = np.sum((yi - design @ coef) ** 2) / (design.shape[0] - design.shape[1])
    cov = sigma2 * np.linalg.inv(design.T @ design)
    x = np.linspace(-3.5, 3.5, 141)
    X = np.vstack((x, np.ones_like(x))).T
    y = X @ coef
    band = np.sqrt(np.sum((X @ cov) * X, axis=1))
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.fill_between(x, y + t.ppf(0.025, 5) * band, y + t.ppf(0.975, 5) * band, color='#cccccc')
    plt.plot(x, y, ':k')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Exercise 10')
    plt.grid(True)
    
def figure8():
    xi = (np.arange(0, 20) + np.random.random(20)) / 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0 + np.random.randn(20) / 4.0
    plt.axhline(0.0, color='k', lw=0.5)
    plt.fill_between([-1, 2], [-1, -1], [4, 4], color='#999999')
    plt.fill_between([6, 11], [-1, -1], [4, 4], color='#999999')
    x = np.linspace(0.0, 10.0, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.plot(x, y, ':k')
    plt.plot(xi, yi, '.k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Windowing')
    plt.axis((-.5, 10.5, -0.2, 3.0))

def figure9():
    xi = (np.arange(0, 20) + np.random.random(20)) / 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0 + np.random.randn(20) / 4.0
    subset = np.abs(xi - 4.0) <= 2.0
    curvefit = np.poly1d(np.polyfit(xi[subset], yi[subset], 2))
    plt.axhline(0.0, color='k', lw=0.5)
    plt.fill_between([-1, 2], [-1, -1], [4, 4], color='#999999')
    plt.fill_between([6, 11], [-1, -1], [4, 4], color='#999999')
    x = np.linspace(0.0, 10.0, 201)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    plt.plot(x, y, ':k', label='original')
    plt.plot(xi[~subset], yi[~subset], '.k')
    plt.plot(xi[subset], yi[subset], '.r')
    x = np.linspace(2.0, 6.0, 81)
    y = curvefit(x)
    plt.plot(x, y, '-r', label='curve fit')
    plt.plot(4.0, curvefit(4.0), 'or', label='curve fit')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Local regression')
    plt.legend(); plt.ylim((-0.2, 3.0))

def figure10():
    xi = (np.arange(0, 40) + 0.5) / 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0 + np.random.randn(40) / 4.0
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok', label='original')
    for w in [3, 9]:
        plt.plot(xi, savgol_filter(yi, w, 1), '.:', label=f'w={w}')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title(f'Savitsky-Golay (p=1)')
    plt.legend(); plt.ylim((-0.2, 3.0))

def figure11():
    xi = (np.arange(0, 40) + 0.5) / 2.0
    yi = 1.5 + np.cos(xi) - np.cos(np.pi * xi) / 4.0 + np.random.randn(40) / 4.0
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok', label='original')
    for p in [0, 2]:
        plt.plot(xi, savgol_filter(yi, 7, p), '.:', label=f'p={p}')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title(f'Savitsky-Golay (w=7)')
    plt.legend(); plt.ylim((-0.2, 3.0))

def figure12():
    xi, yi = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]), np.array([2.0, 2.0, -1.0, -2.0, -1.0])
    curvefit = np.poly1d(np.polyfit(xi[1:4], yi[1:4], deg=1))
    x = np.linspace(0.0, 2.0, 41)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, curvefit(x), ':r')
    plt.plot(2/3, curvefit(2/3), 'or')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True)

def figure13():
    xi, yi = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]), np.array([-4.0, -1.0, 3.0, 1.0, 7.0, 7.0, 8.0])
    curvefit = np.poly1d(np.polyfit(xi[3:], yi[3:], deg=1))
    x = np.linspace(0.0, 3.0, 61)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, curvefit(x), ':r')
    plt.plot(2.0, curvefit(2.0), 'or')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True)

    
def figure(fignum=0):
    fignum = int(fignum)
    caption = eval(f'figure{fignum}()')
    if caption is None:
        caption = plt.gca().get_title()
    plt.show()
    print(f'Figure {fignum}: {caption}')

    
if __name__ == '__main__':
    figure()