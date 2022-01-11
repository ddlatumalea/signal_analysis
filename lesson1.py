'''This module provides the figures for the accompanying Jupyter notebook'''

import matplotlib.pyplot as plt, numpy as np, pandas as pd
from IPython.display import display
from scipy.interpolate import lagrange, BarycentricInterpolator, KroghInterpolator
from scipy.special import gamma


def figure1():
    N = 7
    patients = pd.DataFrame({
        'ID': np.arange(N) + 1,
        'Sex': np.random.choice(['♂', '♀'], N),
        'Age': np.random.randint(20, 50, N),
        'Province': np.random.choice(['Groningen', 'Drenthe', 'Fryslân'], N),
        'BMI': np.round(np.random.random(N) * 10.0 + 20.0, 1),
        'Infected': np.random.choice([True, False], N, p=(1/3, 2/3))
    }).set_index('ID')
    display(patients)
    return 'SARS-CoV-2 patient characteristics'

def figure2():
    years = np.arange(1900, 2030, 10)
    sizes = np.array([
        5_104_000, 5_858_000, 6_754_000, 7_825_000, 8_834_000,
        10_026_773, 11_417_254, 12_957_621, 14_091_014, 14_892_574,
        15_863_950, 16_574_989, 17_424_978    
    ]) / 1.0e6
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(years, sizes, 'o:k')
    plt.xlabel('Year'); plt.ylabel('N ($×10^6$)')
    plt.title('Dutch population size by year')

def figure3():
    x = np.linspace(0.0, 10.0, 51)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    subset = np.logical_and(x > 1, x < 7.5)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k', label='original')
    plt.plot(x[subset], y[subset], '-r', label='truncated')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Truncation')
    plt.legend()
    
def figure4():
    x = np.linspace(0.0, 10.0, 51)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    subset = np.logical_and(x > 1, x < 7.5)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k', label='original')
    plt.plot(x[subset], y[subset], '-r', label='truncated')
    plt.plot(x[subset], y[subset], 'xg', label='sampled')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Sampling')
    plt.legend()

def figure5():
    x = np.linspace(0.0, 10.0, 51)
    y = 1.5 + np.cos(x) - np.cos(np.pi * x) / 4.0
    subset = np.logical_and(x > 1, x < 7.5)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':k', label='original')
    plt.plot(x[subset], y[subset], '-r', label='truncated')
    plt.plot(x[subset], y[subset], 'xg', label='sampled')
    plt.plot(x[subset], np.round(y[subset] * 4.0) / 4.0, '+b', label='quantized')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Quantization')
    plt.legend()

def figure6():
    xi = np.arange(5)
    yi = np.array([0, 1, 5, 14, 30])
    x = np.linspace(-0.5, 4.5, 51)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, x * (x + 1.0) * (2.0 * x + 1.0) / 6.0, ':k')
    plt.xlabel('$n$'); plt.ylabel('$P(n)$')
    plt.title('Square pyramidal numbers')
    plt.grid(True)

def figure7():
    xi = np.array([1.0, 2.0, 3.0, 5.0])
    x = np.linspace(0.0, 6.0, 61)
    plt.axhline(0.0, color='k', lw=0.5)
    for i in range(xi.size):
        cardinal = np.ones_like(x)
        for j in range(xi.size):
            if i != j:
                cardinal *= (x - xi[j]) / (xi[i] - xi[j])
        plt.plot(x, cardinal, '-', label=f'$l_{i+1}(x)$')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Cardinal functions')
    plt.grid(True); plt.legend(); plt.ylim(-0.5, 1.5)

def figure8():
    xi, yi = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, -1.0])
    x = np.linspace(-0.2, 2.2, 37)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, (-2.0 * x + 3.0) * x + 1.0, ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True); plt.ylim(-3.0, 3.0)

def figure9():
    phi = 0.5 + 0.5 * np.sqrt(5.0)
    xi = np.arange(5)
    yi = np.array([0, 1, 1, 2, 3])
    x = np.linspace(-0.5, 4.5, 51)
    y = (phi ** x - np.cos(np.pi * x) * (1.0 / phi) ** x) / np.sqrt(5.0)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok', x, y, ':k')
    plt.xlabel('$n$'); plt.ylabel('$F(n)$')
    plt.title('Fibonacci numbers')
    plt.grid(True)

def figure10():
    xi = np.arange(5)
    yi = np.array([1, 1, 2, 6, 24])
    x = np.linspace(-0.5, 4.5, 51)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(x, gamma(x + 1.0), ':k')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$n$'); plt.ylabel('$n!$')
    plt.title('Factorial numbers')
    plt.grid(True)

def figure11():
    xi, yi = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, -1.0])
    x = np.linspace(-0.2, 2.2, 37)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(x, (x - 3.0) / (2.0 * x - 3.0), ':k')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True); plt.ylim(-3, 3)
    
def figure12():
    xi = np.arange(5)
    yi = np.array([1, 1, 2, 5, 14])
    x = np.linspace(-0.5, 4.5, 51)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(x, gamma(2.0 * x + 1.0) / (gamma(x + 1.0) * gamma(x + 2.0)), ':k')
    plt.plot(xi, yi, 'ok')
    plt.xlabel('$n$'); plt.ylabel('$C(n)$')
    plt.title('Catalan numbers')
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