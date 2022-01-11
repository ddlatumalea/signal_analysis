'''This module provides the figures for the accompanying Jupyter notebook'''

import matplotlib.pyplot as plt, numpy as np, pandas as pd
from IPython.display import display

def figure1():
    x = np.linspace(0.0, 1.0, 101)
    xi = np.linspace(0.0, 1.0, 12, endpoint=False)
    y = np.sin(2.0 * np.pi * x)
    yi = np.sin(2.0 * np.pi * xi)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, ':', color='C0')
    plt.plot(xi, yi, 'o', color='C0')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Sinusoids')

def figure2():
    x = np.linspace(0.0, 1.0, 101)
    xi = np.linspace(0.0, 1.0, 12, endpoint=False)
    y = np.sin(2.0 * np.pi * x)
    z = np.sin(4.0 * np.pi * x) / 2
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y + z, '--', color='k', label='sum')
    plt.plot(x, y, ':', color='C0', label='$f=1$')
    plt.plot(xi, np.sin(2.0 * np.pi * xi), 'o', color='C0')
    plt.plot(x, z, ':', color='C1', label='$f=2$')
    plt.plot(xi, np.sin(4.0 * np.pi * xi) / 2, 'o', color='C1')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Sinusoids')
    plt.legend()
    
def figure3():
    x = np.linspace(0.0, 1.0, 101)
    xi = np.linspace(0.0, 1.0, 12, endpoint=False)
    y = np.sin(2.0 * np.pi * x)
    z = np.sin(4.0 * np.pi * x) / 2
    v = np.sin(6.0 * np.pi * x) / 3
    w = np.sin(8.0 * np.pi * x) / 4
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y + z + v + w, '--', color='k', label='sum')
    plt.plot(x, y, ':', color='C0', label='$f=1$')
    plt.plot(xi, np.sin(2.0 * np.pi * xi), 'o', color='C0')
    plt.plot(x, z, ':', color='C1', label='$f=2$')
    plt.plot(xi, np.sin(4.0 * np.pi * xi) / 2, 'o', color='C1')
    plt.plot(x, v, ':', color='C2', label='$f=3$')
    plt.plot(xi, np.sin(6.0 * np.pi * xi) / 3, 'o', color='C2')
    plt.plot(x, w, ':', color='C3', label='$f=4$')
    plt.plot(xi, np.sin(8.0 * np.pi * xi) / 4, 'o', color='C3')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Sinusoids')
    plt.legend()
    
def figure4():
    x = np.linspace(0.0, 1.0, 101)
    y = sum(np.sin(2.0 * np.pi * f * x) / f for f in range(1, 51))
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, '--', color='k', label='sum')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Sinusoids')
    plt.legend()
    
def figure5():
    x = np.linspace(0.0, 1.0, 101)
    y = 3.0 * np.cos(2.0 * np.pi * x)
    z = -4.0 * np.sin(2.0 * np.pi * x)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, '--', color='C0', label='cosine, $|A| = 3$')
    plt.plot(x, z, '--', color='C1', label='sine, $|A| = 4$')
    plt.plot(x, y + z, '-', color='k', label='sum, $|A| = 5$')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Sum of sinusoids')
    plt.grid(True); plt.legend(); plt.yticks(np.arange(-5, 6))
    
def figure6():
    x = np.linspace(0.0, 1.0, 101)
    xi = np.linspace(0.0, 1.0, 12, endpoint=False)
    plt.figure(figsize=(6.0, 14.0))
    for figure in range(7):
        plt.subplot(12, 1, figure + 1)
        plt.axhline(0.0, color='k', lw=0.5)
        plt.plot(x, np.cos(2.0 * np.pi * figure * x), '-', color='C0', label='cos')
        plt.plot(xi, np.cos(2.0 * np.pi * figure * xi), 'o', color='C0')
        plt.plot(x, np.sin(2.0 * np.pi * figure * x), '-', color='C1', label='sin')
        plt.plot(xi, np.sin(2.0 * np.pi * figure * xi), 'o', color='C1')
        if figure == 6:
            plt.xlabel('$x$')
        else:
            plt.xticks([], [])
        plt.ylabel('$y$')
        if figure == 0: plt.legend()
        plt.ylim([-1.2, 1.2])
    return 'Components f=0 to f=6'

def figure7():
    x = np.linspace(0.0, 1.0, 101)
    xi = np.linspace(0.0, 1.0, 12, endpoint=False)
    plt.figure(figsize=(6.0, 14.0))
    for figure in range(7):
        plt.subplot(12, 1, figure + 1)
        plt.axhline(0.0, color='k', lw=0.5)
        plt.plot(x, np.cos(2.0 * np.pi * (figure + 6) * x), '-', color='C0', label='cos')
        plt.plot(x, np.cos(2.0 * np.pi * (figure - 6) * x), ':', color='C0', label='cos')
        plt.plot(xi, np.cos(2.0 * np.pi * (figure + 6) * xi), 'o', color='C0')
        plt.plot(x, np.sin(2.0 * np.pi * (figure + 6) * x), '-', color='C1', label='sin')
        plt.plot(x, np.sin(2.0 * np.pi * (figure - 6) * x), ':', color='C1', label='cos')
        plt.plot(xi, np.sin(2.0 * np.pi * (figure + 6) * xi), 'o', color='C1')
        if figure == 6:
            plt.xlabel('$x$')
        else:
            plt.xticks([], [])
        plt.ylabel('$y$')
        if figure == 0: plt.legend()
        plt.ylim([-1.2, 1.2])
    return 'Components f=6 to f=12'

def figure8():
    xi, yi = np.array([0.0, 2.0, 4.0, 6.0, 10.0]), np.array([3.0, 3.0, -1.0, -1.0, -5.0])
    x = np.linspace(-0.5, 11.5, 121)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, 4.0 * np.sin(np.pi / 6.0 * (x + 1.0)) - 1.0, ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True)
    
def figure9():
    xi, yi = np.array([0.0, 0.5, 1.0, 1.5, 2.0]), np.array([1.0, 3.0, -1.0, 1.0, 2.0])
    x = np.linspace(-0.5, 2.5, 61)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, 1.0 + np.sqrt(2.0) * np.cos(np.pi * (x - 0.25)), ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True)
    
def figure10():
    C = np.cos(np.pi * np.outer(np.arange(12), np.arange(7)) / 6.0)
    S = np.sin(np.pi * np.outer(np.arange(12), np.arange(1, 6)) / 6.0)
    plt.subplot(1, 2, 1)
    plt.imshow(C, cmap="RdYlBu")
    plt.xlabel('$j$'); plt.ylabel('$i$')
    plt.title('Design matrix $C_{ij}$')
    plt.xticks([]); plt.yticks([])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(S, cmap="RdYlBu")
    plt.xlabel('$j$'); plt.ylabel('$i$')
    plt.title('Design matrix $S_{ij}$')
    plt.xticks([]); plt.yticks([])
    plt.colorbar()
    return 'Design matrix X = [C,S]'

def figure11():
    xi, yi = np.array([0.0, 1.0, 2.0]), np.array([2.5, -2.0, 1.0])
    x = np.linspace(-0.5, 2.5, 61)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, 0.5 + 2.0 * np.cos(2.0 * np.pi / 3.0 * x) - np.sqrt(3.0) * np.sin(2.0 * np.pi / 3.0 * x), ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True)

def figure12():
    xi, yi = np.array([0.0, 0.5, 1.0, 1.5]), np.array([1.0, 3.0, -1.0, 1.0])
    x = np.linspace(-0.25, 1.75, 81)
    plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
    plt.plot(xi, yi, 'ok')
    plt.plot(x, 1.0 + np.sqrt(2.0) * np.cos(np.pi * (x - 0.25)) - np.cos(2.0 * np.pi * x), ':k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Example')
    plt.grid(True)

def figure13():
    xi = np.arange(32)
    yi = np.linspace(-1.0, 1.0, 32)
    zi = np.concatenate((yi[16:], yi[:16]))
    fi = np.fft.rfftfreq(32)
    plt.figure(figsize=(6.0, 6.0))
    for figure, signal in enumerate((yi, zi)):
        spectrum = np.abs(np.fft.rfft(signal, norm='forward'))
        spectrum[1:-1] *= 2.0
        plt.subplot(2, 2, figure + 1)
        plt.axhline(0.0, color='k', lw=0.5)
        plt.plot(xi, signal, ':.')
        if figure % 2 == 0: plt.ylabel('$y$')
        plt.title(f'Signal #{figure+1}')
        plt.subplot(2, 2, figure + 3)
        plt.axhline(0.0, color='k', lw=0.5)
        plt.stem(fi, spectrum, basefmt='None')
        plt.xlabel('$f$')
        if figure % 2 == 0: plt.ylabel('$A$')
        plt.title(f'Spectrum #{figure+1}')

def figure14():
    xi = np.arange(62)
    yi = np.linspace(-1.0, 1.0, 32)
    zi = np.concatenate((yi[16:], yi[:16]))
    yi = np.concatenate((yi, yi[-2:0:-1]))
    zi = np.concatenate((zi, zi[-2:0:-1]))
    fi = np.fft.rfftfreq(62)
    plt.figure(figsize=(6.0, 6.0))
    for figure, signal in enumerate((yi, zi)):
        spectrum = np.abs(np.fft.rfft(signal, norm='forward'))
        spectrum[1:-1] *= 2.0
        plt.subplot(2, 2, figure + 1)
        plt.axhline(0.0, color='k', lw=0.5)
        plt.plot(xi, signal, ':.')
        if figure % 2 == 0: plt.ylabel('$y$')
        plt.title(f'Signal #{figure+1}')
        plt.subplot(2, 2, figure + 3)
        plt.axhline(0.0, color='k', lw=0.5)
        plt.stem(fi, spectrum, basefmt='None')
        plt.xlabel('$f$')
        if figure % 2 == 0: plt.ylabel('$A$')
        plt.title(f'Spectrum #{figure+1}')
        
    
def figure(fignum=0):
    fignum = int(fignum)
    caption = eval(f'figure{fignum}()')
    if caption is None:
        caption = plt.gca().get_title()
    plt.show()
    print(f'Figure {fignum}: {caption}')

    
if __name__ == '__main__':
    figure()