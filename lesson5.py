'''This module provides the figures for the accompanying Jupyter notebook'''

import matplotlib.pyplot as plt, numpy as np, pandas as pd
from IPython.display import display
from scipy.fft import dct, idct
from scipy.signal.windows import boxcar, triang, cosine, hann, tukey, exponential, gaussian, chebwin, flattop

def figure1():
    plt.figure(figsize=(6.0, 6.0))
    x = np.arange(200) / 25.0
    y = np.sin(2.0 * np.pi * x) + 0.5 * np.random.randn(200)
    a = np.abs(np.fft.rfft(y, norm='forward'))
    f = np.fft.rfftfreq(200, 0.04)
    a[1:-1] *= 2.0
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, '.-')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f, a, basefmt='None')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    return 'Signal & Spectrum'

figure1()


def figure2():
    plt.figure(figsize=(6.0, 6.0))
    x = np.arange(200) / 25.0
    y = np.sin(2.0 * np.pi * x) + 0.5 * np.random.randn(200)
    a = np.fft.rfft(y, norm='forward')
    f = np.fft.rfftfreq(200, 0.04)
    t = np.zeros(f.size)
    t[8] = 1.0
    a = a * t
    y = np.fft.irfft(a, norm='forward')
    a[1:-1] *= 2.0
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f, np.abs(a), basefmt='None')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, '.-')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    return 'Spectrum & Signal'

def figure3():
    plt.figure(figsize=(6.0, 6.0))
    x1 = np.arange(20) / 20.0
    y1 = 2.0 * x1 % 1.0 + np.random.randn(20) / 20.0
    x2 = np.arange(60) / 60.0
    y2 = 2.0 * x2 % 1.0 + np.random.randn(60) / 20.0
    f1 = np.fft.rfftfreq(20, 1.0 / 20.0)
    a1 = np.abs(np.fft.rfft(y1, norm='forward'))
    a1[1:-1] *= 2.0
    f2 = np.fft.rfftfreq(60, 1.0 / 60.0)
    a2 = np.abs(np.fft.rfft(y2, norm='forward'))
    a2[1:-1] *= 2.0
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x1, y1, 'o:')
    plt.plot(x2, y2, '.:')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f2 + .05, a2, linefmt='-C1', markerfmt='oC1', basefmt='None')
    plt.stem(f1 - .05, a1, linefmt='-C0', markerfmt='oC0', basefmt='None')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    return 'Signals & Spectra'

def figure4():
    plt.figure(figsize=(6.0, 6.0))
    x1 = np.arange(20) / 20.0
    y1 = 2.0 * x1 % 1.0 + np.random.randn(20) / 20.0
    x2 = np.arange(60) / 20.0
    y2 = 2.0 * x2 % 1.0 + np.random.randn(60) / 20.0
    f1 = np.fft.rfftfreq(20, 1.0 / 20.0)
    a1 = np.abs(np.fft.rfft(y1, norm='forward'))
    a1[1:-1] *= 2.0
    f2 = np.fft.rfftfreq(60, 1.0 / 20.0)
    a2 = np.abs(np.fft.rfft(y2, norm='forward'))
    a2[1:-1] *= 2.0
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x1, y1, 'o:')
    plt.plot(x2, y2, '.:')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f2 + .02, a2, linefmt='-C1', markerfmt='oC1', basefmt='None')
    plt.stem(f1 - .02, a1, linefmt='-C0', markerfmt='oC0', basefmt='None')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    return 'Signals & Spectra'

def figure5():
    plt.figure(figsize=(6.0, 6.0))
    x1 = np.arange(20) / 20.0
    y1 = 2.0 * x1 % 1.0 + np.random.randn(20) / 20.0
    f1 = np.fft.rfftfreq(20, 1.0 / 20.0)
    a1 = np.fft.rfft(y1, norm='forward')
    a1[1:-1] *= 2.0
    f2 = np.fft.rfftfreq(60, 1.0 / 60.0)
    a2 = np.concatenate((a1, np.zeros(20)))
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f2 + .05, np.abs(a2), linefmt='-C1', markerfmt='oC1', basefmt='None')
    plt.stem(f1 - .05, np.abs(a1), linefmt='-C0', markerfmt='oC0', basefmt='None')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    a2[1:-1] /= 2.0
    x2 = np.arange(60) / 60.0
    y2 = np.fft.irfft(a2, norm='forward')
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x1, y1, 'o:')
    plt.plot(x2, y2, '.:')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    return 'FFT interpolation'

def figure6():
    plt.figure(figsize=(6.0, 6.0))
    x1 = np.arange(20) / 20.0
    y1 = 2.0 * x1 % 1.0 + np.random.randn(20) / 20.0
    f1 = np.fft.rfftfreq(38, 1.0 / 20.0)
    a1 = dct(y1, type=1, norm='forward')
    a1[1:-1] *= 2.0
    f2 = np.fft.rfftfreq(114, 1.0 / 60.0)
    a2 = np.concatenate((a1, np.zeros(38)))
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f2 + .05, np.abs(a2), linefmt='-C1', markerfmt='oC1', basefmt='None')
    plt.stem(f1 - .05, np.abs(a1), linefmt='-C0', markerfmt='oC0', basefmt='None')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    a2[1:-1] /= 2.0
    x2 = np.arange(58) / 60.0
    y2 = idct(a2, type=1, norm='forward')
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x1, y1, 'o:')
    plt.plot(x2, y2, '.:')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    return 'DCT interpolation'

def figure7():
    plt.figure(figsize=(6.0, 6.0))
    x = (np.arange(200) - 99.5) / 25.0
    y = 3.0 * np.sin(np.pi * x**2) * (np.abs(x) < 2.0) + 0.1 * np.random.randn(200)
    a = np.fft.rfft(y, norm='forward')
    f = np.fft.rfftfreq(200, 0.04)
    a[1:-1] *= 2.0
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y, '.r')
    plt.plot(x, y, '-k')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f, np.abs(a), basefmt='None')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    return 'Signal & Spectrum'

def figure8():
    plt.figure(figsize=(6.0, 6.0))
    x = (np.arange(200) - 99.5) / 25.0
    y1 = 3.0 * np.sin(np.pi * x**2) * (np.abs(x) < 2.0) + 0.1 * np.random.randn(200)
    a = np.fft.rfft(y1, norm='forward')
    f = np.fft.rfftfreq(200, 0.04)
    t = np.zeros(a.size)
    t[:17] = 1.0
    a *= t
    y2 = np.fft.irfft(a, norm='forward')
    a[1:-1] *= 2.0
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y1, '.r', label='original')
    plt.plot(x, y2, '-k', label='smoothed')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f, np.abs(a), basefmt='None')
    plt.plot(f, t, '-C1', label='$T(f)$')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    plt.legend()
    return 'Signal & Spectrum'

def figure9():
    plt.figure(figsize=(6.0, 6.0))
    x = (np.arange(200) - 99.5) / 25.0
    y1 = 3.0 * np.sin(np.pi * x**2) * (np.abs(x) < 2.0) + 0.1 * np.random.randn(200)
    a = np.fft.rfft(y1, norm='forward')
    f = np.fft.rfftfreq(200, 0.04)
    t = np.zeros(a.size)
    t[:17] = 1.0
    t[12:29] = np.linspace(1.0, 0.0, 17)
    a *= t
    y2 = np.fft.irfft(a, norm='forward')
    a[1:-1] *= 2.0
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y1, '.r', label='original')
    plt.plot(x, y2, '-k', label='smoothed')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f, np.abs(a), basefmt='None')
    plt.plot(f, t, '-C1', label='$T(f)$')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    plt.legend()
    return 'Signal & Spectrum'

def figure10():
    plt.figure(figsize=(6.0, 9.0))
    x = np.linspace(-2.0, 2.0, 81)
    for fig, window in enumerate((boxcar, triang, cosine, hann, tukey, exponential, gaussian, chebwin, flattop)):
        if window == exponential:
            y = np.concatenate((np.zeros(20), window(41, tau=5.0), np.zeros(20)))
        elif window == gaussian:
            y = np.concatenate((np.zeros(20), window(41, std=10), np.zeros(20)))
        elif window == chebwin:
            y = np.concatenate((np.zeros(20), window(41, at=128), np.zeros(20)))
        else:
            y = np.concatenate((np.zeros(20), window(41), np.zeros(20)))
        plt.subplot(3, 3, fig + 1)
        plt.axhline(0.0, color='k', lw=0.5); plt.axvline(0.0, color='k', lw=0.5)
        plt.plot(x, y, '-')
        if fig % 2 == 0: plt.ylabel('$y$')
        plt.title(window.__name__)
    return 'Symmetrical window shapes'

def figure11():
    plt.figure(figsize=(6.0, 6.0))
    x = (np.arange(200) - 99.5) / 25.0
    y1 = 3.0 * np.sin(np.pi * x**2) * (np.abs(x) < 2.0) + 0.1 * np.random.randn(200)
    a = np.fft.rfft(y1, norm='forward')
    f = np.fft.rfftfreq(200, 0.04)
    t = np.zeros(a.size)
    t[:32] = tukey(63)[31:]
    a *= t
    y2 = np.fft.irfft(a, norm='forward')
    a[1:-1] *= 2.0
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.plot(x, y1, '.r', label='original')
    plt.plot(x, y2, '-k', label='smoothed')
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.axhline(0.0, color='k', lw=0.5)
    plt.stem(f, np.abs(a), basefmt='None')
    plt.plot(f, t, '-C1', label='$T(f)$')
    plt.xlabel('$f$'); plt.ylabel('$A$')
    plt.legend()
    return 'Signal & Spectrum'


def figure(fignum=0):
    fignum = int(fignum)
    caption = eval(f'figure{fignum}()')
    if caption is None:
        caption = plt.gca().get_title()
    plt.show()
    print(f'Figure {fignum}: {caption}')


if __name__ == '__main__':
    figure()
