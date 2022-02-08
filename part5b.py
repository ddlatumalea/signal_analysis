def fourier_resample(yi, n):
    """y = fourier_resample(yi, n).
    Fourier interpolation method that resamples a given
    signal yi to n new points. The first element of the
    returned signal y coincides with the first element
    of the provided signal yi.
    yi     = {y1,y2,...,xn}
    n      = desired number of points
    """
    coef = np.fft.rfft(yi, norm='forward')
    if yi.size % 2 == 0:
        coef[1:-1] *= 2.0
    else:
        coef[1:] *= 2.0
    if n > yi.size:
        coef = np.concatenate((coef, np.zeros(n - yi.size)))
    else:
        coef = coef[:n // 2 + 1]
    if n % 2 == 0:
        coef[1:-1] /= 2.0
    else:
        coef[1:] /= 2.0
    y = np.fft.irfft(coef, n, norm='forward')
    return y
