from scipy.signal import get_window

def fourier_smooth(yi, d, fmax, shape='boxcar'):
    """y = fourier_smooth(yi, d, fmax, shape='boxcar').
    Smoothing function that low-pass filters a signal
    yi with sampling time d. Spectral components with
    frequencies above a cut-off fmax are blocked, while
    lower frequencies are multiplied with a transfer
    function with a given shape.
    yi     = {y1,y2,...,xn}
    d      = sampling time
    fmax   = low-pass cut-off frequency
    shape  = transfer function shape (default 'boxcar')
    """
    spectrum = np.fft.rfft(yi, norm='forward')
    width = int(fmax * yi.size * d) + 1
    transfer = np.zeros(spectrum.size)
    window = get_window(shape, 2 * width - 1, False)
    transfer[:width] = window[width - 1:]
    spectrum *= transfer
    y = np.fft.irfft(spectrum, yi.size, norm='forward')
    return y
