def fourier_transform(yi):
    """a, b = fourier_transform(yi).
    Real-valued Fourier transform that determines the
    coefficients of the Fourier series for a given
    signal y. The coefficients of the cosine terms are
    returned in the array a; those of the sine terms
    in the array b. Frequencies start at zero and do
    not exceed the Nyquist frequency.
    yi     = {y1,y2,...,xn}
    """
    xi = np.arange(yi.size)
    length = yi.size // 2 + 1
    a, b = np.empty(length), np.empty(length)
    # Compute zero and Nyquist frequency cases
    a[0] = np.mean(yi)
    a[-1] = yi @ np.cos(np.pi * xi) / yi.size
    b[0] = 0.0
    b[-1] = 0.0 
    # Compute ordinary cases (overwrite Nyquist if odd length)
    for index in range(1, length + yi.size % 2 - 1):
        arg = 2.0 * np.pi * xi * index / yi.size
        a[index] = 2.0 / yi.size * yi @ np.cos(arg)
        b[index] = 2.0 / yi.size * yi @ np.sin(arg)
    return a, b