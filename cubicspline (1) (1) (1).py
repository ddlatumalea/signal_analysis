def cubic_interpolate(xi, yi, x):
    """y = cubic_interpolate(xi, yi, x).
    Cubic spline interpolation method that fits a
    piecewise polynomial of degree three through
    data points {xi,yi}, evaluated at arguments x.
    xi     = {x1,x2,...,xn}
    yi     = {y1,y2,...,xn}
    x      = arguments x
    """
    if xi.size != yi.size:
        raise ValueError('xi and yi must have the same length')
        
    xi = xi.astype('float64')
    yi = yi.astype('float64')
        
    order = np.argsort(xi)
    xi, yi = xi[order], yi[order]

    xknots = xi
    idx = np.searchsorted(xknots, x)

    # left parabola
    if idx-2 == -1:
        left_par_xi = xi[idx-1:idx+1]
        left_par_yi = yi[idx-1:idx+1]
    elif idx-2 == -2:
        left_par_xi = xi[idx:idx+1]
        left_par_yi = yi[idx:idx+1]
    else:
        left_par_xi = xi[idx-2:idx+1]
        left_par_yi = yi[idx-2:idx+1]

    left_par_f = lagrange(left_par_xi, left_par_yi)
    left_par_x = left_par_f(x) 

    # right parabola
    right_par_xi = xi[idx-1:idx+2]
    right_par_yi = yi[idx-1:idx+2]

    right_par_f = lagrange(right_par_xi, right_par_yi)
    right_par_x = right_par_f(x) 

    # calculate applicable weight for left and right
#     left_weight = xi[idx] - x 
#     right_weight = x - xi[idx-1] 

    # uniform weight
    y = 0.5 * left_par_x + 0.5 * right_par_x
    
    return y

def poly_curvefit(xi, yi, x, *, deg=1, alpha=None):
    """y, lower, upper = poly_curvefit(xi, yi, x, *, deg=1, alpha=None).
    Polynomial curve fitting method that fits a polynomial
    of given degree through data points {xi,yi}, evaluated
    at arguments x. If a confidence level is provided, the
    lower and upper bounds of the confidence interval for
    all y is also returned.    
    xi     = {x1,x2,...,xn}
    yi     = {y1,y2,...,xn}
    x      = arguments x
    deg    = polynomial degree (default 1)
    alpha  = confidence level (default None)


		EXAMPLE:
		i, j = 0, 0
		fig, axs = plt.subplots(2,2, figsize=(20, 10))

		x = np.linspace(1900, 2050, 201)

		colors = ['blue', 'orange', 'green', 'red']

		for k in range(4):
				y, lowerbound, upperbound = poly_curvefit(xi, yi, x, deg=k, alpha=0.05)
				
				axs[i][j].plot(xi, yi, 'o:k')
				axs[i][j].plot(x, y, '-', color=colors[k])
				axs[i][j].fill_between(x, lowerbound, upperbound, color=colors[k], alpha=0.25)
				
				axs[i][j].set_ylabel('Size')
				axs[i][j].set_xlabel('Year')
				axs[i][j].set_title('Polynomial degree {}'.format(k))
				j+=1

				if j==2:
						i+=1
						j=0
						
		plt.show()
    """
    if xi.size != yi.size:
        raise ValueError('xi and yi must have the same length')
    if deg >= xi.size:
        raise ValueError('polynomial degree too high for available data')
    if alpha is not None and not 0.0 < alpha < 1.0:
        raise ValueError('confidence level must lie between 0.0 and 1.0')
    
    order = deg + 1
    
    xi = xi + 0.0
    yi = yi + 0.0
    
    X = np.vander(xi, order)
        
    M = X.T @ X
    v = X.T @ yi

    coef = np.linalg.solve(M, v)
    
    y = np.polyval(coef, x)
    
    if alpha is None:
        return y
    
    dof = xi.size - deg - 1
    variance = np.sum((yi - np.polyval(coef, xi)) ** 2) / dof
    covariance = variance * np.linalg.inv(X.T @ X)
    
    vander = np.vander(x, order)
    error = np.sqrt(np.sum(vander @ covariance * vander, axis=1))
    tvalue = -t.ppf(alpha / 2.0, dof)
    
    return y, (y - tvalue * error), (y + tvalue * error)

def savgol_curvefit(xi, yi, x, *, deg=1, window=5):
	"""y = savgol_curvefit(xi, yi, x, *, deg, window).
	Savitsky-Golay method that fits local polynomials
	of given degree through data points {xi,yi} that
	fall in a window centered on x, evaluated
	at arguments x. The window size is specified either
	as a float indicating the half-window size, or as
	an int indicating the number of data points.
	xi     = {x1,x2,...,xn}
	yi     = {y1,y2,...,xn}
	x      = arguments x
	deg    = polynomial degree (default 1)
	window = window size as float/int (default 5)
	"""
	if xi.size != yi.size:
			raise ValueError('xi and yi must have the same length')
			
	if not isinstance(x, (list, np.ndarray)):
			raise ValueError('x must be a numpy array!')
			
	res = np.empty(x.size)
	
	xi = xi.astype('float64')
	yi = yi.astype('float64')

	order = deg + 1
	
	for n, val in enumerate(x):
			# xi = [1, 2, 3, 4, 5, 6]
			# x = 3
			# np.abs(xi - x) -> [2, 1, 0, 1, 2, 3]
			distance = np.abs(xi - val)
			collection = [(i, distance[i]) for i in range(xi.size)] # [(0, 2), (1, 1), (2, 0), (3, 1), (4, 2), (5, 3)]
			collection = sorted(collection, key=lambda x: x[1]) # [(2, 0), (1, 1), (3, 1), (0, 2), (4, 2), (5, 3)]
	
			# nearest neighbours
			if isinstance(window, int):
					x_neighbours_idx = sorted([k for k, v in collection[:window]])
			else:
					# half window
					x_neighbours_idx = sorted([k for k, v in collection if v < window])

			x_neighbours = [xi[key] for key in x_neighbours_idx]
			y_neighbours = [yi[key] for key in x_neighbours_idx]
	
			X = np.vander(x_neighbours, order)
			M = X.T @ X
			v = X.T @ y_neighbours

			a = np.linalg.solve(M, v)

			res[n] = np.polyval(a, val)
	
	return res

# plot all the nice thingies
x = np.linspace(0.0, 1.0, 250)

# b = 1/fi square
y = sum((1/f) * np.sin(2.0 * np.pi * f * x) for f in range(1, 51) if f % 2 == 1)

# plt.axhline(0.0, color='k', lw=0.5)
# plt.plot(x, y, '-r', label='square')
# plt.xlabel('x'); plt.ylabel('y(x)')
# plt.title('square wave')
# plt.legend()
# plt.show()

# a = 1/fi^2
y_tri = sum((1/(f*f)) * np.cos(2.0 * np.pi * f * x) for f in range(1, 51) if f % 2 == 1)

plt.axhline(0.0, color='k', lw=0.5)
plt.plot(x, y, '-r', label='1/f')
plt.plot(x, y_tri, '-b', label='1/f^2')
plt.xlabel('x'); plt.ylabel('y(x)')
plt.title('fourier series odds')
plt.legend()
plt.show()

# b = (-1)^i / fi
y_saw = sum((((-1)**i)/f) * np.sin(2.0 * np.pi * f * x) for i, f in enumerate(range(1, 51)))

# b = f / (f^2 - 1/4)

y_half_cos = sum((f/((f**2)-0.25)) * np.sin(2.0 * np.pi * f * x) for f in range(1, 51))


plt.axhline(0.0, color='k', lw=0.5)
plt.plot(x, y_half_cos, '-r', label='b = f / (f^2 - 1/4)')
plt.plot(x, y_saw, '-b', label='b = (-1)^i / fi')

plt.xlabel('$x$'); plt.ylabel('$y$')
plt.title('fourier series all positive i')
plt.legend()
plt.show()

def trig_curvefit(xi, yi, x, period):
    """y = trig_curvefit(xi, yi, x, period).
    Trigonometric curve fitting method that fits a sinusoid
    with a given period through data points {xi,yi},
    evaluated at arguments x.
    xi     = {x1,x2,...,xn}
    yi     = {y1,y2,...,xn}
    x      = arguments x
    period = sinusoidal period
    """
    if xi.size != yi.size:
        raise ValueError('xi and yi must have the same length')

    f = 1/period

    xi = xi.astype('float64')
    yi = yi.astype('float64')

    X = np.vstack((np.ones(xi.size), np.cos(2 * np.pi * f * xi), np.sin(2 * np.pi * f * xi))).T

    M = X.T @ X
    v = X.T @ yi

    a = np.linalg.solve(M, v)

    y = a[0] + a[1] * np.cos(2 * np.pi * f * x) + a[2] * np.sin(2 * np.pi * f * x)
        
    return y

#### 
    def trig_curvefit(xi, yi, x, period):
    """y = trig_curvefit(xi, yi, x, period).
    Trigonometric curve fitting method that fits a sinusoid
    with a given period through data points {xi,yi},
    evaluated at arguments x.
    xi     = {x1,x2,...,xn}
    yi     = {y1,y2,...,xn}
    x      = arguments x
    period = sinusoidal period
    """
    if xi.size != yi.size:
        raise ValueError('xi and yi must have the same length')
    xi = xi.astype('float64')
    yi = yi.astype('float64')
    frequency = 1/period
    design_matrix = np.vstack((np.ones(xi.size),
                               np.cos(2 * np.pi * frequency * xi ),
                               np.sin(2 * np.pi * frequency * xi))).T
    M = design_matrix.T @ design_matrix
    v = design_matrix.T @ yi
    a = np.linalg.solve(M, v)
    a0, a1, a2 = a[0], a[1], a[2]
    if not type(x) == int or not type(x) == int:
        y = []
        print('this is ran')
        for val in x:
            y.append(a0 + a1 * np.cos(2 * np.pi * frequency * val ) + a2 * np.sin(2 * np.pi * frequency * val))
        return y
    y = a0 + a1 * np.cos(2 * np.pi * frequency * x ) + a2 * np.sin(2 * np.pi * frequency * x)
    return y




################ EX 5

y = sun['Sunshine [%]']
x = pd.to_datetime(sun.Day, format='%Y%m%d')
y = np.asarray(y)

length = 4 # years
samples = len(y) # odd number
sampling_period = length / samples

coef = np.fft.rfft(y, norm='forward')
coef[1::] = 2.0 * np.conj(coef[1::])
a, b = np.real(coef), np.imag(coef)

amplitudes = np.sqrt(a * a + b * b)
frequencies = np.fft.rfftfreq(y.size, sampling_period)

f, A, = obtain_amplitude_values(y, sampling_period = sampling_period)
plot_amplitude_spectrum(frequencies, amplitudes, name = None, xlim = None, ylim = None)

a *= (f % 1 == 0)
b *= (f % 1 == 0)

a *= f < 6
b *= f < 6


coef_clean = a + 1j * b
coef_clean[1::] = 0.5 * np.conj(coef_clean[1::])

y_cleaned = np.fft.irfft(coef_clean, norm='forward')
x = x[0:-1]
y = y[0:-1]
left, right = min(x), max(x)
# plt.figure(figsize=(12.5, 5))
plt.axhline(0.0, color='k', lw=0.5)
plt.axhline(100.0, color='k', lw=0.5)
# plt.scatter(x=x, y=y, label = 'original', alpha = 0.3)
plt.plot(x, y, '.', label = 'original', alpha = 0.3)
plt.plot(x, y_cleaned, '-r')
plt.title('sunshine in Eelde'); plt.xlabel('Years'); plt.ylabel('Sunshine [%]')
plt.xlim(left, right); plt.legend(); plt.show()