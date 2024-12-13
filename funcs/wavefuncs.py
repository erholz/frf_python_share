import numpy as np


####################################  WAVENUMBER  ####################################
def wavenumber(h, T, g):
    """
    :param: h (water depth [m]), T (period [s]), g (gravity [m/s^2])
    :return: hmatch, Tmatch, k, L, C, Cg, n
    """

    import numpy as np
    import sys

    # First, assess input parameters; if int, convert to array
    int_flag = False
    if (isinstance(h, int)) and (isinstance(T, int)):
        h = np.array([h])
        T = np.array([T])
        int_flag = True
    elif (isinstance(T, int)) or (isinstance(T, float) and T.shape == ()):
        T = np.array([T])
    elif (isinstance(h, int)) or (isinstance(h, float) and h.shape == ()):
        h = np.array([h])

    # now, create matching vectors for h and T
    if h.shape == T.shape:  # h and T are the same size, point-wise calcs
        k = np.zeros(shape=h.shape)
        kh = np.zeros(shape=h.shape)
        numcalcs = h.size
        hmatch = h
        Tmatch = T
        if (h.size == 1) and (T.size == 1):
            hmatch = np.ones((2,)) * h
            Tmatch = np.ones((2,)) * T
    elif (h.shape != T.shape) and not ((h.size == 1) or (T.size == 1)):
        print('ERROR! If h and T both sized > 1, then they must be identically shaped. ')
        Tmatch = []
        sys.exit()
    elif (h.shape != T.shape) and (h.size == 1):
        k = np.zeros(shape=T.shape)
        kh = np.zeros(shape=T.shape)
        numcalcs = T.size
        hmatch = np.ones(shape=T.shape) * h
        Tmatch = T
    elif (h.shape != T.shape) and (T.size == 1):
        k = np.zeros(shape=h.shape)
        kh = np.zeros(shape=h.shape)
        numcalcs = h.size
        Tmatch = np.ones(shape=h.shape) * T
        hmatch = h
    else:
        print('ERROR in wavenumber')
        Tmatch = []
        sys.exit()

    # ok, now calculate k iteratively
    sigma = 2. * np.pi / Tmatch
    k[k == 0] = np.nan
    kh[kh == 0] = np.nan
    for ii in np.arange(numcalcs):
        sigmaii = sigma[ii]
        hmatchii = hmatch[ii]
        if hmatchii != 0:
            coth_arg = np.power(sigmaii * sigmaii * hmatchii / g, 3 / 4)
            coth_eval = np.cosh(coth_arg) / np.sinh(coth_arg)
            khii = sigmaii * sigmaii * hmatchii / g * np.power(coth_eval, 2 / 3)
            sech_eval = 1 / np.cosh(khii)
            kh[ii] = khii - ((sigmaii * sigmaii * hmatchii / g - khii * np.tanh(khii)) /
                             (-1. * (np.tanh(khii) + khii * sech_eval * sech_eval)))
            err = abs(kh[ii] - khii) / kh[ii]
            while err > 1e-6:
                khii = kh[ii]
                kh[ii] = khii - ((sigmaii * sigmaii * hmatchii / g - khii * np.tanh(khii)) /
                                 (-1. * (np.tanh(khii) + khii * sech_eval * sech_eval)))
                err = abs(kh[ii] - khii) / kh[ii]
            k[ii] = np.real(kh[ii] / hmatch[ii])

    # calculate L, T, etc from dispersion relation
    L = (2 * np.pi) / k
    # Tmatch = 2 * np.pi / sigma
    C = L / Tmatch
    n = 0.5 * (1 + 2. * k * h / np.sinh(2 * k * h))
    Cg = n * C

    # if input h and T are integers, then return only 1 value
    if int_flag:
        hmatch = hmatch[0]
        Tmatch = Tmatch[0]
        k = k[0]
        L = L[0]
        C = C[0]
        Cg = Cg[0]
        n = n[0]

    return hmatch, Tmatch, k, L, C, Cg, n


def wavenumber_fsolve(h, T, g):
    from scipy import optimize
    import sys

    # First, assess input parameters; if int, convert to array
    int_flag = False
    if (isinstance(h, int)) and (isinstance(T, int)):
        h = np.array([h])
        T = np.array([T])
        int_flag = True
    elif (isinstance(T, int)) or (isinstance(T, float) and T.shape == ()):
        T = np.array([T])
    elif (isinstance(h, int)) or (isinstance(h, float) and h.shape == ()):
        h = np.array([h])

    # now, create matching vectors for h and T
    if h.shape == T.shape:  # h and T are the same size, point-wise calcs
        k = np.zeros(shape=h.shape)
        kh = np.zeros(shape=h.shape)
        numcalcs = h.size
        hmatch = h
        Tmatch = T
        if (h.size == 1) and (T.size == 1):
            hmatch = np.ones((2,)) * h
            Tmatch = np.ones((2,)) * T
    elif (h.shape != T.shape) and not ((h.size == 1) or (T.size == 1)):
        print('ERROR! If h and T both sized > 1, then they must be identically shaped. ')
        Tmatch = []
        sys.exit()
    elif (h.shape != T.shape) and (h.size == 1):
        k = np.zeros(shape=T.shape)
        kh = np.zeros(shape=T.shape)
        numcalcs = T.size
        hmatch = np.ones(shape=T.shape) * h
        Tmatch = T
    elif (h.shape != T.shape) and (T.size == 1):
        k = np.zeros(shape=h.shape)
        kh = np.zeros(shape=h.shape)
        numcalcs = h.size
        Tmatch = np.ones(shape=h.shape) * T
        hmatch = h
    else:
        print('ERROR in wavenumber')
        Tmatch = []
        sys.exit()

    # convert wave period to angular wave frequency
    sigma = 2. * np.pi / Tmatch

    # use fsolve to numerically solve for k, with initial guess of x0
    ksolve = np.zeros(shape=h.shape)
    ksolve[ksolve == 0] = np.nan
    for ii in np.arange(numcalcs):
        sigmaii = sigma[ii]
        hmatchii = hmatch[ii]
        if hmatchii != 0:
            # define dispersion relation as a func with sum = 0
            def myfunc(k, h_in=hmatchii, sigma_in=sigmaii, g_in=g):
                return sigma_in * sigma_in - g_in * k * np.tanh(k * h_in)
            ktmp = optimize.fsolve(myfunc,np.array(1))
            ksolve[ii] = ktmp

    return ksolve


###############################  WAVETRANSFORTM_XSHORE  ###############################

def wavetransform_xshore(H0, theta0, H1, theta1, T, h2, h1, g, breakcrit):
    """
    :param: H0 (deep-water wave height [m]) - can be singular or array,
            theta0 (deep-water direction [deg]) - can be singular or array; *relative to shore-normal*,
            H1 (wave height at Pt.1 [m]) - can be singular or array,
            theta1 (direction at Pt.1 [deg]) - can be singular or array; *relative to shore-normal*,
            T (period [s]) - can be singular or array,
            h2 (depth of transect of interest [m]) - should be an array -- h2(x)
            h1 (depth at Pt.1 [m]) - can be singular or array
            g (gravity [m/s^2])
            breakcrit (depth-limited breaking criterion [m/m]) - range 0-1; if NAN, breaking not applied
    :return: H2 - wave height as a function of h(x) [m],
            theta2 - wave angle as a function of h(x) [rad]
    """

    # Are we given H0/theta0 or H1/theta1?
    start_flag = np.nan
    if (not np.all(np.isnan(H1))) and (not np.all(np.isnan(theta1))) and (not np.all(np.isnan(h1))):
        print('Calculating wave transform from known depth (Pt.1)...')
        start_flag = 1
        theta1_rad = theta1 * np.pi / 180
        if isinstance(h1, int):
            h1 = np.array([h1])
    elif (not np.all(np.isnan(H0))) and (not np.all(np.isnan(theta0))):
        print('Calculating wave transform from Deep Water (Pt.0)...')
        start_flag = 0
        theta0_rad = theta0 * np.pi / 180
    else:
        print('Either H0/theta0 *or* H1/theta1/h1 must be provided')

    # First, determine the size and shape of input depths
    notint_flag = True
    if isinstance(h2, int):
        notint_flag = False
    elif h2.size == 1:
        notint_flag = False
    if not notint_flag:
        print('Are you trying to transform many waves to new depth?  Try wavetransform_point()')

    # Convert T to angular frequency
    sigma = 2 * np.pi / T

    # work from offshore to onshore
    flip_flag = False
    if h2[0] < h2[-1]:
        h2 = np.flip(h2)
        flip_flag = True

    # Ok, calculate conditions at given known location
    if start_flag == 0:
        [htmp, Ttmp, k, L, C, Cg, n] = wavenumber(h2, T, g)
        k1 = k[0]  # Define deep edge of vector as starting point
        h1 = h2[0]  # Define deep edge of vector at starting point
        C1 = sigma / k1
        Cg1 = (C1 / 2) * (1 + (2 * k1 * h1) / (np.sinh(2 * k * h1)))
        C0 = C1 / np.tanh(k * h1)
        theta1_rad = np.arcsin(np.sin(theta0_rad) * C1 / C0)
        H1 = H0 * np.sqrt(C0 / (2 * Cg1)) * np.sqrt(np.cos(theta0_rad) / np.cos(theta1_rad))
    elif start_flag == 1:
        [htmp, Ttmp, k, L, C, Cg, n] = wavenumber(h1, T, g)
        C1 = sigma / k
        Cg1 = (C1 / 2) * (1 + (2 * k * h1) / (np.sinh(2 * k * h1)))
        [htmp, Ttmp, k, L, C, Cg, n] = wavenumber(h2, T, g)

    # propagate landward
    C2 = sigma / k  # celerity C(x)
    Cg2 = (C2 / 2) * (1 + (2 * k * h2) / (np.sinh(2. * k * h2)))  # group speed Cg(x)
    theta2_rad = np.arcsin(np.sin(theta1_rad) * C2 / C1)  # direction [rad]
    theta2 = theta2_rad * 180 / np.pi  # direction [deg] dir(x)
    H2 = H1 * np.sqrt(Cg1 / Cg2) * np.sqrt(np.cos(theta1_rad) / np.cos(theta2_rad))  # wave height H(x)

    if ~np.isnan(breakcrit):
        gamma = H2 / h2
        H2[gamma > breakcrit] = breakcrit * h2[gamma > breakcrit]
    if flip_flag:
        H2 = np.flip(H2)
        theta2 = np.flip(theta2)

    return H2, theta2


###############################  WAVETRANSFORTM_POINT  ###############################

def wavetransform_point(H0, theta0, H1, theta1, T, h2, h1, g, breakcrit):
    """
    Calculates wave transformation to Pt 2 from offshore (H0, theta0) OR inshore (H1,theta1) data

      :param: H0 (deep-water wave height [m]) - can be singular or array,
              theta0 (deep-water direction [deg]) - can be singular or array; *relative to shore-normal*,
              H1 (wave height at Pt.1 [m]) - can be singular or array,
              theta0 (direction at Pt.1 [deg]) - can be singular or array; *relative to shore-normal*,
              T (period [s]) - can be singular or array,
              h2 (depth of POINT of interest [m]) - should be singular
              h1 (depth at Pt.1 [m]) - can be singular or array
              g (gravity [m/s^2])
              breakcrit (depth-limited breaking criterion [m/m]) - range 0-1; if NAN, breaking not applied
      :return: H2 - wave height at point of interest (Pt.2) [m],
              theta2 - wave angle at point of interest (Pt.2) [rad]
      """

    # Are we given H0/theta0 or H1/theta1?
    start_flag = np.nan
    if (not np.all(np.isnan(H1))) and (not np.all(np.isnan(theta1))) and (not np.all(np.isnan(h1))):
        print('Calculating wave transform from known depth (Pt.1)...')
        start_flag = 1
        theta1_rad = theta1 * np.pi / 180
        if isinstance(h1, int):
            h1 = np.array([h1])
    elif (not np.all(np.isnan(H0))) and (not np.all(np.isnan(theta0))):
        print('Calculating wave transform from Deep Water (Pt.0)...')
        start_flag = 0
        theta0_rad = theta0 * np.pi / 180
    else:
        print('Either H0/theta0 *or* H1/theta1/h1 must be provided')

    # First, determine the size and shape of input depths
    notint_flag = True
    if isinstance(h2, int):
        notint_flag = False
        h2 = np.array([h2])
    elif h2.size == 1:
        notint_flag = False
    if notint_flag:
        print('Are you trying to transform a wave over a transect?  Try wavetransform_xshore()')

    # Convert T to angular frequency
    sigma = 2 * np.pi / T

    # Ok, calculate conditions at given known location
    if start_flag == 0:
        [htmp, Ttmp, k, L, C, Cg, n] = wavenumber(h2, T, g)
        C1 = sigma / k
        Cg1 = (C1 / 2) * (1 + (2 * k * h2) / (np.sinh(2 * k * h2)))
        C0 = C1 / np.tanh(k * h2)
        theta1_rad = np.arcsin(np.sin(theta0_rad) * C1 / C0)
        H1 = H0 * np.sqrt(C0 / (2 * Cg1)) * np.sqrt(np.cos(theta0_rad) / np.cos(theta1_rad))
    elif start_flag == 1:
        [htmp, Ttmp, k, L, C, Cg, n] = wavenumber(h1, T, g)
        C1 = sigma / k
        Cg1 = (C1 / 2) * (1 + (2 * k * h1) / (np.sinh(2 * k * h1)))
        [htmp, Ttmp, k, L, C, Cg, n] = wavenumber(h2, T, g)

    # continue calcs...
    C2 = sigma / k  # celerity C(x)
    Cg2 = (C2 / 2) * (1 + (2 * k * h2) / (np.sinh(2. * k * h2)))  # group speed Cg
    theta2_rad = np.arcsin(np.sin(theta1_rad) * C2 / C1)  # direction [rad]
    theta2 = theta2_rad * 180 / np.pi  # direction [deg]
    H2 = H1 * np.sqrt(Cg1 / Cg2) * np.sqrt(np.cos(theta1_rad) / np.cos(theta2_rad))  # wave height H

    # apply wave-breaking if requested - RECALL h2 is a POINT here
    gamma = H2 / h2
    if ~np.isnan(breakcrit):
        H2[gamma > breakcrit] = breakcrit * h2

    return H2, theta2
