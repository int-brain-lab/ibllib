import numpy as np
from numpy.linalg import eig, inv

def fit_circle(x,y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)
    xc_1 = x_m + uc
    yc_1 = y_m + vc
    Ri_1      = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1       = np.mean(Ri_1)
    return xc_1, yc_1, R_1

def pupil_features(segments):
    x = np.zeros(len(segments['pupil_top_r_x']))
    y = np.zeros(len(segments['pupil_top_r_x']))
    diameter = np.zeros(len(segments['pupil_top_r_x']))
    for i in range(len(segments['pupil_top_r_x'])):
        try:
            x[i], y[i], R = fit_circle([segments['pupil_left_r_x'], segments['pupil_top_r_x'], \
                                               segments['pupil_right_r_x'], segments['pupil_bottom_r_x']], \
                                                [segments['pupil_left_r_y'], segments['pupil_top_r_y'], \
                                               segments['pupil_right_r_y'], segments['pupil_bottom_r_y']])
            diameter[i] = R*2
        except:
            x[i] = np.nan
            y[i] = np.nan
            diameter = np.nan
    return x, y, diameter
    