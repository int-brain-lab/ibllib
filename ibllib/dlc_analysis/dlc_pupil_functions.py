import numpy as np


def fit_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv = np.sum(u*v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv])/2.0
    uc, vc = np.linalg.solve(A, B)
    xc_1 = x_m + uc
    yc_1 = y_m + vc
    Ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1 = np.mean(Ri_1)
    return xc_1, yc_1, R_1


def pupil_features(dlc_data):
    vec_x = [dlc_data['pupil_left_r_x'], dlc_data['pupil_right_r_x'],
             dlc_data['pupil_top_r_x'], dlc_data['pupil_bottom_r_x']]
    vec_y = [dlc_data['pupil_left_r_y'], dlc_data['pupil_right_r_y'],
             dlc_data['pupil_top_r_y'], dlc_data['pupil_bottom_r_y']]
    x = np.zeros(len(vec_x[0]))
    y = np.zeros(len(vec_x[0]))
    diameter = np.zeros(len(vec_x[0]))
    for i in range(len(vec_x[0])):
        try:
            x[i], y[i], R = fit_circle([vec_x[0][i], vec_x[1][i], vec_x[2][i], vec_x[3][i]],
                                       [vec_y[0][i], vec_y[1][i], vec_y[2][i], vec_y[3][i]])
            diameter[i] = R*2
        except:
            x[i] = np.nan
            y[i] = np.nan
            diameter = np.nan
    return x, y, diameter
