import numpy as np
import matplotlib.pyplot as plt


def plot_importances(importances, column_names, color='C0'):
    f_imp_median = np.median(importances, axis=0)
    f_imp_q25, f_imp_q75 = np.quantile(importances, axis=0, q=(.25, .75))
    errbar_l = f_imp_q25 - f_imp_median
    errbar_u = f_imp_q75 - f_imp_median

    sort_idx = np.argsort(f_imp_median)
    xerr = (errbar_u[sort_idx], -errbar_l[sort_idx])

    x = np.arange(len(f_imp_median))
    plt.barh(x, width=f_imp_median[sort_idx], xerr=xerr, color=color)
    plt.yticks(x, column_names[sort_idx])
