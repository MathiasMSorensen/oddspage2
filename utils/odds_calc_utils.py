import numpy as np

def bookie_yield(H, D, A):
    try:
        x = (1/H + 1/D + 1/A - 1)
    except ZeroDivisionError:
        x = np.nan
    return x