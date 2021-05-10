import numpy as np

def cagr(end_arr, start_arr, t_in_q):
    """
    Compute implied CAGR to go from start_arr to end_arr
    :param end_arr: Ending ARR
    :param start_arr: Starting ARR
    :param t_in_q: Time in quarters
    :return: CAGR
    """
    # assert t_in_q >= 0 or np.isnan(float(t_in_q))
    t_in_yrs = t_in_q/4.0
    return ((end_arr / start_arr) ** (1 / t_in_yrs)-1)*100

def fv(start_arr, cagr, t_in_q):
    """
    Future value: compute end_arr given start_arr and CAGR
    :param start_arr: Starting ARR
    :param cagr:
    :param t_in_q: time in quarters
    :return:
    """
    return start_arr*(1+cagr/100)**t_in_q

def time_between_arr_ranges(end_arr, start_arr, r):
    """
    Compute time (in quarters) elapsed between start_arr and end_arr assuming a CAGR of r
    :param end_arr:
    :param start_arr:
    :param r: CAGR
    :return: time in quarters
    """
    # returns time in quarters
    return (np.log(end_arr / start_arr) / np.log(1 + r / 100))*4


def interpolate_time(arr, arr_p, t_in_q, arr_i):
    """
    Interpolate the time ARR_i occurs given that arr and arr_p occur t_in_q quarters apart
    :param arr:
    :param arr_p:
    :param t_in_q:
    :param arr_i:
    :return:
    """
    r = cagr(end_arr=arr, start_arr=arr_p, t_in_q=t_in_q)
    t_x = time_between_arr_ranges(end_arr=arr, start_arr=arr_i, r=r)
    return t_x
