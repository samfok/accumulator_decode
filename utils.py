"""Utilities for accumulator experiments"""
import numpy as np
import pickle

def get_snr_gamma(lamtau_out, k):
    """SNR of the synaptically filtered gamma process

    In terms of the output lambda * tau
    """
    if isinstance(lamtau_out, float):
        lamtau_out = np.array([lamtau_out])
    lamtau_in = lamtau_out * k
    snr = np.zeros_like(lamtau_out)
    idx = lamtau_out > 0
    x = lamtau_in
    a = np.sqrt(2*x[idx])
    b_num = (1+x[idx])**k+x[idx]**k
    b_den = (1+x[idx])**k-x[idx]**k
    b = b_num/b_den
    c = 2*x[idx]/k
    snr[idx] = a / np.sqrt(k*(b-c))
    return snr

def get_snr_periodic(lamtau):
    """SNR of the synaptically filtered periodic process"""
    snr = np.zeros_like(lamtau)
    idx = lamtau > 0
    snr[idx] = 1./np.sqrt(
        1./(2.*lamtau[idx])*(1+np.exp(-1/(lamtau[idx])))/(1-np.exp(-1/(lamtau[idx])))-1)
    return snr

def check_bins(bins):
    """Check that binned data is reasonable"""
    print("Collected {:d} non-zero bins.".format(np.sum(bins > 0)))
    ubin_vals = np.unique(bins)
    cutoff = 0
    info = "Bin stats (bin value : N bins)"
    for uval in ubin_vals:
        info += "\n{: 20d} : {:d}".format(uval, np.sum(bins == uval))
    print(info)

    if len(ubin_vals) > 1:
        cutoff = 0
        for idx in range(len(ubin_vals)-1):
            if cutoff == 0 and ubin_vals[idx] > 0:
                if 10*ubin_vals[idx] < ubin_vals[idx+1]:
                    cutoff = ubin_vals[idx+1]
                    break
        if cutoff > 0:
            for uval in ubin_vals[ubin_vals >= cutoff]:
                sticky_idx = bins == uval
                bins[sticky_idx] = 0
                print("zeroed out sticky-bitted bins with value {:d}".format(uval))
    print("Sum(bins) = {:d}".format(np.sum(bins)))
    return bins

def cache_fun(fname_cache, fun):
    """Check whether cached data exists, otherwise call fun and return
    
    Parameters
    ----------
    fname_cache: string
        name of cache to look for
    fun: function
        function to call in case cache doesn't exist
        probably a lambda function
    """
    try:
        with open(fname_cache, 'rb') as fhandle:
            ret = pickle.load(fhandle)
    except (FileNotFoundError, EOFError):
        ret = fun()
        with open(fname_cache, 'wb') as fhandle:
            pickle.dump(ret, fhandle)
    return ret

def optimize_yield_w_cache(ps_orig, fname_cache, calibrator):
    """Wrap the calibrator's optimize yield with caching"""
    def fun():
        ps, dac, est_enc, est_off, dbg = calibrator.optimize_yield(ps_orig)
        return ps
    ps = cache_fun(fname_cache, fun)
    return ps
