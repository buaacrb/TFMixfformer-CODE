import fcwt
from scipy.fft import fft
import numpy as np
import torch
import pywt
import sys


def get_cwt(X, fs, lf, hf, fn, nthreads):
    """
    Generate Continuous Wavelet Transform (CWT) array.

    Parameters
    ----------
        X : numpy.ndarray
            an array to be processed by the CWT transform, and
            must have a 3D shape of (samples,timeseries,features).
        scale_range : list of float
            the start and stop scale for the scale_range.
        wavelet_name : str, default 'morl'
            the mother wavelet that the CWT transform should use.
        rescale : bool, default True
            whether to rescale the output of the CWT transform to other dimensions
        upsample : bool, default False
            doubles the samples per timetrace
        rescale_scales : int, default 30
            the size to rescale the scales to if 'rescale' is set to True
        rescale_steps : int, default 30
            the size to rescale the steps to if 'rescale' is set to True

    Returns
    -------
        X_cwt : numpy.ndarray
            4D array of (samples, scales, timesteps, features).
    """
    # samples = X.shape[0]
    metrics = X.shape[1]

    x_dim = X.shape[0]
    y_dim = fn

    # prepare the output array
    X_cwt = np.ndarray(shape=(y_dim, x_dim), dtype='float32')
    flag = 1

    for sensor in range(metrics):
        signal = X[:, sensor]
        # upsample
        # continuous wavelet transform
        _, cwt = fcwt.cwt(signal, fs, lf[sensor], hf[sensor], fn, nthreads)
        # coeffs, _ = pywt.cwt(series, scale_range[sensor], wavelet_name)
        if flag:
            X_cwt = cwt.T
            flag = 0
            continue
        X_cwt = np.concatenate((X_cwt, cwt.T), axis=1)

    return X_cwt

def getRealImag(data):
    real_divisor = np.real(data)
    imag_divisor = np.imag(data)
    
    div = np.divide(imag_divisor, real_divisor, out=np.zeros_like(imag_divisor), where=real_divisor!=0)
    
    pha = np.arctan(-div)
    norm = np.abs(data)
    return norm, pha
    

def timefreqfeatures(data, fs, num_freq):
    """
    Compute the time-frequency features of the data.

    Parameters
    ----------
    data : 2d array
        Data to compute features for.
    fs : float
        Sampling frequency.
    num_freq : int
        The number of selected frequencies
    ----------
    returns
    data_tf : 2d array
        The time-frequency features of the data.
    """
    freq_min = []
    freq_max = []
    tc_data = data.cpu()
    n_data = tc_data.detach().numpy()
    B, L, D = n_data.shape
    res_l = np.empty([B, L, D*num_freq])
    res_norm = np.empty([B, L, D*num_freq])
    res_pha = np.empty([B, L, D*num_freq])
    for n in range(B):
        raw_data = n_data[n]
        for i in range(D):
            x = raw_data[:, i]
            fft_x = fft(x)  
            amp_x = abs(fft_x) / len(x) * 2   
            label_x = np.linspace(0, int(len(x) / 2) - 1, int(len(x) / 2))  
            amp = amp_x[0:int(len(x) / 2)]  
            amp[0] = 0  
            fre = label_x / len(x) * fs  

            res_fft = np.concatenate((amp.reshape(-1, 1), fre.reshape(-1, 1)), axis=1)
            index = np.argsort(-res_fft, axis=0)
            res_sort = res_fft[index[:, 0]]
            freq = np.sort(res_sort[:num_freq, 1])
            for m in range(len(freq)):
                if freq[m] != 0.0:
                    min_freq = freq[m]
                    break
            freq_min.append(min_freq)
            freq_max.append(freq[-1])

        data_tff = get_cwt(raw_data, fs, freq_min, freq_max, num_freq, 32)
        norm, pha = getRealImag(data_tff)
        res_l[n] = data_tff
        res_norm[n] = norm
        res_pha[n] = pha
    res_data = np.stack(res_l, axis=0)
    res_data_norm = np.stack(res_norm, axis=0)
    res_data_pha = np.stack(res_pha, axis=0)

    data_t = torch.from_numpy(res_data)
    data_norm = torch.from_numpy(res_data_norm).float()
    data_pha = torch.from_numpy(res_data_pha).float()
    return data_t, data_norm, data_pha

def data_smoth(data):
    n_data = data.cpu().numpy()
    B, L, D = n_data.shape
    smoth_data = np.empty([B, L, D])
    for n in range(B):
        raw_data = n_data[n]
        rec_data = np.empty([D, L])
        for i in range(D):
            x = raw_data[:, i]
            wave = pywt.Wavelet('db4')
            coeffs = pywt.wavedec(x, wave)
            coeffs[len(coeffs)-1] *= 0
            coeffs[len(coeffs)-2] *= 0
            rec_x = pywt.waverec(coeffs, wave)
            rec_data[i] = rec_x
        smoth_data[n] = np.swapaxes(rec_data, 0, 1)
    res_data = np.stack(smoth_data, axis=0)
    data_t = torch.from_numpy(res_data)
    return data_t
        
            