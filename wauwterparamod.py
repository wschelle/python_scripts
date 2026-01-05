import numpy as np
from scipy.interpolate import interp1d

def designmat(onsets, maxtime, TR=1, upsample_factor=10):
    nr_factors = int(np.max(onsets[:, 0]) + 1)
    nr_events = int(onsets.shape[0])
    fmat = np.zeros([nr_factors, int(maxtime * upsample_factor)])

    if onsets.shape[1] == 3:
        onsets = np.c_[onsets, np.ones(nr_events)]  # add amplitude = 1

    for i in range(nr_events):
        onset_idx = int(np.round(onsets[i, 1] * upsample_factor))
        offset_idx = int(np.round((onsets[i, 1] + onsets[i, 2]) * upsample_factor))
        fmat[int(onsets[i, 0]), onset_idx:offset_idx] = onsets[i, 3]

    return fmat

def orthogonalize(p, X):
    projection = X @ np.linalg.pinv(X) @ p
    return p - projection

def add_parametric_modulation(dm_source, dm_target, condition_indices, weights):
    mod = np.zeros(dm_source.shape[1])
    for idx, w in zip(condition_indices, weights):
        mod += dm_source[idx] * w
    ortho_mod = orthogonalize(mod, dm_target.T)
    return np.vstack((dm_target, ortho_mod))

def convmat(designmatrix, maxtime, TR=1, upsample_factor=10, hrf='gamma', normalize_f=True):
    if hrf == 'gamma':
        hrf = gammahrf(25, 1 / upsample_factor)
    else:
        hrf = gloverhrf(25, 1 / upsample_factor)

    n_regressors = designmatrix.shape[0]
    fmat_conv = np.zeros([n_regressors, int(maxtime * upsample_factor + len(hrf) - 1)])
    fmat_conv_intp = np.zeros([n_regressors, int(np.round(maxtime / TR))])

    realtime = np.arange(0, maxtime, 1 / upsample_factor)
    scantime = np.arange(0, maxtime, TR)
    if scantime[-1] >= maxtime:
        scantime = scantime[:-1]

    for i in range(n_regressors):
        fconv = np.convolve(designmatrix[i], hrf)
        if normalize_f and np.max(fconv) != 0:
            fconv /= np.max(fconv)
        fcon = interp1d(realtime[:int(maxtime * upsample_factor)], fconv[:int(maxtime * upsample_factor)], kind='cubic')
        interp_vals = fcon(scantime)
        interp_vals -= np.mean(interp_vals)  # mean-centering
        fmat_conv_intp[i] = interp_vals

    return fmat_conv_intp

# --- Example usage ---

# Assume ons1, ons4, ntt, tr, and gammahrf are defined
ons1b=deepcopy(ons1)
ons1b[ons1[:,0]<4,2]=3
ons1b[(ons1[:,0]>=4)&(ons1[:,2]<8),2]=6
ons1b[ons1[:,0]>=8,2]=2
dm1 = designmat(ons1, ntt * tr, TR=tr)
dm4 = designmat(ons4, ntt * tr, TR=tr)

# Parametric modulations
dm4 = add_parametric_modulation(dm1, dm4, [1, 2, 3], [1, 2, 3])
dm4 = add_parametric_modulation(dm1, dm4, [5, 6, 7], [1, 2, 3])

# Convolve and mean-center
cdm4 = convmat(dm4, ntt * tr, TR=tr)
