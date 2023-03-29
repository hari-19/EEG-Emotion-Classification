import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from rich.pretty import pprint
import mne
from sklearn.preprocessing import MinMaxScaler
from torcheeg.datasets import DREAMERDataset
from torcheeg.datasets.constants.emotion_recognition.dreamer import DREAMER_CHANNEL_LOCATION_DICT
from torcheeg import transforms
from torcheeg.utils import plot_signal


def get_tf_feature(eeg, sr, n_channels = 14):
    WinLength = int(0.5*sr) # 500 points (0.5 sec, 500 ms)
    step = int(0.025*sr) # 25 points (or 25 ms)
    final_features = None
    for i in range(n_channels):
        eeg_single = eeg[i].squeeze()
        myparams = dict(nperseg = WinLength, noverlap = WinLength-step, return_onesided=True, mode='magnitude')
        f, nseg, Sxx = signal.spectrogram(x = eeg_single, fs = sr, **myparams)
        if(isinstance(final_features, np.ndarray)):
            final_features = np.concatenate((final_features, Sxx), axis=0)
        else:
            final_features = Sxx
    return final_features

