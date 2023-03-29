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

dataset_path = "./DREAMER.mat"
base_path = "./"

# dataset = DREAMERDataset(
#     io_path=base_path + 'dreamer8sec_unnormalized',
#     mat_path=dataset_path,
#     offline_transform=transforms.Compose([
#         transforms.BaselineRemoval(),
#         transforms.MeanStdNormalize(),
#         transforms.To2d()
#     ]),
#     # online_transform=transforms.ToTensor(),
#     label_transform=transforms.Compose(
#         [transforms.Select('valence'),
#          transforms.Binary(3.0)]),
#     chunk_size=976,
#     baseline_chunk_size=976,
#     num_baseline=8
# )

dataset = DREAMERDataset(
    io_path=base_path + 'dreamer61sec_unnormalized',
    mat_path=dataset_path,
    offline_transform=transforms.Compose([
        transforms.BaselineRemoval(),
        # transforms.MeanStdNormalize(),
        transforms.To2d()
    ]),
    # online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose(
        [transforms.Select('valence'),
         transforms.Binary(3.0)]),
    chunk_size=7808,
    baseline_chunk_size=7808,
    num_baseline=1
)