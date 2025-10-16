import pandas as pd
import numpy as np
from dotwiz import DotWiz
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from poseidon.io.iara.offline import load_sonar_from_csv, Target, load_processed_data
from poseidon.dataset.dataset import SonarRunDataset, SonarRunPairDataset
from poseidon.model_selection import SonarCrossValidator
from poseidon.signal.passivesonar import lofar
from poseidon.signal.utils import resample

from poseidon.visualization import plot_lofargram
import matplotlib.pyplot as plt

config =  {
    "dataset": {
        "name": "iara",
        "metadata_path": "../data/iara.csv",
        "raw_data_path" : "/home/iara/",
        "cache_path" : "./data/iara_cache"
    },
    # "window_size": 1,
    "window_size": 16,
    "overlap" : 0,
    "n_freqs": 512,
    "learning_rate": 0.001,
    "model_name": "MLP",
    "output_size": 4,
    "batch_size": 32,
    "hidden_size": 128,
    "n_splits": 5,
    "fold": 0,
    "epochs": 100,
    "patience": 10,
        "seed": 42
}
config = DotWiz(config)

csv_path = config.dataset.metadata_path
iara_data_root_path = config.dataset.raw_data_path
iara_raw = load_sonar_from_csv(csv_path, data_root_path=iara_data_root_path, target_column="Ship Length Class")

# iara_raw['0']['A-0002']['fs']

cache_dir = config.dataset.cache_path
Path(cache_dir).mkdir(parents=True, exist_ok=True)

def lofar_fn(x):
    signal = resample(x['signal'], x['fs'], 16000)
    fs = 16000
    # x is a dict with keys 'signal' and 'fs'
    # x['signal'] is the audio signal, x['fs'] is the sampling frequency
    return lofar(signal, fs, n_pts_fft=1024, n_overlap=0,
                  spectrum_bins_left=512)


iara_raw.process_and_cache(fn=lofar_fn, max_workers=8, cache_path=cache_dir)

iara_spectrogram = load_processed_data(cache_dir)

cls = '1'
run = 'A-0003'

sxx =  iara_spectrogram[cls][run]['sxx']
freq = iara_spectrogram[cls][run]['freq']
time = iara_spectrogram[cls][run]['time']

from poseidon.visualization import plot_lofargram
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
plot_lofargram(sxx, freq, time, ax=ax)
plt.show()


metadata_df = pd.read_csv(csv_path)
metadata_df['Length'] = metadata_df['Length'].apply(lambda x: np.nan if x == ' - ' else float(x))
metadata_df['Ship Length Class'] = metadata_df['Length'].apply(Target.classify_value)
metadata_df

cross_validator = SonarCrossValidator(
    metadata_df=metadata_df,
    target_column='Ship Length Class',
    stratify_columns=['Ship Length Class'],
    n_splits=config.n_splits,
    random_state=config.seed
)

train_data, test_data = cross_validator.get_fold_data(config.fold, cache_dir)

# from poseidon.utils import calculate_class_weights

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train_dataset = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
# test_dataset = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# class_weights = calculate_class_weights(train_dataset, device)

# input_size = (config.window_size, config.n_freqs)
# model_cl = model_select(config)
# raise NotImplementedError("Model training not implemented yet.")