#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor

from tqdm import tqdm

DATA_CSV = 'data/exp-1six/train.csv'
DEVICE = 'cpu'
SAMPLING_RATE = 48000
MAX_DURATION = 5.0

class AST_Dset(Dataset):
    def __init__(self, df):
        super().__init__()
        self._df = df.reset_index(drop=True)
        self._max_seq_len = int(int(SAMPLING_RATE) * MAX_DURATION)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        sig, sr = torchaudio.load(self._df.loc[idx, 'wav_f_path'])
        sig = torch.mean(sig, dim=0)

        # Resampling
        if sr != SAMPLING_RATE:
            sig = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=SAMPLING_RATE)(sig)

        # Trim
        if sig.shape[0] > self._max_seq_len:
            sig = sig[:self._max_seq_len]

        return {
            'wav_fpath': self._df.loc[idx, 'wav_f_path'],
            'sig': sig.numpy(),
            'labels': self._df.loc[idx, 'label']
        }

def extract_feats(batch, feat_extractor):
    wav_fpath_s = list(map(lambda e: e['wav_fpath'], batch))
    sig_s = list(map(lambda e: e['sig'], batch))
    labels_s = list(map(lambda e: e['labels'], batch))

    # Extract feast: [m, num_frames, num_mels]
    feats = feat_extractor(sig_s,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt")

    return {
        'wav_fpath': wav_fpath_s,
        'feats': feats['input_values'],
        'labels': labels_s
    }

if __name__ == "__main__":
    # Dset
    df = pd.read_csv(DATA_CSV)
    dset = AST_Dset(df)

    # feat extractor config
    feat_extractor = ASTFeatureExtractor(
        sampling_rate=SAMPLING_RATE,
        padding="max_length",
        return_tensors="pt",
        feature_size=1,
        num_mel_bins=128,
        padding_side="right",
        padding_value=0.0,
        return_attention_mask=True,
        max_length=1024,
        do_normalize=False,
        mean=-4.612934112548828,
        std=5.269898891448975
    )

    # dloader
    dloader = DataLoader(dset,
        shuffle=True,
        batch_size=16,
        pin_memory=True,
        collate_fn=lambda batch: extract_feats(batch, feat_extractor),
        num_workers=64)

    mean_s = []
    std_s = []
    max_seq_length = 0
    for i, batch in enumerate(tqdm(dloader, desc="Calculating")):
        cur_mean = torch.mean(batch['feats'])
        cur_std = torch.std(batch['feats'])

        for i in range(batch['feats'].shape[0]):
            max_seq_length = max(max_seq_length, torch.count_nonzero(batch['feats'][i,:, 0]))

        # print(cur_mean, cur_std)
        mean_s.append(cur_mean)
        std_s.append(cur_std)

    print(f"{max_seq_length = }")
    print(f"mean = {np.mean(mean_s)}")
    print(f"std = {np.mean(std_s)}")
