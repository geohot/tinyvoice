#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from tqdm.auto import tqdm
from preprocess import from_text
from multiprocessing import Pool

DATASET = "/raid/ljspeech/LibriSpeech"
XMAX = 1600
YMAX = 250
MS_PER_SAMPLE = 10

mel_transform = {}
def load_example(x):
  import torchaudio
  waveform, sample_rate = torchaudio.load(x, normalize=True)
  if sample_rate not in mel_transform:
    hop_length = int(sample_rate/(1000/MS_PER_SAMPLE))
    mel_transform[sample_rate] = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=hop_length*4, win_length=hop_length*4, hop_length=hop_length, n_mels=80)
  mel_specgram = mel_transform[sample_rate](waveform)
  return mel_specgram[0].T
  #return torch.log10(mel_specgram[0].T)

def proc(xy):
  x,y = xy
  ex = load_example(x)
  ey = torch.tensor(from_text(y), dtype=torch.uint8)
  if ex.shape[0] < XMAX and len(ey) < YMAX:
    return ex, ey, (x, ex.shape[0], len(ey))
  else:
    return None, None, None

if __name__ == "__main__":
  dispatch = []
  for d in tqdm(os.listdir(os.path.join(DATASET, "train-clean-100"))):
    for dl in os.listdir(os.path.join(DATASET, "train-clean-100", d)):
      meta = os.path.join(DATASET, "train-clean-100", d, dl, f"{d}-{dl}.trans.txt")
      meta = open(meta).read().strip().split("\n")
      meta = dict([x.split(" ", 1) for x in meta])

      for dll in os.listdir(os.path.join(DATASET, "train-clean-100", d, dl)):
        x = os.path.join(DATASET, "train-clean-100", d, dl, dll)
        if x.endswith(".flac"):
          y = meta[dll[:-5]]
          dispatch.append((x,y))

  #dispatch = dispatch[0:1000]
  ex_x, ex_y, ameta = [], [], []
  with Pool(processes=32) as pool:
    #for ex,ey,meta in tqdm(map(proc, dispatch), total=len(dispatch)):
    for ex,ey,meta in tqdm(pool.imap_unordered(proc, dispatch), total=len(dispatch)):
      if ex is not None:
        ex_x.append(ex)
        ex_y.append(ey)
        ameta.append(meta)
  sequences_padded = torch.nn.utils.rnn.pad_sequence(ex_x, batch_first=True).type(torch.float16)
  ys_padded = torch.nn.utils.rnn.pad_sequence(ex_y, batch_first=True).type(torch.uint8)
  torch.save([sequences_padded, ys_padded, ameta], "data/libri.pt")


