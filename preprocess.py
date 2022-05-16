#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'

import random
import functools
import csv
from tqdm.auto import tqdm
import torch
from multiprocessing import Pool

CHARSET = " abcdefghijklmnopqrstuvwxyz,.'"
XMAX = 1600
YMAX = 250

import itertools
def to_text(x):
  x = [k for k, g in itertools.groupby(x)]
  return ''.join([CHARSET[c-1] for c in x if c != 0])

def from_text(x):
  return [CHARSET.index(c)+1 for c in x.lower() if c in CHARSET]

mel_transform = {}
def load_example(x):
  import torchaudio
  waveform, sample_rate = torchaudio.load(x, normalize=True)
  if sample_rate not in mel_transform:
    # We extracted 80-channel filterbanks features computed from a 25ms window with a stride of 10ms.
    hop_length = int(sample_rate/(1000/10))
    win_length = int(sample_rate/(1000/25))
    mel_transform[sample_rate] = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=win_length, win_length=win_length, hop_length=hop_length, n_mels=80)
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

def get_librespeech(dirr):
  dispatch = []
  DATASET = "/raid/ljspeech/LibriSpeech"
  for d in tqdm(os.listdir(os.path.join(DATASET, dirr,))):
    for dl in os.listdir(os.path.join(DATASET, dirr, d)):
      meta = os.path.join(DATASET, dirr, d, dl, f"{d}-{dl}.trans.txt")
      meta = open(meta).read().strip().split("\n")
      meta = dict([x.split(" ", 1) for x in meta])

      for dll in os.listdir(os.path.join(DATASET, dirr, d, dl)):
        x = os.path.join(DATASET, dirr, d, dl, dll)
        if x.endswith(".flac"):
          y = meta[dll[:-5]]
          dispatch.append((x,y))
  return dispatch

def get_ljspeech():
  DATASET = "/raid/ljspeech/LJSpeech-1.1"
  ret = []
  with open(os.path.join(DATASET, 'metadata.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for row in reader:
      ret.append((os.path.join(DATASET, 'wavs', row[0]+".wav"), row[1]))
  return ret

def get_cv(f):
  DATASET = "/raid/ljspeech/cv-corpus-9.0-2022-04-27/en"
  ret = []
  with open(f"{DATASET}/{f}.tsv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
      ret.append((f"{DATASET}/clips/{row[1]}", row[2]))
  return ret

def extract(dispatch):
  ex_x, ex_y, ameta = [], [], []
  with Pool(processes=8) as pool:
    #for ex,ey,meta in tqdm(map(proc, dispatch), total=len(dispatch)):
    for ex,ey,meta in tqdm(pool.imap(proc, dispatch, chunksize=100), total=len(dispatch)):
      if ex is not None:
        ex_x.append(ex.clone().type(torch.float16))
        ex_y.append(ey.clone().type(torch.uint8))
        ameta.append(meta)
  sequences_padded = torch.nn.utils.rnn.pad_sequence(ex_x, batch_first=True)
  ys_padded = torch.nn.utils.rnn.pad_sequence(ex_y, batch_first=True)
  return [sequences_padded, ys_padded, ameta]

if __name__ == "__main__":
  dispatch = []
  """
  dispatch += get_librespeech("train-clean-100")
  print(f"got {len(dispatch)}")
  dispatch += get_librespeech("train-clean-360")
  print(f"got {len(dispatch)}")
  dispatch += get_librespeech("test-clean")
  print(f"got {len(dispatch)}")
  dispatch += get_ljspeech()
  """
  dispatch += get_cv("train")
  print(f"got {len(dispatch)}")

  random.seed(1337)
  random.shuffle(dispatch)

  dispatch = dispatch[0:10000]
  X,Y,meta = extract(dispatch)
  print(X.shape, Y.shape)
  """
  X = X.numpy()
  with open("data/big_X.raw", "wb") as f:
    f.write(X.data)
  torch.save([Y,meta], "data/big_Y.pt")
  """
  torch.save([X,Y,meta], "data/cv.pt")

