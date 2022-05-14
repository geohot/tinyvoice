#!/usr/bin/env python3
import os
import functools
#import torchaudio
import csv
from tqdm.auto import tqdm
import torch

CHARSET = " abcdefghijklmnopqrstuvwxyz,.'"
DATASET = "/raid/ljspeech/LJSpeech-1.1"
SAMPLE_RATE = 22050
XMAX = 870    # about 10 seconds
YMAX = 150

import itertools
def to_text(x):
  x = [k for k, g in itertools.groupby(x)]
  return ''.join([CHARSET[c-1] for c in x if c != 0])

def from_text(x):
  return [CHARSET.index(c)+1 for c in x.lower() if c in CHARSET]

@functools.lru_cache(None)
def get_metadata():
  ret = []
  with open(os.path.join(DATASET, 'metadata.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for row in reader:
      answer = from_text(row[1])
      if len(answer) <= YMAX:
        ret.append((os.path.join(DATASET, 'wavs', row[0]+".wav"), answer))
  return ret

#mel_transform = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=1024, win_length=1024, hop_length=256, n_mels=80)
def load_example(x):
  waveform, sample_rate = torchaudio.load(x, normalize=True)
  assert(sample_rate == SAMPLE_RATE)
  mel_specgram = mel_transform(waveform)
  #return 10*torch.log10(mel_specgram[0]).T
  return mel_specgram[0].T

if __name__ == "__main__":
  meta = get_metadata() #[0:1000]
  ex_x, ex_y = [], []
  for x,y in tqdm(meta):
    ex = load_example(x)
    ex_x.append(ex)
    ex_y.append((x, ex.shape[0], y))

  sequences_padded = torch.nn.utils.rnn.pad_sequence(ex_x, batch_first=False) #.type(torch.float16)
  print(sequences_padded.shape, sequences_padded.dtype)
  print(ex_y[0])
  torch.save(sequences_padded, "data/lj_x.pt")
  torch.save(ex_y, "data/lj_y.pt")
