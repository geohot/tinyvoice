#!/usr/bin/env python3
import os
import torchaudio
import torch
from tqdm.auto import tqdm

DATASET = "/raid/ljspeech/LibriSpeech"
XMAX = 1050
YMAX = 250
SAMPLE_RATE = 16000
mel_transform = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=1024, win_length=1024, hop_length=256, n_mels=80)
def load_example(x):
  waveform, sample_rate = torchaudio.load(x, normalize=True)
  assert(sample_rate == SAMPLE_RATE)
  mel_specgram = mel_transform(waveform)
  return mel_specgram[0].T

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

  ex_x, ex_y = [], []
  for x,y in tqdm(dispatch):
    ex = load_example(x)
    if ex.shape[0] < XMAX and len(y) < YMAX:
      ex_x.append(ex)
      ex_y.append((x, ex.shape[0], y))
  sequences_padded = torch.nn.utils.rnn.pad_sequence(ex_x, batch_first=False) #.type(torch.float16)
  print(sequences_padded.shape, sequences_padded.dtype)
  print(ex_y[0])
  torch.save(sequences_padded, "data/libri_x.pt")
  torch.save(ex_y, "data/libri_y.pt")


