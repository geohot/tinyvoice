#!/usr/bin/env python3
import os
import time
import csv
import torch
from tqdm.auto import tqdm
from torch import log_softmax, nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset

DATASET = "/raid/ljspeech/LJSpeech-1.1"
CHARSET = " abcdefghijklmnopqrstuvwxyz,."
XMAX = 870    # about 10 seconds
YMAX = 150
SAMPLE_RATE = 22050

def get_metadata():
  ret = []
  with open(os.path.join(DATASET, 'metadata.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for row in reader:
      answer = [CHARSET.index(c)+1 for c in row[1].lower() if c in CHARSET]
      if len(answer) <= YMAX:
        ret.append((os.path.join(DATASET, 'wavs', row[0]+".wav"), answer))
  print("got metadata", len(ret))
  return ret

mel_transform = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=1024, win_length=1024, hop_length=256, n_mels=80)
def load_example(x):
  waveform, sample_rate = torchaudio.load(x, normalize=True)
  assert(sample_rate == SAMPLE_RATE)
  mel_specgram = mel_transform(waveform)
  #return 10*torch.log10(mel_specgram[0]).T
  return mel_specgram[0].T

cache = {}
class LJSpeech(Dataset):
  def __init__(self):
    self.meta = get_metadata()

  def __len__(self):
    return len(self.meta)

  def __getitem__(self, idx):
    if idx not in cache:
      x,y = self.meta[idx]
      cache[idx] = load_example(x), y
    return cache[idx]

class Rec(nn.Module):
  def __init__(self):
    super().__init__()
    # (L, N, C)
    self.prepare = nn.Sequential(
      nn.Linear(80, 128),
      #nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Linear(128, 128),
      #nn.BatchNorm1d(128),
      nn.ReLU())
    self.encoder = nn.GRU(128, 128, batch_first=False)
    self.decode = nn.Sequential(
      nn.Linear(128, 64),
      #nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, len(CHARSET))
    )

  def forward(self, x):
    x = self.prepare(x)
    x = nn.functional.relu(self.encoder(x)[0])
    x = self.decode(x)
    return torch.nn.functional.log_softmax(x, dim=2)

def pad_sequence(batch):
  sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
  input_lengths = [x[0].shape[0] for x in sorted_batch]
  #input_lengths = [sorted_batch[0][0].shape[0] for x in sorted_batch]
  target_lengths = [len(x[1]) for x in sorted_batch]
  sequences = [x[0] for x in sorted_batch]
  sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False)
  labels = [x[1]+[0]*(YMAX - len(x[1])) for x in sorted_batch]
  labels = torch.LongTensor(labels)
  return sequences_padded, labels[:, :max(target_lengths)], input_lengths, target_lengths

def get_dataloader(batch_size):
  dset = LJSpeech()
  trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_sequence)
  return dset, trainloader

import wandb

def train():
  wandb.init(project="tinyvoice", entity="geohot")

  epochs = 100
  learning_rate = 0.001
  batch_size = 32
  wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size
  }

  timestamp = int(time.time())
  dset, trainloader = get_dataloader(batch_size)
  ctc_loss = nn.CTCLoss(reduction='mean', zero_infinity=True).cuda()
  model = Rec().cuda()
  #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  val = torch.tensor(load_example('data/LJ037-0171.wav')).cuda()
  for epoch in range(epochs):
    mguess = model(val[None])
    pp = ''.join([CHARSET[c-1] for c in mguess[:, 0, :].argmax(dim=1).cpu() if c != 0])
    print("VALIDATION", pp)
    torch.save(model.state_dict(), f"models/tinyvoice_{timestamp}_{epoch}.pt")

    t = tqdm(trainloader, total=len(dset)//batch_size)
    for data in t:
      input, target, input_lengths, target_lengths = data
      input = input.to('cuda:0', non_blocking=True)
      target = target.to('cuda:0', non_blocking=True)
      optimizer.zero_grad()
      guess = model(input)
      #print(input)
      #print(guess)
      #print(target)
      #print(guess.shape, target.shape, input_lengths, target_lengths)

      pp = ''.join([CHARSET[c-1] for c in guess[:, 0, :].argmax(dim=1).cpu() if c != 0])
      if len(pp) > 0:
        print(pp)

      loss = ctc_loss(guess, target, input_lengths, target_lengths)
      #print(loss)
      #loss = loss.mean()
      loss.backward()
      optimizer.step()
      t.set_description("loss: %.2f" % loss.item())
      wandb.log({"loss": loss})
      wandb.watch(model)


if __name__ == "__main__":
  train()
