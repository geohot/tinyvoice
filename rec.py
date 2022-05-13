#!/usr/bin/env python3
import os
import time
import csv
import torch
import random
from tqdm.auto import tqdm
from torch import log_softmax, nn
import torch.optim as optim
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from preprocess import load_example, to_text, CHARSET

print("loading data X")
ex_x = torch.load('data/lj_x.pt') #, map_location="cuda:0")
print("copying to GPU", ex_x.shape)
ex_x = ex_x.to(device="cuda:0", non_blocking=True)
print("loading data Y")
ex_y = torch.load('data/lj_y.pt')
print("data loaded")

def get_sample(samples):
  input = ex_x[:, samples]
  input_lengths = [ex_y[i][1] for i in samples]
  target = sum([ex_y[i][2] for i in samples], [])
  target_lengths = [len(ex_y[i][2]) for i in samples]
  return input, target, input_lengths, target_lengths

class TemporalBatchNorm(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.bn = nn.BatchNorm1d(channels)

  def forward(self, x):
    # (L, N, C) -> (N, C, L) -> (L, N, C)
    return self.bn(x.permute(1,2,0)).permute(2,0,1)

class Rec(nn.Module):
  def __init__(self):
    super().__init__()
    H = 256
    # (L, N, C)
    self.prepare = nn.Sequential(
      nn.Linear(80, H),
      TemporalBatchNorm(H),
      nn.ReLU(),
      nn.Linear(H, H),
      TemporalBatchNorm(H),
      nn.ReLU(),
      nn.Linear(H, H),
      TemporalBatchNorm(H),
      nn.ReLU())
    #self.encoder = nn.GRU(H, H//2, batch_first=False, bidirectional=True)
    self.encoder = nn.GRU(H, H, batch_first=False)
    self.decode = nn.Sequential(
      nn.Linear(H, H//2),
      TemporalBatchNorm(H//2),
      nn.ReLU(),
      nn.Linear(H//2, H//4),
      TemporalBatchNorm(H//4),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(H//4, len(CHARSET))
    )

  def forward(self, x):
    x = self.prepare(x)
    x = self.encoder(x)[0]
    x = self.decode(x)
    return torch.nn.functional.log_softmax(x, dim=2)

WAN = os.getenv("WAN") != None
if WAN:
  import wandb

def train():
  epochs = 100
  learning_rate = 0.005
  batch_size = 128

  if WAN:
    wandb.init(project="tinyvoice", entity="geohot")
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size
    }

  timestamp = int(time.time())
  ctc_loss = nn.CTCLoss().cuda()
  model = Rec().cuda()
  #model.load_state_dict(torch.load('models/tinyvoice_1652479269_25.pt'))

  split = int(ex_x.shape[1]*0.9)
  trains = [x for x in list(range(split))*4]
  vals = [x for x in range(split, ex_x.shape[1])]
  val_batches = np.array(vals)[:len(vals)//batch_size * batch_size].reshape(-1, batch_size)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  #import apex
  #optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=learning_rate)

  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
    steps_per_epoch=len(trains)//batch_size, epochs=epochs, anneal_strategy='linear', verbose=False)

  single_val = load_example('data/LJ037-0171.wav').cuda()

  # TODO: is this the correct shape? possible we are masking batch?
  # from docs, specgram (Tensor): Tensor of dimension (..., freq, time).
  train_audio_transforms = nn.Sequential(
    # 80 is the full thing
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    # 256 is the hop size, so 86 is one second
    torchaudio.transforms.TimeMasking(time_mask_param=35)
  )

  for epoch in range(epochs):
    if WAN:
      wandb.watch(model)

    with torch.no_grad():
      model.eval()

      mguess = model(single_val[:, None])
      pp = to_text(mguess[:, 0, :].argmax(dim=1).cpu())
      print("VALIDATION", pp)
      if epoch%5 == 0 and epoch != 0:
        torch.save(model.state_dict(), f"models/tinyvoice_{timestamp}_{epoch}.pt")

      losses = []
      for samples in (t:=tqdm(val_batches)):
        input, target, input_lengths, target_lengths = get_sample(samples)
        target = torch.tensor(target, dtype=torch.int32, device='cuda:0')
        guess = model(input)
        loss = ctc_loss(guess, target, input_lengths, target_lengths)
        #print(loss)
        losses.append(loss)
      val_loss = torch.mean(torch.tensor(losses)).item()
      print(f"val_loss: {val_loss:.2f}")

    if WAN:
      wandb.log({"val_loss": val_loss, "lr": scheduler.get_lr()})

    random.shuffle(trains)
    model.train()
    batches = np.array(trains)[:len(trains)//batch_size * batch_size].reshape(-1, batch_size)
    j = 0
    for samples in (t:=tqdm(batches)):
      input, target, input_lengths, target_lengths = get_sample(samples)
      # input is (time, batch, freq) -> (batch, freq, time)
      input = train_audio_transforms(input.permute(1,2,0)).permute(2,0,1)
      target = torch.tensor(target, dtype=torch.int32, device='cuda:0')

      optimizer.zero_grad()
      guess = model(input)

      """
      pp = to_text(guess[:, 0, :].argmax(dim=1).cpu())
      if len(pp) > 0:
        print(pp)
      """

      loss = ctc_loss(guess, target, input_lengths, target_lengths)
      loss.backward()
      optimizer.step()
      scheduler.step()

      t.set_description(f"epoch: {epoch} loss: {loss.item():.2f}")
      if WAN and j%10 == 0:
        wandb.log({"loss": loss})
      j += 1

if __name__ == "__main__":
  train()
