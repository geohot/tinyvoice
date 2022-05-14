#!/usr/bin/env python3
import os
import time
import csv
import torch
import random
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch import log_softmax, nn
import torch.optim as optim
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from preprocess import to_text, CHARSET, from_text
from preprocess_libri import load_example
from model import Rec

def load_data(dset):
  global ex_x, ex_y, meta
  print("loading data")
  ex_x, ex_y, meta = torch.load('data/'+dset+'.pt') #, map_location="cuda:0")
  """
  print("copying to GPU", ex_x.shape, ex_x.dtype)
  ex_x = ex_x.to(device="cuda:0", non_blocking=True)
  print("copying to GPU", ex_y.shape, ex_y.dtype)
  ex_y = ex_y.to(device="cuda:0", non_blocking=True)
  """
  print("data loaded")

# TODO: is this the correct shape? possible we are masking batch?
# from docs, specgram (Tensor): Tensor of dimension (..., freq, time).
train_audio_transforms = nn.Sequential(
  # 80 is the full thing
  torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
  # 256 is the hop size, so 86 is one second
  torchaudio.transforms.TimeMasking(time_mask_param=35)
)

def get_sample(samples, val=False):
  input_lengths = [meta[i][1] for i in samples]
  target_lengths = [meta[i][2] for i in samples]
  max_input_length = max(input_lengths)
  X = ex_x[samples, :max_input_length].to(device='cuda:0', non_blocking=True).type(torch.float32)
  Y = ex_y[samples].to(device='cuda:0', non_blocking=True)

  # 4x downscale in encoder
  input_lengths = [x//4 for x in input_lengths]

  # to the GPU
  #input_lengths = torch.tensor(input_lengths, dtype=torch.int32).cuda()
  #target_lengths = torch.tensor(target_lengths, dtype=torch.int32).cuda()

  if not val:
    X = train_audio_transforms(X.permute(0,2,1)).permute(0,2,1)
  return X, Y, input_lengths, target_lengths

WAN = os.getenv("WAN") != None
if WAN:
  import wandb

def train():
  epochs = 100
  learning_rate = 0.002
  batch_size = 64

  if WAN:
    wandb.init(project="tinyvoice", entity="geohot")
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size
    }

  timestamp = int(time.time())

  model = Rec().cuda()
  #model.load_state_dict(torch.load('models/tinyvoice_1652557893_30.pt'))

  split = int(ex_x.shape[0]*0.9)
  trains = [x for x in list(range(split))]
  vals = [x for x in range(split, ex_x.shape[0])]
  val_batches = np.array(vals)[:len(vals)//batch_size * batch_size].reshape(-1, batch_size)

  #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  import apex
  optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=learning_rate)

  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, pct_start=0.2,
    steps_per_epoch=len(trains)//batch_size, epochs=epochs, anneal_strategy='linear', verbose=False)

  single_val = load_example('data/LJ037-0171.wav').cuda()

  for epoch in range(epochs):
    if WAN:
      wandb.watch(model)

    with torch.no_grad():
      model.eval()

      mguess = model(single_val[None], [single_val.shape[0]])
      pp = to_text(mguess[:, 0, :].argmax(dim=1).cpu())
      print("VALIDATION", pp)
      if epoch%5 == 0:
        fn = f"models/tinyvoice_{timestamp}_{epoch}.pt"
        print(f"saving model {fn}")
        torch.save(model.state_dict(), fn)

      losses = []
      for samples in (t:=tqdm(val_batches)):
        input, target, input_lengths, target_lengths = get_sample(samples, val=True)
        guess = model(input, input_lengths)
        loss = F.ctc_loss(guess, target, input_lengths, target_lengths)
        losses.append(loss)
      val_loss = torch.mean(torch.tensor(losses)).item()
      print(f"val_loss: {val_loss:.2f}")

    if WAN:
      wandb.log({"val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})

    random.shuffle(trains)
    model.train()
    batches = np.array(trains)[:len(trains)//batch_size * batch_size].reshape(-1, batch_size)
    j = 0

    def run_model(samples):
      input, target, input_lengths, target_lengths = samples
      optimizer.zero_grad()
      guess = model(input, input_lengths)
      loss = F.ctc_loss(guess, target, input_lengths, target_lengths)
      loss.backward()
      optimizer.step()
      scheduler.step()
      return loss

    sample = None
    for samples in (t:=tqdm(batches)):
      if sample is not None:
        loss = run_model(sample)
        sample = get_sample(samples)
      else:
        sample = get_sample(samples)
        loss = run_model(sample)

      t.set_description(f"epoch: {epoch} loss: {loss.item():.2f}")
      if WAN and j%10 == 0:
        wandb.log({"loss": loss})
      j += 1

if __name__ == "__main__":
  load_data('libri')
  #load_data('lj')
  train()
