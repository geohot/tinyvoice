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
from preprocess import to_text, CHARSET, from_text, load_example
from model import Rec

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def load_data(dset):
  print("loading data")
  data = torch.load('data/'+dset+'.pt')
  print("data loaded")
  return data

# TODO: is this the correct shape? possible we are masking batch?
# from docs, specgram (Tensor): Tensor of dimension (..., freq, time).
train_audio_transforms = nn.Sequential(
  # 80 is the full thing
  torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
  # 256 is the hop size, so 86 is one second
  torchaudio.transforms.TimeMasking(time_mask_param=35)
)

def get_sample(samples, data, device, val=False):
  ex_x, ex_y, meta = data
  input_lengths = [meta[i][1] for i in samples]
  target_lengths = [meta[i][2] for i in samples]
  max_input_length = max(input_lengths)
  X = ex_x[samples, :max_input_length].to(device=device, non_blocking=True).type(torch.float32)
  Y = ex_y[samples].to(device=device, non_blocking=True)

  # 4x downscale in encoder
  #input_lengths = [fix_dim(x) for x in input_lengths]

  # to the GPU
  input_lengths = torch.tensor(input_lengths, dtype=torch.int32, device=device)
  target_lengths = torch.tensor(target_lengths, dtype=torch.int32, device=device)

  if not val:
    X = train_audio_transforms(X.permute(0,2,1)).permute(0,2,1)
  return X, Y, input_lengths, target_lengths

WAN = os.getenv("WAN") != None

def train(rank, world_size, data):

  # split dataset
  ex_x, ex_y, meta = data
  sz = ex_x.shape[0]
  sz = sz//world_size
  offset = rank*sz
  ex_x = ex_x[offset:sz+offset]
  ex_y = ex_y[offset:sz+offset]
  meta = meta[offset:sz+offset]
  data = ex_x, ex_y, meta

  print(f"hello from process {rank}/{world_size} data {offset}-{offset+sz}")
  torch.cuda.set_device(rank)
  torch.cuda.empty_cache()

  if WAN and rank == 0:
    import wandb
    wandb.init(project="tinyvoice", entity="geohot")

  if world_size > 1:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

  epochs = 200
  learning_rate = 0.001
  #batch_size = 256//world_size
  batch_size = 96//world_size

  timestamp = int(time.time())

  device = f"cuda:{rank}"
  model = Rec().to(device)
  if world_size > 1:
    model = DDP(model, device_ids=[rank])

  #state_dict = torch.load('demo/tinyvoice_1652627131_165_0.11.pt')
  #state_dict = {k[7:]:v for k,v in state_dict.items()}
  #model.load_state_dict(state_dict)

  sz = ex_x.shape[0]
  split = int(sz*0.97)
  trains = [x for x in range(0, split)]
  vals = [x for x in range(split, sz)]
  val_batches = np.array(vals)[:len(vals)//batch_size * batch_size].reshape(-1, batch_size)

  #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  import apex
  optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=learning_rate)

  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, pct_start=0.2,
    steps_per_epoch=len(trains)//batch_size, epochs=epochs, anneal_strategy='linear', verbose=False)

  single_val = load_example('data/LJ037-0171.wav').to(device)
  last = time.monotonic()
  for epoch in range(epochs):
    if WAN and rank == 0:
      wandb.watch(model)

    with torch.no_grad():
      model.eval()

      mguess, _ = model(single_val[None], torch.tensor([single_val.shape[0]], dtype=torch.int32, device=device))
      pp = to_text(mguess[:, 0, :].argmax(dim=1).cpu())
      print("VALIDATION", pp)

      epoch_time = time.monotonic()-last
      print(f"epoch {epoch-1} took {epoch_time:.2f} s, train in {epoch_time*epochs//60} min, done in {epoch_time*(epochs-epoch)//60} min")
      last = time.monotonic()

      losses = []
      for samples in (t:=tqdm(val_batches)):
        input, target, input_lengths, target_lengths = get_sample(samples, data, device, val=True)
        guess, input_lengths_mod = model(input, input_lengths)
        loss = F.ctc_loss(guess, target, input_lengths_mod, target_lengths, zero_infinity=True)
        losses.append(loss)
      val_loss = torch.mean(torch.tensor(losses)).item()
      print(f"val_loss: {val_loss:.2f}")

      if (epoch%5 == 0 or epoch == epochs-1) and rank == 0:
        fn = f"models/tinyvoice_{timestamp}_{epoch}_{val_loss:.2f}.pt"
        torch.save(model.state_dict(), fn)
        print(f"saved model {fn} with size {os.path.getsize(fn)}")

    if WAN and rank == 0:
      wandb.log({"val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})

    random.shuffle(trains)
    model.train()
    batches = np.array(trains)[:len(trains)//batch_size * batch_size].reshape(-1, batch_size)
    j = 0

    def run_model(samples):
      input, target, input_lengths, target_lengths = samples
      optimizer.zero_grad()
      guess, input_lengths_mod = model(input, input_lengths)
      loss = F.ctc_loss(guess, target, input_lengths_mod, target_lengths, zero_infinity=True)
      loss.backward()
      optimizer.step()
      scheduler.step()
      return loss

    sample = None
    for samples in (t:=tqdm(batches)):
      if sample is not None:
        loss = run_model(sample)
        sample = get_sample(samples, data, device)
      else:
        sample = get_sample(samples, data, device)
        loss = run_model(sample)

      t.set_description(f"epoch: {epoch} loss: {loss.item():.2f} rank: {rank}")
      if WAN and j%10 == 0 and rank == 0:
        wandb.log({"loss": loss})
      j += 1

if __name__ == "__main__":
  data = load_data('cv')

  #load_data('lj')
  """
  world_size = 8
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  mp.spawn(train,
           args=(world_size,data),
           nprocs=world_size,
           join=True)
  """

  train(0, 1, data)
