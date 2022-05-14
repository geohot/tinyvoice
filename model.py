#!/usr/bin/env python3
import time
import torch
from torch import nn
from preprocess import CHARSET
import torch.optim as optim

class ResBlock(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(c, c, 3, padding='same'),
      nn.BatchNorm2d(c),
      nn.ReLU(c),
      nn.Conv2d(c, c, 3, padding='same'),
      nn.BatchNorm2d(c))
  
  def forward(self, x):
    return nn.functional.relu(x + self.block(x))

class TemporalBatchNorm(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.bn = nn.BatchNorm1d(channels)
  def forward(self, x):
    return self.bn(x.permute(1,2,0)).permute(2,0,1)

class Rec(nn.Module):
  def __init__(self):
    super().__init__()

    C, H = 16, 256
    self.encode = nn.Sequential(
      nn.Conv2d(1, C, 1, stride=2),
      ResBlock(C),
      ResBlock(C),
      nn.Conv2d(C, C, 1, stride=2),
      ResBlock(C),
      ResBlock(C),
    )
    self.flatten = nn.Linear(320, H)

    self.gru = nn.GRU(H, H, batch_first=False)
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
    # (time, batch, freq)
    #print(x.shape)
    x = x[:, None] # (batch, time, freq) -> (batch, 1, time, freq)
    # (batch, C, H, W)
    x = self.encode(x).permute(2, 0, 1, 3) # (H, batch, C, W)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = self.flatten(x)
    x = self.gru(x)[0]
    x = self.decode(x)
    return torch.nn.functional.log_softmax(x, dim=2)

if __name__ == "__main__":
  batch_size = 32
  learning_rate = 0.002

  target_length = 200
  target = [1]*(batch_size*target_length)
  input_lengths = [999//4]*batch_size
  target_lengths = [target_length]*batch_size

  ctc_loss = nn.CTCLoss().cuda()
  model = Rec().cuda()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  input = torch.zeros(1000, batch_size, 80, device='cuda:0')
  target = torch.tensor(target, dtype=torch.int32, device='cuda:0')

  for i in range(10):
    st = time.monotonic()
    optimizer.zero_grad()
    guess = model(input)
    #print(guess.shape, guess.device, guess.dtype)
    #print(target.shape, target.device, target.dtype)

    loss = ctc_loss(guess, target, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()
    et = time.monotonic() - st
    print(f"{et*1000:.2f} ms  {1/et:.2f} its/sec")


  #exit(0)