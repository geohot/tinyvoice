#!/usr/bin/env python3
import time
import torch
from torch import nn
import torch.nn.functional as F
from preprocess import CHARSET
import torch.optim as optim
from torchaudio.models import Conformer

class ResBlock(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.block = nn.Sequential(
      # depthwise
      nn.Conv2d(c, c, 3, groups=c, padding='same', bias=False),
      nn.BatchNorm2d(c),
      nn.ReLU(c),
      # project
      nn.Conv2d(c, c, 1, bias=False),
      nn.BatchNorm2d(c))
  
  def forward(self, x):
    return nn.functional.relu(x + self.block(x))

class Rec(nn.Module):
  def __init__(self):
    super().__init__()

    """
    C, H = 16, 320
    self.encode = nn.Sequential(
      nn.Conv2d(1, C, 1, stride=2, bias=False),
      nn.BatchNorm2d(C),
      nn.ReLU(),
      ResBlock(C),
      ResBlock(C),
      nn.Conv2d(C, C, 1, stride=2, bias=False),
      nn.BatchNorm2d(C),
      nn.ReLU(),
      ResBlock(C),
      ResBlock(C),
    )
    self.gru = nn.GRU(H, H, batch_first=False)
    """
    self.conformer = Conformer(80, 4, 128, 4, 31)
    H = 80
    self.decode = nn.Sequential(
      nn.Linear(H, H//2),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(H//2, len(CHARSET))
    )

  def forward(self, x, y):
    # (time, batch, freq)
    #print(x.shape)
    """
    x = x[:, None] # (batch, time, freq) -> (batch, 1, time, freq)
    # (batch, C, H, W)
    x = self.encode(x).permute(2, 0, 1, 3) # (H, batch, C, W)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = self.gru(x)[0]
    x = self.decode(x)
    """
    x = self.conformer(x,y)[0].permute(1,0,2)
    x = self.decode(x)
    return torch.nn.functional.log_softmax(x, dim=2)


if __name__ == "__main__":
  import torch.autograd.profiler as profiler

  batch_size = 32
  learning_rate = 0.001

  target_length = 200
  target = [1]*(batch_size*target_length)
  input_lengths = [1600]*batch_size
  input_lengths[0] -= 1
  target_lengths = [target_length]*batch_size

  input_lengths = torch.tensor(input_lengths).cuda()
  target_lengths = torch.tensor(target_lengths).cuda()

  #ctc_loss = nn.CTCLoss().cuda()
  #model = Conformer(80, 4, 128, 4, 31).cuda()
  model = Rec().cuda()
  #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  import apex
  optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=learning_rate)

  input = torch.zeros(batch_size, 1600, 80, device='cuda:0', dtype=torch.float32)
  #target = torch.ones(batch_size*target_length, device='cuda:0', dtype=torch.int32)
  target = torch.ones(batch_size*target_length, dtype=torch.int32)

  def run_model():
    optimizer.zero_grad()
    guess = model(input, input_lengths)
    #print(guess.shape)
    loss = F.ctc_loss(guess, target, input_lengths, target_lengths)
    #loss = guess.mean()
    loss.backward()
    optimizer.step()
    return loss
  run_model()

  GRAPH = False
  if GRAPH:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
      for _ in range(10):
        loss = run_model()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
      loss = run_model()

  #with profiler.profile(with_stack=True, profile_memory=True) as prof:
  for i in range(30):
    st = time.monotonic()
    if GRAPH:
      g.replay()
    else:
      loss = run_model()
    rloss = loss.item()
    et = time.monotonic() - st
    print(f"{et*1000:.2f} ms  {1/et:.2f} its/sec {rloss}")

  #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))



  #exit(0)