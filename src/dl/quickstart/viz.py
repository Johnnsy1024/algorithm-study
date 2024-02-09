import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

lstm = nn.LSTM(20, 20, num_layers=2, batch_first=True, bidirectional=True)
x = torch.randn(64, 10, 20)

writer = SummaryWriter("./viz")
