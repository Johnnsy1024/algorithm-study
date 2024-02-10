import torch
import torch.nn as nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class LSTM2LINEAR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=10,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            device=device,
        )
        self.linear1 = nn.Linear(40, 1)
        # self.linear2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2, inplace=True)

    def forward(self, x: torch.tensor):
        x = self.lstm1(x)[1][1]  # (4, batch_size, 10)
        self.dropout(x)
        x = x.reshape((x.shape[1], -1))  # (batch_size, 40)
        # x = self.relu(x)
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)

        return x
