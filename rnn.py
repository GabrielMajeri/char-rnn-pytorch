import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.W_xh = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.W_hy = nn.Parameter(torch.zeros(hidden_size, output_size))

        nn.init.xavier_normal_(self.W_xh)
        nn.init.xavier_normal_(self.W_hh)
        nn.init.xavier_normal_(self.W_hy)

    def forward(self, x, hidden_state):
        hidden_state = torch.tanh(hidden_state @ self.W_hh + x @ self.W_xh)
        y = hidden_state @ self.W_hy
        return y, hidden_state
