import torch
import torch.nn as nn

class GRU_Block(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU_Block, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.ux = nn.Linear(input_size, hidden_size)
        self.uh = nn.Linear(hidden_size, hidden_size)

        self.rx = nn.Linear(input_size, hidden_size)
        self.rh = nn.Linear(hidden_size, hidden_size)

        self.ox = nn.Linear(input_size, hidden_size)
        self.oh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        z = self.sigmoid(self.ux(x) + self.uh(h))
        r = self.sigmoid(self.rx(x) + self.rh(h))

        ch = self.tanh(self.ox(x) + r * self.oh(h))

        ht = z * h + (1 - z) * ch

        return ht

class OneToManyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(OneToManyGRU, self).__init__()
        self.gru_block = GRU_Block(input_size, hidden_size)
        self.num_outputs = num_outputs

    def forward(self, x, hidden_state):
        outputs = []
        for _ in range(self.num_outputs):
            output = self.gru_block(x, hidden_state)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

class ManyToOneGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManyToOneGRU, self).__init__()
        self.gru_block = GRU_Block(input_size, hidden_size)

    def forward(self, x, hidden_state):
        sequence_length = x.size(1)
        outputs = []
        for i in range(sequence_length):
            output = self.gru_block(x[:, i, :], hidden_state)
            outputs.append(output)
        return outputs[-1]

class ManyToManyGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManyToManyGRU, self).__init__()
        self.gru_block = GRU_Block(input_size, hidden_size)

    def forward(self, x, hidden_state):
        sequence_length = x.size(1)
        outputs = []
        for i in range(sequence_length):
            output = self.gru_block(x[:, i, :], hidden_state)
            outputs.append(output)
        return torch.stack(outputs, dim=1)
