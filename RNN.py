import torch
import torch.nn as nn

class RNN_Block(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.wxh = nn.Linear(input_size, hidden_size)
        self.whh = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x, h):
        hidden = self.tanh(self.wxh(x) + self.whh(h))
        output = self.linear(hidden)
        return output, hidden

class RNN_ManyToOne(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = RNN_Block(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size)
        out = None
        for inp in x:
            out, h = self.rnn(inp, h)

        return out

class RNN_ManyToMany(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = RNN_Block(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size)
        outputs = []
        for inp in x:
            out, h = self.rnn(inp, h)
            outputs.append(out)

        return outputs, h

class RNN_OneToMany(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.rnn = RNN_Block(input_size, hidden_size)

    def forward(self, x):
        batch_size, _, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size)
        outputs = []
        out, h = self.rnn(x, h)
        outputs.append(out)
        for _ in range(1, self.sequence_length):
            out, h = self.rnn(out, h)
            outputs.append(out)

        return outputs, h


