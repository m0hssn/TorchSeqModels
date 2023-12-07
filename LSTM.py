import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_gate_i = nn.Linear(input_size, hidden_size)
        self.input_gate_h = nn.Linear(hidden_size, hidden_size)

        self.forget_gate_i = nn.Linear(input_size, hidden_size)
        self.forget_gate_h = nn.Linear(hidden_size, hidden_size)

        self.update_gate_i = nn.Linear(input_size, hidden_size)
        self.update_gate_h = nn.Linear(hidden_size, hidden_size)

        self.output_gate_i = nn.Linear(input_size, hidden_size)
        self.output_gate_h = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, hidden):
        h_t, c_t = hidden

        # Input gate
        i_t = self.sigmoid(self.input_gate_i(x) + self.input_gate_h(h_t))

        # Forget gate
        f_t = self.sigmoid(self.forget_gate_i(x) + self.forget_gate_h(h_t))

        # Update cell
        g_t = self.tanh(self.update_gate_i(x) + self.update_gate_h(h_t))

        # New cell state
        c_t = f_t * c_t + i_t * g_t

        # Output gate
        o_t = self.sigmoid(self.output_gate_i(x) + self.output_gate_h(h_t))

        h_t = o_t * self.tanh(c_t)

        return h_t, (h_t, c_t)

class ManyToManyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = LSTMBlock(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size)  # Initial hidden state
        c_t = torch.zeros(batch_size, self.hidden_size)  # Initial cell state
        output_seq = []

        for i in range(seq_len):
            h_t, (h_t, c_t) = self.lstm(x[:, i, :], (h_t, c_t))
            output_seq.append(self.output_layer(h_t))

        output_seq = torch.stack(output_seq, dim=1)
        return output_seq

class ManyToOneLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = LSTMBlock(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)

        for i in range(seq_len):
            h_t, (h_t, c_t) = self.lstm(x[:, i, :], (h_t, c_t))

        output = self.output_layer(h_t)
        return output

class OneToManyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super().__init__()
        self.lstm = LSTMBlock(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.seq_length = seq_length
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)
        output_seq = []

        for i in range(self.seq_length):
            h_t, (h_t, c_t) = self.lstm(x, (h_t, c_t))
            output_seq.append(self.output_layer(h_t))
            x = self.output_layer(h_t)

        output_seq = torch.stack(output_seq, dim=1)
        return output_seq

