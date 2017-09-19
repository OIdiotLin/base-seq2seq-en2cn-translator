import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from seq2seq import use_cuda


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )

    def forward(self, x, h_state):
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, h_state = self.gru(output, h_state)
        return output, h_state

    def init_hidden(self):
        out = Variable(torch.zeros(1, 1, self.hidden_size))
        return out.cuda() if use_cuda else out


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, maxlen=35):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_len = maxlen

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size*2, self.max_len)
        self.attention_comb = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
        )
        self.out = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size,
        )

    def forward(self, x, h_state, encoder_output, encoder_outputs):
        embedded = self.dropout(self.embedding(x).view(1, 1, -1))
        attention_w = F.softmax(self.attention(torch.cat((embedded[0], h_state[0]), 1)))

        attention_applied = torch.bmm(attention_w.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = self.attention_comb(
            torch.cat((embedded[0], attention_applied[0]), 1)
        ).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, h_state = self.gru(output, h_state)

        output = F.log_softmax(self.out(output[0]))
        return output, h_state, attention_w

    def init_hidden(self):
        out = Variable(torch.zeros(1, 1, self.hidden_size))
        return out.cuda() if use_cuda else out

