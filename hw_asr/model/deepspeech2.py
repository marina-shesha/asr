from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           bias=True, batch_first=True)

    def forward(self, x, lengths):

        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, hidden = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x


class MyConv(nn.Module):
    def __init__(self):
        super(MyConv, self).__init__()
        self.conv = Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, input_length):
        for l in self.conv:
            x = l(x)
            if isinstance(l, nn.Conv2d):
                input_length = (input_length + 2 * l.padding[1] - l.dilation[1] * (l.kernel_size[1] - 1) - 1)/l.stride[1] + 1
        return x, input_length.int()


class DeepSpeech(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size, context_size,  **batch):
        super().__init__(n_feats, n_class, **batch)
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.context_size = context_size
        self.conv = MyConv()
        self.input_size_rnn = 32*32

        self.rnn = Sequential(MyRNN(self.input_size_rnn, self.hidden_size),
                               MyRNN(self.hidden_size, self.hidden_size),
                               MyRNN(self.hidden_size, self.hidden_size))

        self.lookahead = Sequential(nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=self.context_size,
            stride=1,
            groups=self.hidden_size,
            padding=self.context_size // 2,
            bias=False),
            nn.ReLU(inplace=True))
        self.layer_linear_norm = nn.BatchNorm1d(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.n_class)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = spectrogram.unsqueeze(dim=1)
        x, lengths = self.conv(x, spectrogram_length)
        batch_size, channels, num_features, len = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch_size, len, channels * num_features)
        for l in self.rnn:
            x = l(x, lengths)
        x = x.permute(0, 2, 1)
        x = self.lookahead(x)
        x = self.layer_linear_norm(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)

        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        for l in self.conv.conv:
            if isinstance(l, nn.Conv2d):
                input_lengths = (input_lengths + 2 * l.padding[1] - l.dilation[1] * (l.kernel_size[1] - 1) - 1) / l.stride[1] + 1
        return input_lengths.int()
