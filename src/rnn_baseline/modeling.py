import torch
from torch import nn
from torch.nn import functional as F

from scene_layouts.datasets import X_PAD, Y_PAD, O_PAD


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(
            hidden_size, hidden_size, bidirectional=True, batch_first=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)

        return outputs, hidden


class TextAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_v = nn.Linear(hidden_size * 2, hidden_size)
        self.W_h = nn.Linear(hidden_size * 2, hidden_size)
        self.W_u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        # [k, m]
        o_v = self.W_v(encoder_outputs)
        # [1, m]
        o_h = self.W_h(hidden_state)
        # [k, m]
        f = torch.tanh(o_v + o_h.unsqueeze(1))
        # [k, 1]
        o_attn = self.W_u(f)
        # [k, 1]
        a = F.softmax(o_attn, dim=1)
        # [k]
        z = torch.sum(encoder_outputs * a, dim=1)

        return z


class ArrangementsContinuousDecoder(nn.Module):
    def __init__(self, num_cliparts, vocab_size, hidden_size, device):
        super().__init__()
        self.clip_embed = nn.Embedding(num_cliparts, hidden_size)
        self.clip_rnn = nn.GRU(
            hidden_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.text_enc = TextEncoder(vocab_size, hidden_size)
        self.text_attn = TextAttention(hidden_size)
        self.xy_head = nn.Linear(hidden_size * 4, 2)
        self.o_head = nn.Linear(hidden_size * 4, O_PAD - 1)
        self.device = device
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, text_inds, clip_inds):
        text_enc, _ = self.text_enc(text_inds)
        clip_enc, _ = self.clip_rnn(self.clip_embed(clip_inds))
        x_outs = torch.zeros(clip_inds.size()[0], clip_inds.size()[1]).to(self.device)
        y_outs = torch.zeros(clip_inds.size()[0], clip_inds.size()[1]).to(self.device)
        o_outs = torch.zeros(clip_inds.size()[0], clip_inds.size()[1], 2).to(
            self.device
        )
        for i in range(clip_enc.size()[1]):
            attn = self.text_attn(clip_enc[:, i, :], text_enc)
            hidden = torch.cat([clip_enc[:, i, :], attn], dim=-1).to(self.device)
            x_outs[:, i] = torch.sigmoid(self.xy_head(hidden))[:, 0] * (X_PAD - 2)
            y_outs[:, i] = torch.sigmoid(self.xy_head(hidden))[:, 1] * (Y_PAD - 2)
            o_outs[:, i, :] = self.log_softmax(self.o_head(hidden))

        return x_outs, y_outs, o_outs


class ArrangementsDiscreteDecoder(nn.Module):
    def __init__(self, num_cliparts, vocab_size, hidden_size, device):
        super().__init__()
        self.clip_embed = nn.Embedding(num_cliparts, hidden_size)
        self.clip_rnn = nn.GRU(
            hidden_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.text_enc = TextEncoder(vocab_size, hidden_size)
        self.text_attn = TextAttention(hidden_size)
        self.x_head = nn.Linear(hidden_size * 4, X_PAD - 1)
        self.y_head = nn.Linear(hidden_size * 4, Y_PAD - 1)
        self.o_head = nn.Linear(hidden_size * 4, O_PAD - 1)
        self.device = device
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, text_inds, clip_inds):
        text_enc, _ = self.text_enc(text_inds)
        clip_enc, _ = self.clip_rnn(self.clip_embed(clip_inds))
        x_outs = torch.zeros(clip_inds.size()[0], clip_inds.size()[1], X_PAD - 1).to(
            self.device
        )
        y_outs = torch.zeros(clip_inds.size()[0], clip_inds.size()[1], Y_PAD - 1).to(
            self.device
        )
        o_outs = torch.zeros(clip_inds.size()[0], clip_inds.size()[1], O_PAD - 1).to(
            self.device
        )
        for i in range(clip_enc.size()[1]):
            attn = self.text_attn(clip_enc[:, i, :], text_enc)
            hidden = torch.cat([clip_enc[:, i, :], attn], dim=-1).to(self.device)
            x_outs[:, i, :] = self.log_softmax(self.x_head(hidden))
            y_outs[:, i, :] = self.log_softmax(self.y_head(hidden))
            o_outs[:, i, :] = self.log_softmax(self.o_head(hidden))

        return x_outs, y_outs, o_outs
