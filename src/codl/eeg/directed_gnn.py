"""Small directed polynomial GNN pilot; encode() exposes 256 EEG features."""
import math
import torch
from torch import nn


class DirectedCheb(nn.Module):
    def __init__(self, incoming, outgoing, terms=5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(terms, incoming, outgoing))
        self.bias = nn.Parameter(torch.zeros(outgoing))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, lap):
        # lap[target, source]: multiplying propagates source to target.
        t0 = x
        out = t0 @ self.weight[0]
        if len(self.weight) > 1:
            t1 = lap @ x
            out = out + t1 @ self.weight[1]
            for k in range(2, len(self.weight)):
                t2 = 2 * (lap @ t1) - t0
                out = out + t2 @ self.weight[k]
                t0, t1 = t1, t2
        return out + self.bias


class DirectedGNN(nn.Module):
    def __init__(self, settings):
        super().__init__()
        dims = [200] + settings['hidden_dims']
        self.convs = nn.ModuleList([DirectedCheb(a, b, settings['cheb_terms']) for a,b in zip(dims[:-1], dims[1:])])
        self.norms = nn.ModuleList([nn.BatchNorm1d(d) for d in dims[1:]])
        self.drop = nn.Dropout(settings['dropout'])
        d = dims[-1]
        self.qkv = nn.Linear(d, 3*d)
        self.attdrop = nn.Dropout(settings['attention_dropout'])
        self.ff = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(settings['attention_dropout']), nn.Linear(64,d))
        self.norm = nn.LayerNorm(d)
        self.project = nn.Linear(19*d, settings['head_hidden'])
        self.head = nn.Linear(settings['head_hidden'], 1)

    def encode(self, x, lap):
        x = x / 100.0  # fixed unit conversion; not a fitted or LEAD normalization
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            old = x
            x = norm(conv(x, lap).transpose(1,2)).transpose(1,2)
            if i in (1,4,7):
                x = x + old
            x = self.drop(torch.relu(x))
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        attention = torch.softmax(q @ k.transpose(-1,-2) / math.sqrt(q.shape[-1]), dim=-1)
        a = self.attdrop(attention) @ v
        x = self.norm(x + a + self.ff(a))
        return torch.relu(self.project(x.flatten(1)))

    def forward(self, x, lap):
        return self.head(self.drop(self.encode(x, lap))).squeeze(-1)
