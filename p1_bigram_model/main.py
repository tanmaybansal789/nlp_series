import torch
from torch import nn
from torch.nn import functional as F

# text <-> tokens
class Encoder:
    def __init__(self, text):
        self.decoder = sorted(set(text))
        self.encoder = { c : i for i, c in enumerate(self.decoder) }

    def decode(self, l):
        return ''.join(self.decoder[i] for i in l)

    def encode(self, s):
        return [self.encoder[c] for c in s]

    @property
    def n_vocab(self):
        return len(self.decoder)

        import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import pandas as pd

from dataclasses import dataclass


# text <-> tokens
class Encoder:
    def __init__(self, text):
        self.decoder = sorted(set(text))
        self.encoder = { c : i for i, c in enumerate(self.decoder) }

    def decode(self, l):
        return ''.join(self.decoder[i] for i in l)

    def encode(self, s):
        return [self.encoder[c] for c in s]
    
    @property
    def n_vocab(self):
        return len(self.decoder)


# split training/validation data    
def train_val_split(data, train_frac):
    i = int(len(data) * train_frac)
    return data[:i], data[i:]

# batch training data
def get_batch(data, block_size, batch_size):
    # random offsets into the training data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # input
    xb = torch.stack([data[i : i + block_size] for i in ix])
    # offset by 1 because each token should predict the next one in the sequence
    yb = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]) 
    return xb, yb

# model
class BigramModel(nn.Module):
    def __init__(self, vocab_size, max_block_size=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)
        self.block_size = max_block_size

    def forward(self, idx):
        logits = self.embed(idx)
        return logits

    def generate(self, idx, n_toks=500):
        for _ in range(n_toks):
            logits = self(idx)
            # only get the last element in the time dimension
            logits = logits[:, -1, :]
            # softmax by squishing the channel dimension - for bigram, T == C
            probs = F.softmax(logits, dim=-1)
            # 1 random sample using this generated probability distribution
            idx_next = torch.multinomial(probs, 1)
            # concatenate on the time dimension
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# traing loop
def train_bigram(model, train_data, batch_size=32, n_steps=10000):
    optimiser = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for step in range(n_steps):
        xb, yb = get_batch(train_data, model.block_size, batch_size)

        logits = model(xb)
        B, T, C = logits.shape # (batch size, time dimension, channel)

        logits = logits.view(B*T, C)
        yb = yb.view(B*T)

        loss = criterion(logits, yb)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % 1000 == 0:
            print(f'{step}: loss={loss.item()}')