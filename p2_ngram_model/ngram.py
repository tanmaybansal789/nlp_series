import torch
from torch import nn
from torch.nn import functional as F


# device selection: prefer CUDA, then MPS (Apple Silicon), else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

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
def get_batch(data, n, batch_size):
    block_size = n - 1
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    yb = torch.stack([data[i + block_size] for i in ix]).to(device)
    return xb, yb

# model
class NGramModel(nn.Module):
    def __init__(self, vocab_size, n=3, embed_dim=32):
        super().__init__()
        self.n = n

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear((n - 1) * embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        assert T == self.n - 1, "Input sequence length must be n-1"
        # embed tokens
        x = self.token_embedding(idx)     # (B, T, E)
        # concatenate context tokens
        x = x.view(B, T * x.size(-1))     # (B, (n-1)*E)
        logits = self.fc(x)               # (B, vocab)
        return logits

    def generate(self, idx, n_toks=500):
        # assume that idx at least has n-1 tokens
        for _ in range(n_toks):
            # we will extract the last 'n - 1' tokens
            logits = self(idx[:, -self.n + 1:]) # (B, T)
            # softmax across the vocab dimension
            probs = F.softmax(logits, dim=1)
            # same as for bigram
            idx_next = torch.multinomial(probs, 1)
            # concatenate on the time dimension
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# traing loop
def train_ngram(model, train_data, batch_size=32, n_steps=10_000):
    optimiser = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for step in range(n_steps):
        xb, yb = get_batch(train_data, model.n, batch_size)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % 100 == 0:
            print(f'{step}: loss={loss.item()}')

# entry point
if __name__ == '__main__':
    # load text and build encoder
    should_train = False

    input_path = 'p2_ngram_model/code_input.txt'
    model_path = 'p2_ngram_model/code_ngram_model_tokenised.pt'

    text = open(input_path).read()
    # encoder = Encoder(text)
    import tiktoken
    encoder = tiktoken.get_encoding('gpt2')

    data = torch.tensor(encoder.encode(text), dtype=torch.long).to(device)
    train_data, val_data = train_val_split(data, 0.9)

    model = NGramModel(encoder.n_vocab, n=5).to(device)
    if should_train:
        train_ngram(model, train_data, n_steps=90_000)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    print(encoder.decode((model.generate(torch.zeros((1, model.n - 1), dtype=torch.long, device=device), n_toks=1000).tolist())[0]))