import math
import torch
import torch.nn as nn
import razdel
from torch.nn import functional as F


class Config:
    n_head = 12
    n_embd = 768
    n_layer = 12
    seq_len = 32
    embd_pdrop = 0.5
    resid_pdrop = 0.5
    attn_pdrop = 0.5
    vocab_size = 1024


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        assert Config.n_embd % Config.n_head == 0, "n_embd должен быть кратен n_head"
        # key, query, value projections for all heads
        self.key = nn.Linear(Config.n_embd, Config.n_embd)
        self.query = nn.Linear(Config.n_embd, Config.n_embd)
        self.value = nn.Linear(Config.n_embd, Config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(Config.attn_pdrop)
        self.resid_drop = nn.Dropout(Config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, Config.n_head, C // Config.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, Config.n_head, C // Config.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, Config.n_head, C // Config.n_head).transpose(1, 2)

        # casual self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        # nn.BatchNorm1d заменен на nn.LayerNorm тк была ошибка размерности
        self.norm1 = nn.LayerNorm(Config.n_embd)
        self.norm2 = nn.LayerNorm(Config.n_embd)
        self.attn = Attention()
        self.mlp = nn.Sequential(
            nn.Linear(Config.n_embd, Config.n_embd // 16),
            nn.Linear(Config.n_embd // 16, Config.n_embd),
            nn.Dropout(Config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Языковая модель
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # nn.Linear заменен на nn.Embedding тк была ошибка типов
        # input embedding stem
        self.tok_emb = nn.Embedding(Config.vocab_size, Config.n_embd)
        # transformer
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(Config.n_layer)])
        # decoder head
        self.norm_f = nn.LayerNorm(Config.n_embd)  
        self.head = nn.Linear(Config.n_embd, Config.vocab_size, bias=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.zero_()
            module.bias.data.zero_()

    def forward(self, idx, targets):
        b, t = idx.size()

        assert t <= Config.seq_len, "Cannot forward, model block size is exhausted."

        x = self.tok_emb(idx)
        x = self.blocks(x)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction='none')
        return logits, loss


lm = LanguageModel()
vocab = ["мама", "компьютер", "мыла", "раму", "текст", "сгенерировал", "длинный"] 
batch = ["Мама мыла раму", "Компьютер сгенерировал длинный текст"]  

PAD = 0
BOS = 1
EOS = 2
UNK = 3
token2idx = {token: idx for idx, token in enumerate(vocab)}

# вместо hard-coded значений использованы константные
batch = [[token2idx.get(token, UNK) for token in razdel.tokenize(text.lower())] for text in batch]
MAX_LEN = Config.seq_len
batch_input = torch.zeros((len(batch), MAX_LEN), dtype=torch.long)

# padleft each sample with PAD to MAX_LEN
for i, row in enumerate(batch):
    row = torch.tensor(row)
    length = len(row)
    batch_input[i, -length - 2] = BOS
    batch_input[i, -length - 2:-2] = row
    batch_input[i, -1] = EOS

targets = batch_input.clone()
logits, loss = lm(batch_input, targets)
print(loss.shape)
loss.mean().backward()
print(logits, loss)
