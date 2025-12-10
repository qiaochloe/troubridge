
import torch.nn as nn
from create_tokenizer import create_vocab, tokenize

class HotnessPredictor(nn.Module):
    def __init__(self, embedding_dim=32, hidden_size=64, num_layers=1, num_categories=2):
        super().__init__()
        self.vocab = create_vocab()
        self.embedding = nn.Embedding(len(self.vocab), embedding_dim=embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_categories)
        self.softmax = nn.Softmax(dim=0)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        _, x = self.gru(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
    def train(self, x, y):
        x = self.forward(x)
        loss = self.loss_fn(x, y)
        return loss

def train_one_epoch