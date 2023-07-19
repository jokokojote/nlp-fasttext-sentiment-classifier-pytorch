import torch
import torch.nn.functional as F

class FastTextClassifier(torch.nn.Module): 

    def __init__(self, vocab_size: int, hidden_size: int, num_labels: int):
        super(FastTextClassifier, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, hidden_size) # expects indices not sparse one hot vectors
        self.linear = torch.nn.Linear(hidden_size, num_labels)


    def forward(self, one_hot_sentence):
        embeds = self.embedding(one_hot_sentence)
        pooled = torch.mean(embeds, 1)
        out = self.linear(pooled)

        return F.log_softmax(out, dim=1)
