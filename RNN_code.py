import torch
import torch.nn as nn

# Vocabulary
word_to_idx = {
    "I": 0,
    "love": 1,
    "learning": 2,
    "Python": 3,
    "daily": 4
}

# Input: I love learning Python
inputs = torch.tensor([[0, 1, 2, 3]])

# Target: love learning Python daily
targets = torch.tensor([[1, 2, 3, 4]])

class SimpleRNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=4
        )

        self.rnn = nn.RNN(
            input_size=4,
            hidden_size=3,
            batch_first=True
        )

        self.fc = nn.Linear(
            3,
            5
        )

    def forward(self, x):

        x = self.embedding(x)

        output, hidden = self.rnn(x)

        output = self.fc(output)

        return output

model = SimpleRNN()

prediction = model(inputs)

print(prediction.shape)