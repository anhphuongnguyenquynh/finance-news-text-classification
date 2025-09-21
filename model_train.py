import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN text classification model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size = 3, stride=1, padding=1)
        self.fc = nn.Linear(embed_dim, num_classes) # 42 classes
    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)    # (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim, seq_len)
        conved = F.relu(self.conv(embedded))                # (batch_size, embed_dim, seq_len)
        conved = conved.mean(dim=2)                         # (batch_size, embed_dim)
        return self.fc(conved)
    
#LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # convert token IDs → embeddings
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)   # [batch, seq_len] → [batch, seq_len, embed_size]
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))  # [batch, hidden] → [batch, output_size]
        return output
    
#GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden.squeeze(0))
        return output
    
#BiLSTMModel
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)         # *2 for bidirectional output
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text):
        # text = [batch size, seq len]
        embedded = self.dropout(self.embedding(text))           # embedded = [batch size, seq len, embedding dim]
        output, (hidden, cell) = self.lstm(embedded)
                                                                # output = [batch size, seq len, hidden dim * 2]
                                                                # hidden = [num layers * 2, batch size, hidden dim]
                                                                # cell = [num layers * 2, batch size, hidden dim]

        # Use the concatenated final hidden state from both directions
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))   # hidden = [batch size, hidden dim * 2]

        return self.fc(hidden)
    

if __name__ == "__main__":

    print(CNNModel(1000, 128, 42))

    print(GRUModel(128, 256, 42))

    print(LSTMModel(1000, 128, 256, 42))

    print(BiLSTM(1000, 128, 256, 42, 2, 0.3))