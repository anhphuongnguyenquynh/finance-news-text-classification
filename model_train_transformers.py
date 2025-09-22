import torch
import torch.nn as nn
import torch.nn.functional as F
    
#Tranformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_len, dropout):
        super(TransformerModel, self).__init__()

        # Embedding + Positional Encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # makes input shape (batch, seq, embed)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand_as(x)

        x = self.embedding(x) + self.pos_embedding(positions)  # (batch, seq_len, embed)
        x = self.transformer_encoder(x)  # (batch, seq_len, embed)

        # Take [CLS]-like representation (mean pooling or first token)
        x = x.mean(dim=1)  
        out = self.fc(x)  # (batch, num_classes)
        return out

    
if __name__ == "__main__":
    
    # Example usage
    #input_test[0] = torch.Size([64,12])
    #input_test[1] = torch.Size([64])
    #output_test = torch.Size([64,27])
    input = torch.randn(64, 12)  # (batch_size, seq_length, embed_dim)

    label = torch.rand(64)
    print(TransformerModel(30522, 128, 8, 512, 2, 27, max_len=512, dropout=0.1))
