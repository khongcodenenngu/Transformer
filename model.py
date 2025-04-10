import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(vocab_size, model_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.model_dim)

class PositionalEncoding(nn.Module): # sinusoidal pos encoding
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0, "d_model must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads
        self.Wq = nn.Linear(model_dim, model_dim)
        self.Wk = nn.Linear(model_dim, model_dim)
        self.Wv = nn.Linear(model_dim, model_dim)
        self.Wo = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def scaled_dotproduct_attention(self, Q, K, V, mask=None): # query, key, value
        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (batch_size, num_heads, seq_length, seq_length)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0, -1e9)
        attention_probs = self.dropout(torch.softmax(attention_scores, dim=-1))
        output = torch.matmul(attention_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, model_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim) -> for computational efficiency along all heads

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dim) # transpose'll make the tensor uncontiguous, we have to contiguous() it to use view()

    def forward(self, Q, K, V, mask=None): # Q/K/V in input are all input embeds
        Q = self.split_heads(self.Wq(Q))
        K = self.split_heads(self.Wk(K))
        V = self.split_heads(self.Wv(V))

        attention_output = self.scaled_dotproduct_attention(Q, K, V, mask)
        output = self.Wo(self.combine_heads(attention_output))
        return output

class FeedForward(nn.Module): # Position-wise Feed-Forward
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attention_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x

class Encoder(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.cross_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attention_output))
        attention_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attention_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x

class Decoder(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim, num_heads, ff_dim, num_layers, max_seq_length, dropout=0.1):
        super().__init__()
        self.src_embedding = InputEmbedding(src_vocab_size, model_dim)
        self.tgt_embedding = InputEmbedding(tgt_vocab_size, model_dim)
        self.src_positional_encoding = PositionalEncoding(model_dim, max_seq_length)
        self.tgt_positional_encoding = PositionalEncoding(model_dim, max_seq_length)
        self.encoder = Encoder(model_dim, num_heads, ff_dim, num_layers, dropout)
        self.decoder = Decoder(model_dim, num_heads, ff_dim, num_layers, dropout)
        self.projection_layer = nn.Linear(model_dim, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encode
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src)
        encoder_output = self.encoder(src, src_mask)

        # decode
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        logits = self.projection_layer(decoder_output)
        return logits







