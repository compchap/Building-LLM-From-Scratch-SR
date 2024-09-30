import torch
import tiktoken
import torch.nn as nn

# Create dataset and dataloader that extract chunks from the input text dataset
import torch
from torch.utils.data import Dataset, DataLoader
 
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # Creating tokens form text and generating token ids
        token_ids = tokenizer.encode(txt)
 
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
 
    def __len__(self):
        return len(self.input_ids)
 
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
        stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
 
    return dataloader

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
 
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # print("W_q_weights: ", self.W_query.state_dict()['weight'])

        # output projection layer
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            'mask',
             torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
 
    def forward(self, x):

        # b is the batch size.
        # num_tokens is the number of tokens in each sequence.
        # d_in is the dimensionality of each token's embedding.
        
        
        b, num_tokens, d_in = x.shape
        # print(f"b: {b} | num_tokens: {num_tokens} | d_in: {d_in}")
        # Unpacks the shape of the input tensor x. b is the batch size, num_tokens is the number of tokens, 
        # and d_in is the input dimensionality (which should match the d_in given during initialization).

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # reshaping (b, num_tokens, self.d_out) to (b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # bring num_heads dimensions before num_tokens dimension
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Computes the attention scores by performing a batch matrix multiplication of queries and the transpose of keys. 
        # This results in a score matrix of shape (b, num_tokens, num_tokens), 
        # where each score represents the similarity between a query and a key. 
        # The transposition ensures that each query compares against every key.
        attn_scores = queries @ keys.transpose(2, 3)

        # Applies the causal mask to the attention scores. 
        # The mask is of shape (context_length, context_length) but is sliced to match the number of tokens in the input sequence. 
        # This mask ensures that positions can only attend to previous positions (not future ones). 
        # The use of -torch.inf effectively nullifies the attention to masked positions, 
        # making them irrelevant when computing the softmax.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalizes the attention scores using the softmax function. 
        # The division by the square root of the key dimension (keys.shape[-1]**0.5) is a scaling factor 
        # that helps stabilize the gradients and improves convergence (a common practice in attention mechanisms). 
        # The result is a set of attention weights that sum to 1 across each row.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Applies dropout to the attention weights to prevent overfitting. 
        # This randomly zeros some of the attention weights during training, 
        # encouraging the model to not rely on any single position too much.
        attn_weights = self.dropout(attn_weights)

        # Computes the context vectors by performing a weighted sum of the value vectors, using the attention weights as coefficients. 
        # This results in a new representation of each token, informed by the relevant tokens before it.
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Returns the context-aware output of shape (b, num_tokens, d_out), 
        # where each token has been processed to attend to previous relevant tokens according to the causal attention mechanism.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
 
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
 
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
 
    def forward(self, x):

        shortcut = x
        x = self.norm1(x) # Layer normalization (Pre-LayerNorm)
        x = self.att(x) # multi-head attention mechanism
        x = self.drop_shortcut(x) # dropout is applied to regularize the model and prevent overfitting
        x = x + shortcut  # helps gradients flow through the network during training and improves the learning of deep models
 
        shortcut = x
        x = self.norm2(x) # Layer normalization (Pre-LayerNorm)
        x = self.ff(x)
        x = self.drop_shortcut(x) # dropout is applied to regularize the model and prevent overfitting
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) # Create Sequential stack of Transformer blocks
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
 
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
       
        logits = logits[:, -1, :] # to select the predictions for the last token in each sequence across all batches.
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
 
    return idx