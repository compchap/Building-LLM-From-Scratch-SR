import torch.nn as nn
import torch

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
