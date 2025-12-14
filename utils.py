import torch
import torch.nn as nn

###########################################################################################################################

class Softmax(nn.Module):
    def __init__(self, dim=-1): # dim : A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        super().__init__()
        self.dim = dim

    def forward(self, x, dim=None):
        
        # Determine which dimension to use
        target_dim = dim if dim is not None else self.dim
        
        # 1. Find the max along the specific dimension for numerical stability
        # keepdim=True is needed to allow broadcasting during subtraction
        max_val, _ = torch.max(x, dim=target_dim, keepdim=True)
        
        # 2. Subtract the max (Shift property)
        # This prevents e^x from overflowing if x is large
        shifted_x = x - max_val
        
        # 3. Compute exponentials
        exp_x = torch.exp(shifted_x)
        
        # 4. Compute the sum of exponentials along the dimension
        sum_exp_x = torch.sum(exp_x, dim=target_dim, keepdim=True)
        
        # 5. Divide to normalize (resulting sum is 1.0)
        softmax_output = exp_x / sum_exp_x
     
        return softmax_output

################################################################################################################

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        
        # mask ~ Bernoulli(1-p)
        mask = (torch.rand_like(x) > self.p).float()  # Random mask of 0s and 1s, if the random number > p, then 1 else 0
        # torch.rand_like() returns a tensor with the same size as input that is 
        # filled with random numbers from a "uniform distribution" on the interval [0,1).
        # torch.rand_like(input) === torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)

        # scale
        return x * mask / (1 - self.p)

################################################################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out # output dimesion
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = Dropout(dropout)
        # self.softmax=Softmax()
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape # _ = d_in

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Dot product for each head 
        attn_scores = queries @ keys.transpose(2, 3)  # [b, num_heads, num_tokens, head_dim] @ [b, num_heads, head_dim, num_tokens] -> [b, num_heads, num_tokens, num_tokens]

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # self.mask is upper triangular matrix where above the triangle every elements are '1'. 
        # By converting it to boolean using mask.bool(), we get 'True' for 1s and 'False' for 0s.

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf) # where there the indices are True/1 in mask matrix (mask_bool), replace these indices of attn_scores with -inf
        

        # attn_weights = self.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # better for efficiency
   
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        context_vec = self.out_proj(context_vec) # optional projection
        # Why projection layer needed : 
        # If we want to change the output dimension to match a specific size required by the model architecture, 
        # we use this projection layer, a Linear layer (nural net layer) that combines the outputs from all heads 
        # into a single vector of the desired dimension.

        return context_vec
    
#############################################################################################################################################

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # variance
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
#############################################################################################################################################

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

#############################################################################################################################################

class FeedForward(nn.Module):
    def __init__(self, model_config): # model_config = Model configuration/configuration dictionary is just a Python dictionary (dict) 
                             # that stores model hyperparameters. Here we are passing "GPT_CONFIG_124M"
        '''Simplified GPT-2 style model_config (configuration dictionary
        model_config = {
            "vocab_size": 50257,
            "ctx_len": 1024,        # max sequence length
            "emb_dim": 768,         # hidden size
            "n_layers": 12,         # number of transformer blocks
            "n_heads": 12,          # number of attention heads
            "ff_mult": 4,           # feed-forward multiplier (4× embedding)
            "dropout": 0.1,
        }'''


        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(model_config["emb_dim"], 4 * model_config["emb_dim"]),
            GELU(),
            nn.Linear(4 * model_config["emb_dim"], model_config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    
############################################################################################################

class TransformerBlock(nn.Module):
    '''Simplified GPT-2 style model_config (configuration dictionary
        model_config = {
            "vocab_size": 50257,
            "ctx_len": 1024,        # max sequence length
            "emb_dim": 768,         # hidden size
            "n_layers": 12,         # number of transformer blocks
            "n_heads": 12,          # number of attention heads
            "ff_mult": 4,           # feed-forward multiplier (4x embedding)
            "dropout": 0.1,
        }'''
    def __init__(self, model_config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=model_config["emb_dim"],
            d_out=model_config["emb_dim"],
            context_length=model_config["context_length"],
            num_heads=model_config["n_heads"], 
            dropout=model_config["drop_rate"],
            qkv_bias=model_config["qkv_bias"])
        self.ff = FeedForward(model_config)
        self.norm1 = LayerNorm(model_config["emb_dim"])
        self.norm2 = LayerNorm(model_config["emb_dim"])
        self.drop_shortcut = Dropout(model_config["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x) # LayerNorm1
        x = self.att(x)  # Maked Multi-head Attention. # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x) # Dropout Layer
        x = x + shortcut  # Shortcut Connection. Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x) # LayerNorm2
        x = self.ff(x) # Feed Forward
        x = self.drop_shortcut(x) # Dropout Layer
        x = x + shortcut  # Shortcut Connection. Add the original input back

        return x



####################################################################################################################

class GPTModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.tok_emb = nn.Embedding(model_config["vocab_size"], model_config["emb_dim"])
        self.pos_emb = nn.Embedding(model_config["context_length"], model_config["emb_dim"])
        # nn.Embedding stores a matrix of shape (num_embeddings, embedding_dim).
        # When you pass in token IDs, it simply looks up the corresponding rows (each row contains word embedding for the corresponding word).
        # It's essentially a learnable matrix (we will upadte the word/possitional/attention embedding through training) that maps discrete tokens (like word IDs) to dense vectors.
        
        
        self.drop_emb = Dropout(model_config["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(model_config) for _ in range(model_config["n_layers"])]) # There are 12 Transformer blocks in GPT2
        
        self.final_norm = LayerNorm(model_config["emb_dim"])
        self.out_head = nn.Linear(
            model_config["emb_dim"], model_config["vocab_size"], bias=False,
        )

    def forward(self, in_idx): # in_idx = input
        _, seq_len = in_idx.shape #   _ = batch_size
        
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        
        x = self.trf_blocks(x) # Here in GPT-2, there are 12 Transformer blocks
        
        x = self.final_norm(x)
        logits = self.out_head(x) # batch, sequence, vocab
        
        return logits 
    
    
    
    
    
######################################################################################################################

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # For-loop: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
        
    return idx