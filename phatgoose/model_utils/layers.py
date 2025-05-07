from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(SelfAttention, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
    
    def forward(self, x):
        return self.attention_layer(x, x, x)[0]

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
    