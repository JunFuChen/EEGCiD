
import torch
import torch.nn as nn
import numpy as np
import math



class TokendeEmbedding(nn.Module):
    def __init__(self, d_model, input_size):
        super(TokendeEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=d_model, out_channels=input_size,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
    
class qkvEmbedd(nn.Module):
    def __init__(self, d_model):
        super(qkvEmbedd, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv1_q = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=1, bias=False)
        self.tokenConv2_q = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv1_k = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=1, bias=False)
        self.tokenConv2_k = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv1_v = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=1, bias=False)
        self.tokenConv2_v = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(device=x.device)

        q = self.tokenConv1_q(x.permute(0, 2, 1))
        q = self.tokenConv2_q(q).transpose(1, 2)
        k = self.tokenConv1_k(x.permute(0, 2, 1))
        k = self.tokenConv2_k(k).transpose(1, 2)
        v = self.tokenConv1_v(x.permute(0, 2, 1))
        v = self.tokenConv2_v(v).transpose(1, 2)
        return q,k,v
  
class SAttention(nn.Module):
    def __init__(self, d_model, num_heads=4,  attn_drop=0., proj_drop=0.):
        super(SAttention, self).__init__()
        assert d_model % num_heads == 0, 'd_model should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5
        self.qkvgen = qkvEmbedd(d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self,x):
        B, N, C = x.shape 
        q,k,v = self.qkvgen(x)     
        q = q.reshape(B, N,  self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        k = k.reshape(B, N,  self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3) #
        v = v.reshape(B, N,  self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale 

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 

        x = self.proj(x.permute(0, 2, 1))
        x = x.transpose(2, 1)
        x = self.proj_drop(x)
        return x

   
class TAttention(nn.Module):
    def __init__(self, d_model,  attn_drop=0., proj_drop=0.):
        super(TAttention, self).__init__()

        self.scale = d_model ** -0.5
        self.qkvgen = qkvEmbedd(d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                   kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self,x):
        B, N, C = x.shape 
        q,k,v = self.qkvgen(x)  

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 

        x = self.proj(x.permute(0, 2, 1))
        x = x.transpose(2, 1)
        x = self.proj_drop(x)
        return x


        


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):

        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings




class STSABlock(nn.Module):
    
    def __init__(self, d_model, num_heads = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        self.attnSSA = TAttention(d_model)
        self.attnTSA = SAttention(d_model, num_heads=4)

    def forward(self, x, c):
        x = self.norm1(x+c)
        attSSA = self.attnSSA(x)
        attTSA = self.attnTSA(x)
        x = x + attSSA + attTSA
        return x


class DiffT(nn.Module):

    def __init__(
        self,
        input_size=750,
        in_channels=22,
        hidden_size=1024,
        depth=3,
        num_heads=8,
        class_dropout_prob=0.0,
        num_classes=2,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads

        self.x_embedder = TokenEmbedding(input_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pos_embedfunc = PositionalEmbedding(hidden_size, max_len=5000)

        self.blocks = nn.ModuleList([
            STSABlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.TokendeEmbedding = TokendeEmbedding(hidden_size,input_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)



    def forward(self, x, t, y):

        x = self.x_embedder(x) 
        pos = self.pos_embedfunc(x)
        x = pos + x  
        t = self.t_embedder(t)                  
        y = self.y_embedder(y, self.training)    
        c = t + y                                
        for block in self.blocks:
            x = block(x, c)                     
        x = self.TokendeEmbedding(x)                  
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)




from torch.autograd import Variable


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]



