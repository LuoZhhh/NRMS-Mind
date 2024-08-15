import torch.nn as nn
import torch

from transformers import BertModel



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in=256, num_heads=16):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_in % num_heads == 0, "dim must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_in
        self.dim_v = dim_in
        self.num_heads = num_heads
        self.linear_q = nn.Linear(self.dim_in, self.dim_k, bias=False)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v, bias=False)
        self.multiheadattention=torch.nn.MultiheadAttention(embed_dim=self.dim_in, num_heads=self.num_heads, kdim=self.dim_k, vdim=self.dim_v, batch_first=True)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # (batch, nh, n, dk)
        k = self.linear_k(x)  # (batch, nh, n, dk)
        v = self.linear_v(x)  # (batch, nh, n, dv)

        att, _ = self.multiheadattention(q, k, v)

        return att


class News_Encoder(nn.Module):
    def __init__(self, model_path):
        super(News_Encoder, self).__init__()
        self.embedding  = BertModel.from_pretrained(model_path).get_input_embeddings()
        self.h_dim = 256 if 'bert-mini' in model_path else 512
        self.multi_head = MultiHeadSelfAttention(self.h_dim)
        self.news_layer = nn.Sequential(nn.Linear(self.h_dim, 200),
                                        nn.Tanh(),  
                                        nn.Linear(200, 1),
                                        nn.Flatten(), nn.Softmax(dim=0)) 
    def forward(self, x):
        # import pdb; pdb.set_trace()
        outputs = self.embedding(x)  # [len_history, max_news_length, hid_dim]
        multi_attention=self.multi_head(outputs)  # [len_history, max_news_length, hid_dim]
        attention_weight = self.news_layer(multi_attention).unsqueeze(2)  # [len_history, max_news_length, 1]
        new_emb = torch.sum(multi_attention * attention_weight, dim=1)  # [len_history, hid_dim]
        return new_emb


class User_Encoder(nn.Module):
    def __init__(self, model_news):
        super(User_Encoder, self).__init__()
        self.news_encoder = model_news
        self.h_dim = self.news_encoder.h_dim
        self.multi_head = MultiHeadSelfAttention(dim_in=self.h_dim)
        self.news_layer = nn.Sequential(nn.Linear(self.h_dim, 200),
                                        nn.Tanh(),  
                                        nn.Linear(200, 1),
                                        nn.Flatten(), nn.Softmax(dim=0)) 
    def forward(self, x):  # x -- [len_history, max_news_length]
        # import pdb; pdb.set_trace()
        outputs = self.news_encoder(x).unsqueeze(0)  # [1, len_history, hid_dim]
        multi_attention = self.multi_head(outputs)  # [1, len_history, hid_dim]
        attention_weight = self.news_layer(multi_attention).unsqueeze(2)  # [1, len_history, 1]
        new_emb = torch.sum(multi_attention * attention_weight, dim=1)  # [1, hid_dim]
        return new_emb