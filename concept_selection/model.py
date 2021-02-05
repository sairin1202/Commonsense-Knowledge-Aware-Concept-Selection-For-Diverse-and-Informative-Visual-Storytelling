import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
import pickle
import numpy as np


use_glove = False


import torch
from torch import nn
from torch.nn import functional as F
import random


class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        # embedding (input) layer options
        self.feat_size = 2048
        self.embed_dim = 300
        self.story_size = 5
        # rnn layer options
        self.visual_emb = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                        nn.BatchNorm1d(self.embed_dim),
                                        nn.ReLU())
        self.position_embed = nn.Embedding(self.story_size, self.embed_dim)
    def forward(self, input, hidden=None):
        """
        inputs:
        - input  	(batch_size, 5, feat_size)
        - hidden 	(num_layers * num_dirs, batch_size, hidden_dim // 2)
        return:
        - out 		(batch_size, 5, rnn_size), serve as context
        """
        batch_size, seq_length = input.size(0), input.size(1)

        # visual embeded
        emb = self.visual_emb(input.view(-1, self.feat_size))
        emb = emb.view(batch_size, seq_length, -1)  # (Na, album_size, embedding_size)

        for i in range(emb.size(-2)):
            position = input.data.new(batch_size).long().fill_(i)
            emb[:, i, :] = emb[:, i, :] + self.position_embed(position)
        return emb




# Embedder
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model, weights_matrix):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        if weights_matrix is not None:
            print("Initialize embedding layer with Glove ", weights_matrix.shape)
            self.embed.load_state_dict({'weight': torch.Tensor(weights_matrix)})
            self.embed.weight.requires_grad = False
    def forward(self, x):
        # print(self.embed.weight.grad)
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 600):
        super().__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len]
        pe.require_grad = False
        pe = pe.cuda()
        x = x + pe
        # print(pe.size())
        return x



# SubLayers
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    # if mask is not None:
    #     scores = scores.masked_fill(mask == 0, 1e-18)
    attn = scores
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)


        # calculate attention using function we will define next
        scores, attn = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        # B * H * S * dim/H
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        # B * S * dim
        output = self.out(concat)
        # B * S * dim
        return output, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask)[0])
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x




# Transformer
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, weights_matrix):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, im_feat, mask):
        # print(x.size(), im_feat.size(), mask.size())
        x = im_feat
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        # print("encoder2 size", x.size())
        return self.norm(x)



class Transformer(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout, weights_matrix=None):
        super().__init__()
        self.embed = Embedder(vocab, d_model, weights_matrix)
        self.visual_encoder = VisualEncoder()
        self.encoder = Encoder(vocab, d_model, N, heads, dropout, weights_matrix)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

    def forward(self, im_feat, word_idx, adjacent_matrix, src_mask):
        word_feat = self.embed(word_idx)
        im_feat = self.visual_encoder(im_feat)
        feat = torch.cat([word_feat, im_feat], dim=-2)
        e_outputs = self.encoder(feat, adjacent_matrix)
        e_outputs = e_outputs[:,:501,:]
        # e_outputs = torch.relu(e_outputs)
        # print(e_outputs.size())
        k_outputs = self.linear_k(e_outputs)
        v_outputs = self.linear_v(e_outputs)
        scores = torch.matmul(k_outputs, v_outputs.transpose(-2, -1))/300
        # scores = self.linear(scores)
        # print(scores)
        scores = torch.sigmoid(scores)
        return scores, None



def get_model(vocab, load_epoch, d_model, n_layers, heads, dropout):

    assert d_model % heads == 0
    assert dropout < 1
    # init glove vector

    #  init model
    transformer = Transformer(vocab, d_model, n_layers, heads, dropout)
    if load_epoch is not None:
        print("loading pretrained weights...")
        transformer.load_state_dict(torch.load(f'./models/transformer{load_epoch}.pth'))
    else:
        total_cnt = 0
        cnt = 0
        for p in transformer.parameters():
            total_cnt += p.view(-1).size(0)
            if p.dim() > 1 and p.requires_grad == True:
                cnt += p.view(-1).size(0)
                nn.init.xavier_uniform_(p)
        print("Total parameters {} , initilized {} parameters".format(total_cnt, cnt))

    transformer = transformer.cuda()
    return transformer