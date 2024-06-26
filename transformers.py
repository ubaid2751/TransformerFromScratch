import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformers():
    def word_embedding(self):
        pass
    
    def positional_embedding(self, sequence_length, d_model):
        pos_encoding = torch.zeros(sequence_length, d_model)
        pos = torch.arange(0, sequence_length, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        
        return pos_encoding
        
    def scalar_dot_product(self, query, keys, values):
        d_k = query.size(-1)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(d_k)
        softmax = F.softmax(scores, dim=-1)
        attention_scores = torch.matmul(softmax, values)
        
        return attention_scores