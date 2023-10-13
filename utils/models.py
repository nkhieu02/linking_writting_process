from torch import nn
from torch.nn.utils import rnn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## RNN 

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim,
                          batch_first = True)

    def forward(self, x, seq_lengths):
        packed_x = rnn.pack_padded_sequence(x, lengths=seq_lengths,
                                            batch_first=True,
                                            enforce_sorted=False)
        output, hidden = self.gru(packed_x)
        return output, hidden

class Grader(nn.Module):
    def __init__(self, hidden_dim,linear_hidden_dim,
                 class_size_dict,embedding_size,
                  c_feat, d_feat):
        super().__init__()
        self.c_feat= c_feat
        self.d_feat = d_feat
        self.embedding_size = embedding_size
        self.class_size_dict = class_size_dict
        self.input_dim = len(c_feat) + len(d_feat) * embedding_size
        self.encoder = Encoder(self.input_dim, hidden_dim)
        self.embedding = self._init_embedding(class_size_dict,
                                              embedding_size)
        self.linear1 = nn.Linear(hidden_dim, linear_hidden_dim)
        self.linear2 = nn.Linear(linear_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def _init_embedding(self, class_size_dict, embedding_size):
        return {f: nn.Embedding(s, embedding_size).to(device) for f,s in \
            class_size_dict.items()}
    def forward(self, c_datas, d_datas, seq_lengths):
        '''
        v: (B, L, D)
        '''
        d_datas_embeded = []
        for i,f in enumerate(self.d_feat):
            d_datas_embeded.append(self.embedding[f](d_datas[:, :, i]))
        inputs = torch.cat([c_datas] + d_datas_embeded, dim= -1)
        _, hidden = self.encoder(inputs, seq_lengths)
        output = self.sigmoid(self.linear2(self.relu(self.linear1(hidden.squeeze(0)))))
        return output * 6
        # return output


## Transformer