
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Dipole(nn.Module):
    def __init__(self, config):
        super(Dipole, self).__init__()
        params = config['hyper_params']
        
        self.input_dim = params['n_feat']
        self.day_dim = params['day_dim']
        self.output_dim = params['n_class']
        self.rnn_hiddendim = params['rnn_hidden']
        self.keep_prob = params['keep_prob']
        # self.opt = opt
        # self.L2 = L2

        self.day_embedding = nn.Linear(self.input_dim, self.day_dim)
        self.dropout = nn.Dropout(self.keep_prob)
        self.attn = nn.Linear(self.rnn_hiddendim * 2, 1)
        self.gru = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.gru_reverse = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.attn_out = nn.Linear(self.rnn_hiddendim * 4, self.day_dim)
        self.out = nn.Linear(self.day_dim, self.output_dim)

    def attentionStep(self, h_0):
        day_emb = self.day_emb 
        rnn_h = self.gru(day_emb, h_0)[0]
        day_emb_reverse = self.day_emb.flip(dims=[0])  
        rnn_h_reverse = self.gru_reverse(day_emb_reverse, h_0)[0]

        rnn_h = torch.cat((rnn_h, rnn_h_reverse), 2)    # shape=(seq_len, batch_size, 2*hidden_size)

        Alpha = self.attn(rnn_h)  
        Alpha = torch.squeeze(Alpha, dim=2)     # shape=(seq_len, batch_size)
        Alpha = torch.transpose(F.softmax(torch.transpose(Alpha, 0, 1)), 0, 1) 

        attn_applied = Alpha.unsqueeze(2) * rnn_h  
        c_t = torch.mean(attn_applied, 0)  
        h_t = torch.cat((c_t, rnn_h[-1]), dim=1)    # shape=(batch_size, 4*hidden_size)

        h_t_out = self.attn_out(h_t)    # shape=(batch_size, day_dim)
        return h_t_out

    def forward(self, x):
        # x = torch.tensor(x)
        # embedding
        batch_size = x.shape[1]
        h_0 = self.initHidden(batch_size)
        self.day_emb = self.day_embedding(x)    # shape=(seq_len, batch_size, day_dim)

        # LSTM
        if self.keep_prob < 1.0:
            self.day_emb = self.dropout(self.day_emb)

        h_t_out = self.attentionStep(h_0) 

        y_hat = self.out(h_t_out)   
        y_hat = F.softmax(y_hat, dim=1)    # shape=(batch_size, out_dim)

        return y_hat

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn_hiddendim, device=torch.device('cuda:0'))

    def padMatrixWithoutTime(self, seqs):
        lengths = np.array([len(seq) for seq in seqs]).astype("int32")
        n_samples = len(seqs)
        maxlen = np.max(lengths)

        x = np.zeros([maxlen, n_samples, self.input_dim]).astype(np.float32)
        for idx, seq in enumerate(seqs):
            for xvec, subseq in zip(x[:, idx, :], seq):
                for code in subseq:
                    xvec[code] = 1.
        return x, lengths

    
