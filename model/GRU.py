
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class GRU(nn.Module):
    def __init__(self, config, bidirectional=True):
        super(GRU, self).__init__()
        params = config['hyper_params']
        self.out_dim = params["n_class"]
        if params["n_class"] == 2:
            self.out_dim = 1        
        self.n_direct = int(bidirectional)+1
        self.input_dim = params['n_feat']
        self.day_dim = params['day_dim']
        self.rnn_hiddendim = params['rnn_hidden']
        self.n_layer = params['n_layer']
        # self.opt = opt
        # self.L2 = L2

        self.day_embedding = nn.Linear(self.input_dim, self.day_dim)
        self.gru = nn.GRU(self.day_dim, self.rnn_hiddendim, num_layers = self.n_layer , bidirectional=bidirectional)
        self.fc = nn.Linear(self.rnn_hiddendim*self.n_direct, self.out_dim)
        if self.out_dim == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        day_emb = self.day_embedding(x)    # shape=(seq_len, batch_size, day_dim)
        out, _ = self.gru(day_emb)
        out = self.fc(out[-1])   
        out = self.activation(out)    # shape=(batch_size, out_dim)

        return out

    def padMatrixWithoutTime(self, seqs):
        lengths = np.array([len(seq) for seq in seqs]).astype("int32")
        n_samples = len(seqs)
        maxlen = np.max(lengths)

        x = np.zeros([maxlen, n_samples, self.input_dim]).astype(np.float32)
        for idx, seq in enumerate(seqs):
            for xvec, subseq in zip(x[:, idx, :], seq):
                xvec[subseq] = 1.
        return x, lengths

    
