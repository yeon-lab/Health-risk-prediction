import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import warnings
import random
from model.utils import PositionalEncoding, clones, FinalAttentionQKV, MultiHeadedAttention, SublayerConnection, PositionwiseFeedForward, SingleAttention

warnings.filterwarnings('ignore')

n_gpu = torch.cuda.device_count()
device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')

class AdaDiag(nn.Module):
    def __init__(self, config, criterion):
        super(AdaDiag, self).__init__()
        
        params = config['hyper_params']
        self.out_dim = params["n_class"]
        if params["n_class"] == 2:
            self.out_dim = 1        
        self.input_dim = params['n_feat']
        self.day_dim = params['day_dim']
        self.hidden_dim = params['rnn_hidden']
        self.criterion = criterion
        
        if self.out_dim == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
            
        n_layer = 2
        for nhead in range(1, 10):
            try:
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=nhead, batch_first=True, dim_feedforward=self.day_dim)
                break
            except:
                continue
            
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.out_dim),
            self.activation
            )

        self.domain = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim), nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=1),
            nn.Sigmoid()
            )
        self.criterion_domain = nn.BCELoss()

    def forward(self, x): 
        x = x.permute(1, 0, 2)
        features = self.transformer_encoder(x)
        features = features[:,-1]

        y_pred = self.classifier(features)
        d_pred = self.domain(features)
        
        return y_pred.squeeze(1), d_pred.squeeze(1)
        
    def predict(self, x, y, d, x_weights=None):
        y_pred, d_pred = self.forward(x)
        y_loss = self.criterion(y_pred, y).mean()
        d_loss = self.criterion_domain(d_pred, d.float())
        loss = y_loss + d_loss
        return y_pred, loss
        
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

################################################################################################################
########  GRU  #################################################################################################
################################################################################################################
class GRU(nn.Module):
    def __init__(self, config, criterion, bidirectional=True):
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
        self.criterion = criterion

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
        out = self.fc(out[-1])   # shape=(batch_size, out_dim)
        out = self.activation(out)    # shape=(batch_size, out_dim)
        return out
        
    def predict(self, x, y, x_weights=None):
        pred = self.forward(x)
        pred = pred.squeeze(1)
        if x_weights is None:
            loss = self.criterion(pred, y).mean()
        else:
            losses = self.criterion(pred, y).view(1, -1)
            x_weights = x_weights.view(-1, 1)
            weighted_loss = losses.mm(x_weights)
            loss = weighted_loss.view(1)
        return pred, loss

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                for layer_p in m._all_weights:
                    for p in layer_p:
                        if 'weight' in p:
                            torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                        if 'bias' in p:
                            torch.nn.init.constant_(m.__getattr__(p), 0.0)   


################################################################################################################
########  LSTM  ################################################################################################
################################################################################################################
class LSTM(nn.Module):
    def __init__(self, config, criterion, bidirectional=True):
        super(LSTM, self).__init__()
        params = config['hyper_params']
        self.out_dim = params["n_class"]
        if params["n_class"] == 2:
            self.out_dim = 1        
        self.n_direct = int(bidirectional)+1
        self.input_dim = params['n_feat']
        self.day_dim = params['day_dim']
        self.rnn_hiddendim = params['rnn_hidden']
        self.n_layer = params['n_layer']
        self.criterion = criterion

        self.day_embedding = nn.Linear(self.input_dim, self.day_dim)
        self.lstm = nn.LSTM(self.day_dim, self.rnn_hiddendim, num_layers = self.n_layer , bidirectional=bidirectional)
        self.fc = nn.Linear(self.rnn_hiddendim*self.n_direct, self.out_dim)
        if self.out_dim == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        day_emb = self.day_embedding(x)    # shape=(seq_len, batch_size, day_dim)
        out, (h_n, c_n) = self.lstm(day_emb)
        out = self.fc(out[-1])   # shape=(batch_size, out_dim)
        out = self.activation(out)    # shape=(batch_size, out_dim)
        return out
        
    def predict(self, x, y, x_weights=None):
        pred = self.forward(x)
        pred = pred.squeeze(1)
        
        if x_weights is None:
            loss = self.criterion(pred, y).mean()
        else:
            losses = self.criterion(pred, y).view(1, -1)
            x_weights = x_weights.view(-1, 1)
            weighted_loss = losses.mm(x_weights)
            loss = weighted_loss.view(1)
                
        return pred, loss

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                for layer_p in m._all_weights:
                    for p in layer_p:
                        if 'weight' in p:
                            torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                        if 'bias' in p:
                            torch.nn.init.constant_(m.__getattr__(p), 0.0)           

################################################################################################################
#########  Dipole  #############################################################################################
################################################################################################################
class Dipole(nn.Module):
    def __init__(self, config, criterion):
        super(Dipole, self).__init__()
        params = config['hyper_params']

        self.criterion = criterion
        self.input_dim = params['n_feat']
        self.day_dim = params['day_dim']
        self.rnn_hiddendim = params['rnn_hidden']
        self.keep_prob = params['keep_prob']
        
        if params["n_class"] == 2:
            self.out_dim = 1
        else:
            self.out_dim = params["n_class"]
        if self.out_dim  == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
            
        self.day_embedding = nn.Linear(self.input_dim, self.day_dim)
        self.dropout = nn.Dropout(1-self.keep_prob)
        self.attn = nn.Linear(self.rnn_hiddendim * 2, 1)
        self.gru = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.gru_reverse = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.attn_out = nn.Linear(self.rnn_hiddendim * 4, self.day_dim)
        self.classifier = nn.Sequential( 
                    nn.Linear(self.day_dim, self.out_dim),
                    self.activation
                    )



    def attentionStep(self, h_0):
        day_emb = self.day_emb 
        rnn_h = self.gru(day_emb, h_0)[0]
        day_emb_reverse = self.day_emb.flip(dims=[0]) 
        rnn_h_reverse = self.gru_reverse(day_emb_reverse, h_0)[0]

        rnn_h = torch.cat((rnn_h, rnn_h_reverse), 2)    # hape=(seq_len, batch_size, 2*hidden_size)

        Alpha = self.attn(rnn_h)    # shape=(seq_len, batch_size, 1)
        Alpha = torch.squeeze(Alpha, dim=2)     # shape=(seq_len, batch_size)
        Alpha = torch.transpose(F.softmax(torch.transpose(Alpha, 0, 1)), 0, 1) 

        attn_applied = Alpha.unsqueeze(2) * rnn_h   
        c_t = torch.mean(attn_applied, 0)  
        h_t = torch.cat((c_t, rnn_h[-1]), dim=1)   

        h_t_out = self.attn_out(h_t)    
        return h_t_out

    def forward(self, x):
        # x = torch.tensor(x)
        # embedding
        batch_size = x.shape[1]
        h_0 = self.initHidden(batch_size, x.device)
        self.day_emb = self.day_embedding(x)    # shape=(seq_len, batch_size, day_dim)
        # LSTM
        if self.keep_prob < 1.0:
            self.day_emb = self.dropout(self.day_emb)
        h_t_out = self.attentionStep(h_0) # shape=(batch_size, day_dim)
        return h_t_out

    def predict(self, x, y, x_weights=None):
        feature = self.forward(x)
        pred = self.classifier(feature)  # shape=(batch_size, out_dim)
        pred = pred.squeeze(1)
        if x_weights is None:
            loss = self.criterion(pred, y).mean()
        else:
            losses = self.criterion(pred, y).view(1, -1)
            x_weights = x_weights.view(-1, 1)
            weighted_loss = losses.mm(x_weights)
            loss = weighted_loss.view(1)
        return pred, loss
        
    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.rnn_hiddendim, device=device)
        
    def weights_init(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                    for layer_p in m._all_weights:
                        for p in layer_p:
                            if 'weight' in p:
                                torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                            if 'bias' in p:
                                torch.nn.init.constant_(m.__getattr__(p), 0.0)       
                                
################################################################################################################
#########  DG  #############################################################################################
################################################################################################################
class DG(nn.Module): # Dipole model based
    def __init__(self, config, criterion):
        super(DG, self).__init__()
        params = config['hyper_params']

        self.criterion = criterion
        self.input_dim = params['n_feat']
        self.day_dim = params['day_dim']
        self.rnn_hiddendim = params['rnn_hidden']
        self.keep_prob = params['keep_prob']
        
        if params["n_class"] == 2:
            self.out_dim = 1
        else:
            self.out_dim = params["n_class"]
        if self.out_dim  == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
            
        self.day_embedding = nn.Linear(self.input_dim, self.day_dim)
        self.dropout = nn.Dropout(1-self.keep_prob)
        self.attn = nn.Linear(self.rnn_hiddendim * 2, 1)
        self.gru = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.gru_reverse = nn.GRU(self.day_dim, self.rnn_hiddendim)
        self.attn_out = nn.Linear(self.rnn_hiddendim * 4, self.day_dim)
        self.classifier = nn.Sequential( 
                    nn.Linear(self.day_dim, self.out_dim),
                    self.activation
                    )
        self.discriminater = nn.Sequential(
                  nn.Linear(self.day_dim, params["n_domains"]),
                  nn.Softmax(dim=1)
                )
        self.criterion_domain = nn.CrossEntropyLoss()



    def attentionStep(self, h_0):
        day_emb = self.day_emb 
        rnn_h = self.gru(day_emb, h_0)[0]
        day_emb_reverse = self.day_emb.flip(dims=[0]) 
        rnn_h_reverse = self.gru_reverse(day_emb_reverse, h_0)[0]

        rnn_h = torch.cat((rnn_h, rnn_h_reverse), 2)    # hape=(seq_len, batch_size, 2*hidden_size)

        Alpha = self.attn(rnn_h)    # shape=(seq_len, batch_size, 1)
        Alpha = torch.squeeze(Alpha, dim=2)     # shape=(seq_len, batch_size)
        Alpha = torch.transpose(F.softmax(torch.transpose(Alpha, 0, 1)), 0, 1) 

        attn_applied = Alpha.unsqueeze(2) * rnn_h   
        c_t = torch.mean(attn_applied, 0)  
        h_t = torch.cat((c_t, rnn_h[-1]), dim=1)   

        h_t_out = self.attn_out(h_t)    
        return h_t_out

    def forward(self, x):
        # x = torch.tensor(x)
        # embedding
        batch_size = x.shape[1]
        h_0 = self.initHidden(batch_size, x.device)
        self.day_emb = self.day_embedding(x)    # shape=(seq_len, batch_size, day_dim)
        # LSTM
        if self.keep_prob < 1.0:
            self.day_emb = self.dropout(self.day_emb)
        h_t_out = self.attentionStep(h_0) # shape=(batch_size, day_dim)
        return h_t_out
        
    def predict(self, x, y, d):
        feature = self.forward(x)
        
        y_pred = self.classifier(feature)
        y_pred = y_pred.squeeze(1)
        y_loss = self.criterion(y_pred, y).mean()

        d_pred = self.discriminater(feature)
        d_loss = self.criterion_domain(d_pred.squeeze(1), d)

        loss = y_loss + d_loss
        return y_pred, loss
        
    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.rnn_hiddendim, device=device)
        
    def weights_init(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                    for layer_p in m._all_weights:
                        for p in layer_p:
                            if 'weight' in p:
                                torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                            if 'bias' in p:
                                torch.nn.init.constant_(m.__getattr__(p), 0.0)       
                                                                
        
################################################################################################################
########  Retain  ##############################################################################################
################################################################################################################
class Retain(nn.Module):
    def __init__(self, config, criterion):
        super(Retain, self).__init__()
        """
        num_embeddings(int): size of the dictionary of embeddings
        embedding_dim(int) the size of each embedding vector
        """
        params = config['hyper_params']

        self.n_feat = params["n_feat"]
        self.criterion = criterion
        embedding_dim =  params['rnn_hidden']
        self.var_rnn_hidden_size = params['rnn_hidden']
        var_rnn_output_size = params['rnn_hidden']
        self.visit_rnn_hidden_size = params['rnn_hidden']
        visit_rnn_output_size = params['rnn_hidden']
        var_attn_output_size = params['rnn_hidden']
        visit_attn_output_size = 1
        embedding_output_size = params['rnn_hidden']
        

        if params["n_class"] == 2:
            self.out_dim = 1
        else:
            self.out_dim = params["n_class"]
            
        self.emb_layer = nn.Linear(in_features=params["n_feat"], out_features=embedding_dim)
        self.dropout = nn.Dropout(params["dropout_p"])
        self.variable_level_rnn = nn.GRU(self.var_rnn_hidden_size, var_rnn_output_size)
        self.visit_level_rnn = nn.GRU(self.visit_rnn_hidden_size, visit_rnn_output_size)
        self.variable_level_attention = nn.Linear(var_rnn_output_size, var_attn_output_size)
        self.visit_level_attention = nn.Linear(visit_rnn_output_size, visit_attn_output_size)
        self.output_dropout = nn.Dropout(params["output_dropout_p"])
        self.output_layer = nn.Linear(embedding_output_size, self.out_dim)


        self.n_samples = config["data_loader"]["args"]["batch_size"]
        self.reverse_rnn_feeding = True
        
        if self.out_dim  == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.shape[1]
        var_rnn_hidden, visit_rnn_hidden = self.init_hidden(batch_size, input.device)
        v = self.emb_layer(input)
        v = self.dropout(v)


        if self.reverse_rnn_feeding:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(torch.flip(v, [0]), visit_rnn_hidden)
            alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        else:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(v, visit_rnn_hidden)
            alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = F.softmax(alpha, dim=0)

        if self.reverse_rnn_feeding:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(torch.flip(v, [0]), var_rnn_hidden)
            beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        else:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(v, var_rnn_hidden)
            beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = torch.tanh(beta)


        attn_w = visit_attn_w * var_attn_w

        c = torch.sum(attn_w * v, dim=0)
        c = self.output_dropout(c)
        output = self.output_layer(c)
        output = self.activation(output)

        return output
        
    def predict(self, x, y, x_weights=None):
        pred = self.forward(x)
        pred = pred.squeeze(1)
        
        if x_weights is None:
            loss = self.criterion(pred, y).mean()
        else:
            losses = self.criterion(pred, y).view(1, -1)
            x_weights = x_weights.view(-1, 1)
            weighted_loss = losses.mm(x_weights)
            loss = weighted_loss.view(1)
                
        return pred, loss

    def init_hidden(self, current_batch_size, device):
        return torch.zeros(current_batch_size, self.var_rnn_hidden_size).unsqueeze(0).to(device), torch.zeros(current_batch_size, self.visit_rnn_hidden_size).unsqueeze(0).to(device)

    def weights_init(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                    for layer_p in m._all_weights:
                        for p in layer_p:
                            if 'weight' in p:
                                torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                            if 'bias' in p:
                                torch.nn.init.constant_(m.__getattr__(p), 0.0)         

################################################################################################################
########  Stagenet  #############################################################################################
################################################################################################################
class Stagenet(nn.Module):
    def __init__(self, config, criterion, hidden_dim=384, conv_size=10, levels=3, dropout=0.3, dropconnect = 0.3, dropres=0.3):
        super(Stagenet, self).__init__()

        self.criterion = criterion
        assert hidden_dim % levels == 0
        
        params = config['hyper_params']
        input_dim = params['n_feat']
        self.input_dim = input_dim
        
        
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.levels = levels

        self.hidden_dim = hidden_dim
        conv_dim = hidden_dim
        self.conv_size = conv_size
        self.chunk_size = hidden_dim // levels

        if params["n_class"] == 2:
            self.out_dim = 1
        else:
            self.out_dim = params["n_class"]
        
        self.kernel = nn.Linear(input_dim+1, hidden_dim*4+levels*2)
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(hidden_dim+1, hidden_dim*4+levels*2)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)
        
        self.nn_scale = nn.Linear(hidden_dim, hidden_dim // 6)
        self.nn_rescale = nn.Linear(hidden_dim // 6, hidden_dim)
        self.nn_conv = nn.Conv1d(hidden_dim, conv_dim, conv_size, 1)
        self.nn_output = nn.Linear(conv_dim, self.out_dim)
        
        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)
    
    def cumax(self, x, mode='l2r'):
        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x
    
    def step(self, inputs, c_last, h_last):
        x_in = inputs
        delta_t = np.array([1.0])
        interval = torch.ones((x_in.size(0),1),dtype=torch.float32).to(device)
        x_out1 = self.kernel(torch.cat((x_in,interval),dim=-1))
        x_out2 = self.recurrent_kernel(torch.cat((h_last,interval),dim=-1))
        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, :self.levels], 'l2r')
        f_master_gate = f_master_gate.unsqueeze(2)
        i_master_gate = self.cumax(x_out[:, self.levels:self.levels*2], 'r2l')
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels*2:]
        x_out = x_out.reshape(-1, self.levels*4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, :self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels:self.levels*2])
        o_gate = torch.sigmoid(x_out[:, self.levels*2:self.levels*3])
        c_in = torch.tanh(x_out[:, self.levels*3:])
        c_last = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + (f_master_gate - overlap) * c_last + (i_master_gate - overlap) * c_in
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

        
    def forward(self, input):
        input = input.permute(1, 0, 2)
        batch_size, time_step, feature_dim = input.size()
        c_out = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_out = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        #s*B*H
        tmp_h = torch.zeros_like(h_out, dtype=torch.float32).view(-1).repeat(self.conv_size).view(self.conv_size, batch_size, self.hidden_dim).to(device)
        tmp_dis = torch.zeros((self.conv_size, batch_size)).to(device)
        h = []
        origin_h = []
        distance = []
        for t in range(time_step):
            out, c_out, h_out = self.step(input[:, t, :], c_out, h_out)
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim:self.hidden_dim+self.levels], -1)
            cur_distance_in = torch.mean(out[..., self.hidden_dim+self.levels:], -1)
            origin_h.append(out[..., :self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:], out[..., :self.hidden_dim].unsqueeze(0)), 0)
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)
            distance.append(cur_distance)
            
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)
            
            local_theme = torch.mean(local_h, dim=-1)
            local_theme = self.nn_scale(local_theme)
            local_theme = torch.relu(local_theme)
            local_theme = self.nn_rescale(local_theme)
            local_theme = torch.sigmoid(local_theme)
            
            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h.append(local_h)  

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        if self.dropres > 0.0:
            origin_h = self.nn_dropres(origin_h)
        rnn_outputs = rnn_outputs + origin_h
        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)
        output = self.nn_output(rnn_outputs)
        output = output.contiguous().view(batch_size, time_step, self.out_dim)
        output = output[:,-1,:]
        output = torch.sigmoid(output)
        
        return output.view(batch_size, -1)
        
    def predict(self, x, y, x_weights=None):
        pred = self.forward(x)
        pred = pred.squeeze(1)
        
        if x_weights is None:
            loss = self.criterion(pred, y).mean()
        else:
            losses = self.criterion(pred, y).view(1, -1)
            x_weights = x_weights.view(-1, 1)
            weighted_loss = losses.mm(x_weights)
            loss = weighted_loss.view(1)
                
        return pred, loss
        
    def weights_init(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                    for layer_p in m._all_weights:
                        for p in layer_p:
                            if 'weight' in p:
                                torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                            if 'bias' in p:
                                torch.nn.init.constant_(m.__getattr__(p), 0.0)                 
        
################################################################################################################
########  Concare  ##############################################################################################
################################################################################################################
class Concare(nn.Module):
    def __init__(self, config, criterion, keep_prob=0.5):
        super(Concare, self).__init__()
        
        self.criterion = criterion
        params = config['hyper_params']
        self.input_dim = params['n_feat']
        # hyperparameters
        
        hidden_dim = 64
        self.hidden_dim = hidden_dim  # d_model
        self.d_model = hidden_dim
        self.MHD_num_head = 4
        self.d_ff = 256
        self.keep_prob = keep_prob
        
        if params["n_class"] == 2:
            self.out_dim = 1
        else:
            self.out_dim = params["n_class"]

        # layers
        self.PositionalEncoding = PositionalEncoding(self.d_model, dropout = 0, max_len = 400)

        self.GRUs = clones(nn.GRU(1, self.hidden_dim, batch_first = True), self.input_dim)
        self.LastStepAttentions = clones(SingleAttention(self.hidden_dim, 8, attention_type='new', demographic_dim=12, time_aware=True, use_demographic=False),self.input_dim)
        
        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, attention_type='mul',dropout = 1 - self.keep_prob)

        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.d_model,dropout = 1 - self.keep_prob)
        self.SublayerConnection = SublayerConnection(self.d_model, dropout = 1 - self.keep_prob)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=0.1)


        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.out_dim)

        self.dropout = nn.Dropout(p = 1 - self.keep_prob)
        if self.out_dim  == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
        self.relu=nn.ReLU()

        
    def forward(self, input):
        input = input.permute(1, 0, 2) # input shape [batch_size, timestep, feature_dim]
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert(feature_dim == self.input_dim)# input Tensor : 256 * 48 * 76
        assert(self.d_model % self.MHD_num_head == 0)

        # forward
        GRU_embeded_input = self.GRUs[0](input[:,:,0].unsqueeze(-1), Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device))[0] # b t h
        Attention_embeded_input = self.LastStepAttentions[0](GRU_embeded_input)[0].unsqueeze(1)# b 1 h
        for i in range(feature_dim-1):
            embeded_input = self.GRUs[i+1](input[:,:,i+1].unsqueeze(-1), Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device))[0] # b 1 h
            embeded_input = self.LastStepAttentions[i+1](embeded_input)[0].unsqueeze(1)# b 1 h
            Attention_embeded_input = torch.cat((Attention_embeded_input, embeded_input), 1)# b i h


        posi_input = self.dropout(Attention_embeded_input) # batch_size * d_input+1 * hidden_dim

        contexts = self.SublayerConnection(posi_input, lambda x: self.MultiHeadedAttention(posi_input, posi_input, posi_input, None))# # batch_size * d_input * hidden_dim
        contexts = contexts[0]

        contexts = self.SublayerConnection(contexts, lambda x: self.PositionwiseFeedForward(contexts))[0]# # batch_size * d_input * hidden_dim

        weighted_contexts = self.FinalAttentionQKV(contexts)[0]
        output = self.output1(self.relu(self.output0(weighted_contexts)))# b 1
        output = self.activation(output)
          
        return output
        
    def predict(self, x, y, x_weights=None):
        pred = self.forward(x)
        pred = pred.squeeze(1)
        if x_weights is None:
            loss = self.criterion(pred, y).mean()
        else:
            losses = self.criterion(pred, y).view(1, -1)
            x_weights = x_weights.view(-1, 1)
            weighted_loss = losses.mm(x_weights)
            loss = weighted_loss.view(1)
        return pred, loss

    def weights_init(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                    for layer_p in m._all_weights:
                        for p in layer_p:
                            if 'weight' in p:
                                torch.nn.init.uniform_(m.__getattr__(p), -0.1, 0.1)
                            if 'bias' in p:
                                torch.nn.init.constant_(m.__getattr__(p), 0.0)         
