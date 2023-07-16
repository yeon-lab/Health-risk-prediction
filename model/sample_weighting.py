import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from model.utils import auto_encoder

SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_gpu = torch.cuda.device_count()
device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')


def padMatrix(seqs, n_feat, maxlen=None):
    lengths = np.array([len(seq) for seq in seqs]).astype('int32')
    n_samples = len(seqs)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen, n_feat))
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[idx, :, :], seq):
            xvec[subseq] = 1.
    return x

class Autoencoder(nn.Module):
    def __init__(self, n_feat, kl_dim, epochs=1000, lr=0.001, batch_size=1024):
        super(Autoencoder, self).__init__()
        self.model = auto_encoder(n_feat, kl_dim).to(device)
        self.model.weights_init()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, input):
        data_loader = DataLoader(input, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.model.train()
        for step in range(1, self.epochs+1):
            for batch_idx, x in enumerate(data_loader):
                x = x.to(device)
                x_hat = self.model(x)
                loss = self.criterion(x_hat, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batch_idx == 0:
                    if step % 100 == 0:
                        print('autoencoder {} step) loss: {}'.format(step, loss.item())) 
                    
    def get_features(self, x):
        n_samples = len(x)
        self.model.eval()
        x = self.model.encoder(x)
        return x.view(n_samples, -1)
        
def get_train_weigts(x, y, steps, step_lr, n_feat, kl_weight, dist_weight, kl_dim):
    x, y = torch.FloatTensor(x), torch.FloatTensor(y)
    x_n_samples = len(x)
    
    y_code = y.sum(1)
    y_dist = (y_code.sum(0)/y_code.sum()).to(device)
    x_code = x.sum(1).to(device)
    
    autoencoder = Autoencoder(n_feat, kl_dim)
    autoencoder.train(torch.cat((x,y),0))

    x_feat = autoencoder.get_features(x.to(device))
    y_feat = autoencoder.get_features(y.to(device))
    q = F.softmax(y_feat.mean(0))

    mse = nn.MSELoss()
    kl_div = nn.KLDivLoss(reduction="batchmean")
    relu = nn.ReLU()
    
    weight = nn.Parameter(torch.ones(size=(x_n_samples, 1), device=device), requires_grad=True)
    optimizer = optim.Adam([weight], lr=step_lr)
    for step in range(1, steps+1):
        x_code_weighted = relu(weight)*x_code
        x_dist = x_code_weighted.sum(0)/x_code_weighted.sum()
        loss_dist = mse(x_dist, y_dist)
        
        x_feat_weighted = relu(weight)*x_feat
        p = F.log_softmax(x_feat_weighted.mean(0))
        loss_KL = kl_div(p, q)

        loss_weight = (relu(weight).sum()-x_n_samples).pow(2)
        
        loss = loss_dist*dist_weight + loss_weight + loss_KL*kl_weight
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
 
        if step % 100 == 0:
            print('{}-th step) loss_dist: {}, loss_weight: {}, loss_KL: {}'.format(
                    step, loss_dist, loss_weight, loss_KL
            ))
        
    return relu(weight).cpu().detach().numpy()
    

def reweight_data(data, maxlen, n_feat, steps, step_lr, kl_weight, dist_weight, kl_dim):
    train = padMatrix(data['train']['DX'], n_feat, maxlen)
    valid = padMatrix(data['valid']['DX'], n_feat, maxlen)
    test = padMatrix(data['reweight']['DX'], n_feat, maxlen)

    weights = get_train_weigts(train, test, steps, step_lr, n_feat, kl_weight, dist_weight, kl_dim)
    data['weights'] = weights
    
    valid_sim_to_train = np.matmul(valid.reshape((len(valid),-1)), train.reshape((len(train),-1)).transpose())
    valid_sim_to_train = np.argmax(valid_sim_to_train, 1)
    data['valid_weights'] = weights[valid_sim_to_train]

    return data
