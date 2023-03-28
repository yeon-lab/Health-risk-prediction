import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

device = torch.device("cuda:0")

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
    def __init__(self, n_feat, kl_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential( 
            nn.Linear(n_feat, kl_dim), 
            nn.ReLU(),
        )
        self.decoder = nn.Sequential( 
            nn.Linear(kl_dim, n_feat), 
            nn.Sigmoid(),
        )
    def get_features(self, x):
        return self.encoder(x)
        
    def forward(self, x):
        output = self.decoder(self.get_features(x)) 
        return output
        
def get_train_weigts(x, y, steps, step_lr, n_feat, kl_weight, dist_weight, kl_dim):
    
    x = torch.FloatTensor(x)#.to(device)
    y = torch.FloatTensor(y)#.to(device)
    x_n_samples, y_n_samples = len(x), len(y)
    
    autoencoder = Autoencoder(n_feat, kl_dim).to(device)
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)
    
    mse = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    relu = nn.ReLU()
    
    #dataloader = DataLoader(np.concatenate((x,y),0), batch_size=batch_size, shuffle=True, drop_last= False)
    dataloader = DataLoader(torch.cat((x,y),0), batch_size=128, shuffle=True, drop_last= False)

    autoencoder.train()
    for step in range(100):
        for batch_idx, samples in enumerate(dataloader):
            inputs, labels = samples.to(device), samples.to(device)
            outputs = autoencoder(inputs)
            loss = mse(outputs, labels)
            autoencoder_optimizer.zero_grad()
            loss.backward()
            autoencoder_optimizer.step()
            if (step%10 == 0)&(batch_idx==0) or (step==steps-1)&(batch_idx==0):
                print('autoencoder {} step) loss: {}'.format(step, loss.item()))                
    
    del dataloader, inputs, labels, samples
    
    weight = nn.Parameter(torch.ones(size=(x_n_samples, 1), device=device), requires_grad=True)
    weight_optimizer = optim.Adam([weight], lr=step_lr)
    
    y_sum = y.sum(1)
    y_dist = y_sum.sum(0)/y_sum.sum()
    y_dist = y_dist.to(device)
    x_dist_input = x.sum(1).to(device)
    
    x = x.to(device)
    y = y.to(device)

    autoencoder.eval()    
    for step in range(steps):
        weight_optimizer.zero_grad()
        weighted_x = relu(weight)*x_dist_input
        x_dist = weighted_x.sum(0)/weighted_x.sum()

        loss_dist = mse(x_dist, y_dist)
        loss_weight = (relu(weight).sum()-x_n_samples).pow(2)
        
        x_features = autoencoder.get_features(x)
        x_features = x_features.view(x_n_samples, -1)
        x_features = weight*x_features
        
        
        #x_features = x_features.mean(0)
        #x_features = x_features/x_features.sum()
        x_features = x_features.sum(0)/x_features.sum()

        y_features = autoencoder.get_features(y)
        y_features = y_features.view(y_n_samples, -1)
        
        #y_features = y_features.mean(0)
        #y_features = y_features/y_features.sum()        
        y_features = y_features.sum(0)/y_features.sum()
        
        if step == 0:
            print('y_dist shape:', y_dist.shape)
            print('x_dist shape:', x_dist.shape)
            print('weight shape:', weight.shape)
            print('x_features shape:', x_features.shape)
            print('y_features shape:', y_features.shape)
        
        loss_KL = kl_loss(x_features.log(), y_features)
        
        loss = loss_dist*dist_weight + loss_weight + loss_KL*kl_weight
            
        loss.backward(retain_graph=True)
        weight_optimizer.step()
            
        if step%100 == 0  or step==steps-1:
            print('{} step) loss_dist: {}, loss_weight: {}, loss_KL: {}, w_sum: {}'.format(
                step, loss_dist, loss_weight, loss_KL, weight.sum().item()))
        
    weight = relu(weight).cpu().detach().numpy()
    
    print('weight.shape:', weight.shape, 'weight.sum:',weight.sum(), 'abs(weight).sum:',np.absolute(weight).sum())
    return weight
    

def reweight_data(data, maxlen, n_feat, steps, step_lr, kl_weight, dist_weight, kl_dim):
    train = data['train']['DX'].copy().tolist()
    valid = data['valid']['DX'].copy().tolist()
    test = data['reweight_test']['DX'].copy().tolist()

    train = padMatrix(train, n_feat, maxlen)
    valid = padMatrix(valid, n_feat, maxlen)
    test = padMatrix(test, n_feat, maxlen)

    weights = get_train_weigts(train, test, steps, step_lr, n_feat, kl_weight, dist_weight, kl_dim)
    data['weights'] = weights
    
    valid_sim_to_train = np.matmul(valid.reshape((len(valid),-1)), train.reshape((len(train),-1)).transpose())
    valid_sim_to_train = np.argmax(valid_sim_to_train, 1)
    data['valid_weights'] = weights[valid_sim_to_train]

    return data
    
