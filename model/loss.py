import torch
import torch.nn as nn
       
def BCELoss(output, target):
    cr = nn.BCELoss(reduction='none')
    return cr(output, target)       
       

