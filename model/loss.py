import torch
import torch.nn as nn


def CrossEntropyLoss(output, target):
    cr = nn.CrossEntropyLoss()
    return cr(output, target)
       
def BCELoss(output, target):
    cr = nn.BCELoss(reduction='none')
    return cr(output, target)       
       
