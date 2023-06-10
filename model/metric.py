import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc
from torchmetrics import AUROC

def accuracy(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = torch.where(output > 0.5, 1, 0).cpu().numpy().reshape(-1,1)
    return accuracy_score(target, output)

def auroc(target, output):
    target = target.long()
    auroc_func = AUROC(pos_label=1)
    return auroc_func(output, target).cpu().numpy()
    
def auprc(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = output.detach().cpu().numpy().reshape(-1,1)
    precision, recall, thresholds = precision_recall_curve(target, output)
    return auc(recall, precision)
    
def precision(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = torch.where(output > 0.5, 1, 0).cpu().numpy().reshape(-1,1)
    return precision_score(target, output, average='macro')
    
def recall(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = torch.where(output > 0.5, 1, 0).cpu().numpy().reshape(-1,1)
    return recall_score(target, output, average='macro')
    
def confusion(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = torch.where(output > 0.5, 1, 0).cpu().numpy().reshape(-1,1)
    return confusion_matrix(target,output)
