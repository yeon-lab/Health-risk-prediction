import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torchmetrics import AUROC



def accuracy(target, output, out_dim):
    target = target.cpu().numpy().reshape(-1,1)
    if out_dim == 1:
        output = torch.where(output > 0.5, 1, 0).cpu().numpy().reshape(-1,1)
    else:
        output = output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1,1)
    return accuracy_score(target, output)

    
def roc_auc(target, output, out_dim):
    target = target.long()
    if out_dim == 1:
        auroc = AUROC(pos_label=1)
    else:
        auroc = AUROC(num_classes = out_dim)
    return auroc(output, target).cpu().numpy()

def f1(target, output, out_dim):
    target = target.cpu().numpy().reshape(-1,1)
    if out_dim == 1:
        output = torch.where(output > 0.5, 1, 0).cpu().numpy().reshape(-1,1)
    else:
        output = output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1,1)
    return f1_score(target, output, average='macro')
    
    
def confusion(target, output, out_dim):
    target = target.cpu().numpy().reshape(-1,1)
    if out_dim == 1:
        output = torch.where(output > 0.5, 1, 0).cpu().numpy().reshape(-1,1)
    else:
        output = output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1,1)
    return confusion_matrix(target,output)
