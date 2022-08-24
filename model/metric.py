import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score


'''
def classification_report_(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1,1)
    return classification_report(target,output)
'''

def accuracy(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1,1)
    return accuracy_score(target, output)


def roc_auc(target, output):
    target = target.unsqueeze(1)
    y_true_oh = torch.zeros(output.shape).cuda().scatter_(1, target, 1)
    auc = roc_auc_score(y_true=y_true_oh.detach().cpu().numpy(), y_score=output.detach().cpu().numpy(), average=None)
    return auc

def f1(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1,1)
    return f1_score(target, output, average='macro')
    
    
def confusion(target, output):
    target = target.cpu().numpy().reshape(-1,1)
    output = output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1,1)
    return confusion_matrix(target,output )
