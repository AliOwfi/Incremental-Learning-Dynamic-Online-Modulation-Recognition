import torch
import numpy as np
import random
from resent import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


def eval_dl(model, dl, verbose=True, task_id=-1, class_inc=False): 
    model.eval()
    n_correct = 0
    n_total = 0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        if class_inc == False:
            y_hat = model(x)[task_id]
            y_hat = torch.argmax(y_hat, dim=1)
        else:
            y_hat = model.classify(x, mode=0)

        n_correct += torch.sum(y_hat == y).item()
        n_total += y.shape[0]

    if verbose:
        print(f'Accuracy: {n_correct / n_total * 100}')

    return n_correct / n_total * 100


def generate_save_name(save_dict):
    name = ''
    for k in save_dict.keys():
        if type(save_dict[k]) == float or type(save_dict[k]) == int:
            name += f'{k}_{str(save_dict[k])}'  
        elif type(save_dict[k]) == str:
            name += f'{k}_{save_dict[k]}'
        elif k == 'arch':
            name += f'{k}_{"_".join([str(i) for i in save_dict[k]])}'
        
        name += '_'

    return name[:-1]


def create_model_class_inc(**kwargs):
    
    if kwargs['model_type'] == 'resnet':    
        model = ResNet18(task_num=kwargs['task_num'], nclasses=kwargs['class_num'], include_head=kwargs['include_head'],
                         nf=kwargs['nf'], final_feat_sz=kwargs['final_feat_sz']).to(device)
        
    elif kwargs['model_type'] == 'cnn1d':
        model = CNN1DClassifier(n_way=kwargs['class_num'], indclude_head=False).to(device)   
    

    return model


def create_optimizer(model, optim_name, lr=1e-3, weight_decay=0):
    if optim_name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optim_name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)   

    return optim    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
