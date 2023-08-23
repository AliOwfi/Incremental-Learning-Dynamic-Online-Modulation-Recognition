from torchvision import datasets, transforms    
import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_cil_mnist(task_order):
    ds_train = datasets.MNIST('./data', train=True, download=True)    
    ds_test = datasets.MNIST('./data', train=False, download=True)

    if task_order == None:
        task_order = np.arange(10).reshape(5, 2)
    
    taks_num = task_order.shape[0]  
    ds_dict = {}
    ds_dict['train'] = []   
    ds_dict['test'] = []    

    for task_ind in range(taks_num):  
        train_data_ = []
        test_data_ = [] 
        train_lbls_ = []    
        test_lbls_ = []    
        for c in task_order[task_ind]:
            train_idx = ds_train.targets == c   
            test_idx = ds_test.targets == c 

            train_data_.append(ds_train.data[train_idx])   
            test_data_.append(ds_test.data[test_idx])   
            train_lbls_.append(ds_train.targets[train_idx])
            test_lbls_.append(ds_test.targets[test_idx])

        train_data_ = torch.cat(train_data_, dim=0).float() / 255.
        test_data_ = torch.cat(test_data_, dim=0).float() / 255.
        train_lbls_ = torch.cat(train_lbls_, dim=0)
        test_lbls_ = torch.cat(test_lbls_, dim=0)    

        ds_dict['train'].append(TensorDataset(train_data_, train_lbls_))    
        ds_dict['test'].append(TensorDataset(test_data_, test_lbls_))   



    return ds_dict  

        
# get_cil_mnist(None) 
