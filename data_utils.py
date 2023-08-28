from torchvision import datasets, transforms    
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset    
from torchvision.datasets import CIFAR100



class CustomTenDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data = data_tensor
        self.targets = target_tensor
        
        # Check if the number of samples in data and targets match
        assert len(self.data) == len(self.targets), "Data and target tensors must have the same length."
        
    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        
        return data, target
    
    def __len__(self):
        return len(self.data)



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


def generate_split_cifar100_tasks_class_inc(task_num, seed=0, rnd_order=True, order=None ):
    np.random.seed(seed)    
    torch.manual_seed(seed) 

    if rnd_order:
        rnd_cls_order = np.random.permutation(100)
    else:
        rnd_cls_order = order
        
    tasks_cls = []

    cls_per_task = 100 // task_num  
    for i in range(task_num):
        tasks_cls.append(rnd_cls_order[i*cls_per_task:(i+1)*cls_per_task])
    
    ds_train = CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    ds_tst = CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
    ds_train.targets = torch.tensor(ds_train.targets)   
    ds_tst.targets = torch.tensor(ds_tst.targets)   

    ds_dict = {}
    ds_dict['train'] = []
    ds_dict['test'] = []
    
    for i in range(task_num):
        train_task_idx_ = []
        tst_task_idx_ = []
        train_task_idx = torch.zeros(len(ds_train.targets)).bool()  
        tst_task_idx = torch.zeros(len(ds_tst.targets)).bool()
        for j in range(cls_per_task):
            train_task_idx_.append(ds_train.targets == tasks_cls[i][j])  
            tst_task_idx_.append(ds_tst.targets == tasks_cls[i][j])  
    
            train_task_idx = torch.logical_or(train_task_idx, train_task_idx_[-1])  
            tst_task_idx = torch.logical_or(tst_task_idx, tst_task_idx_[-1])

        x_train_task = ds_train.data[train_task_idx] / 255. 
        y_train_task = ds_train.targets[train_task_idx]

        x_tst_task = ds_tst.data[tst_task_idx] / 255.
        y_tst_task = ds_tst.targets[tst_task_idx]
    
        y_train_task = torch.tensor(y_train_task)   
        y_tst_task = torch.tensor(y_tst_task)
        x_train_task = torch.tensor(x_train_task).permute(0, 3, 1, 2).float()
        x_tst_task = torch.tensor(x_tst_task).permute(0, 3, 1, 2).float()   

        ds_dict['train'].append(CustomTenDataset(x_train_task, y_train_task))  
        ds_dict['test'].append(CustomTenDataset(x_tst_task, y_tst_task))


        # print('Task {} has {} classes'.format(i, np.unique(y_train_task))) 


    return ds_dict, tasks_cls


def get_dataset_specs_class_inc(**kwargs):
    emb_fact = 1  
    
    if kwargs['dataset'] == 'split_cifar100':  
        order = np.arange(100)  
        ds_dict, task_order = generate_split_cifar100_tasks_class_inc(10, seed=kwargs['seed'], 
                                                            order=order, rnd_order=False )

        im_sz=32
        class_num = 10


    return ds_dict, task_order, im_sz, class_num, emb_fact


# get_cil_mnist(None) 
