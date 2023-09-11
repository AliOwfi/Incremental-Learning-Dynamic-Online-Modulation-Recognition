import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import  StepLR
from data_utils import CustomTenDataset, cosntruct_accumulative_ds
import numpy as np

from models import BiasLayer, CNN1DClassifier
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


class BICTrainer:
    def __init__(self, total_cls, ds_dict, task_num=10):
        self.seen_cls = 0
        self.task_num = task_num    
        self.dataset = ds_dict
        self.acc_lst = []   
        self.task_cls = []

        self.model = CNN1DClassifier(total_cls, indclude_head=True)
        self.model.to("cuda")
        
        self.bias_layers = nn.ModuleList()

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.memory = {'train': {'x': [], 'y': []}, 'val': {'x': [], 'y': []}}

        print("Solver total trainable parameters : ", total_params)

    def test(self, testdata):
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            pred = p.argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc

    def eval(self, criterion, evaldata):
        self.model.eval()
        losses = []
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(evaldata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p, label)
            losses.append(loss.item())
            pred = p.argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        print("Validation Loss: {}".format(np.mean(losses)))
        print("Validation Acc: {}".format(100*correct/(correct+wrong)))
        self.model.train()
        return

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def reduce_exemplar_nums(self, m, key):
        for cls_id in range(len(self.memory[key]['x'])):
            self.memory[key]['x'][cls_id] = self.memory[key]['x'][cls_id][:m]
            self.memory[key]['y'][cls_id] = self.memory[key]['y'][cls_id][:m]

    def store_in_mem(self, max_sz, cur_cls_num, train_ds, eval_ds, eval_ratio=.1): #TODO: ERROR random order
        sample_per_cls = max_sz // cur_cls_num  

        x_train, y_train = train_ds.data, train_ds.targets  
        x_eval, y_eval = eval_ds.data, eval_ds.targets  

        val_store_num = int(sample_per_cls * eval_ratio)    
        train_store_num = sample_per_cls - val_store_num

        self.reduce_exemplar_nums(train_store_num, 'train')
        self.reduce_exemplar_nums(val_store_num, 'val')

        cls_prev = len(self.memory['train']['x'])

        for cls_id in range(cls_prev, cur_cls_num):
            train_x_cls, train_y_cls = x_train[y_train == cls_id], y_train[y_train == cls_id]   
            val_x_cls, val_y_cls = x_eval[y_eval == cls_id], y_eval[y_eval == cls_id]

            self.memory['train']['x'].append(train_x_cls[:train_store_num]) 
            self.memory['train']['y'].append(train_y_cls[:train_store_num])
            self.memory['val']['x'].append(val_x_cls[:val_store_num])
            self.memory['val']['y'].append(val_y_cls[:val_store_num])

        assert len(self.memory['train']['x']) == cur_cls_num
        assert len(self.memory['val']['x']) == cur_cls_num

    def combine_dataset_with_exemplars(self, dataset, key):
        if len(self.memory['train']['x']) == 0:
            return dataset

        x_ten, y_ten = [], []   
        for cls_id in range(len(self.memory[key]['x'])):
            x_ten.append(self.memory[key]['x'][cls_id])
            y_ten.append(self.memory[key]['y'][cls_id]) 

        x_ten = torch.cat(x_ten, dim=0)
        y_ten = torch.cat(y_ten, dim=0)

        x_ten = torch.cat([x_ten, dataset.data], dim=0) 
        y_ten = torch.cat([y_ten, dataset.targets], dim=0)  

        new_ds = CustomTenDataset(x_ten, y_ten) 
        
        return new_ds   
    
    def get_memory_ds(self, key):
        x_ten, y_ten = [], []
        for cls_id in range(len(self.memory[key]['x'])):
            x_ten.append(self.memory[key]['x'][cls_id])
            y_ten.append(self.memory[key]['y'][cls_id]) 
            

        x_ten = torch.cat(x_ten, dim=0)
        y_ten = torch.cat(y_ten, dim=0)

        new_ds = CustomTenDataset(x_ten, y_ten) 
        
        return new_ds


    def train(self, batch_size, epoches, lr, max_size):
        
        criterion = nn.CrossEntropyLoss()

        previous_model = None

        dataset = self.dataset
        test_xs = []
        test_ys = []
        train_xs = []
        train_ys = []

        test_accs = []
        for inc_i in range(self.task_num):
            print(f"Incremental num : {inc_i}")
            curr_task_cls_num = len(dataset['train'][inc_i].targets.unique())   
            self.task_cls.append((self.seen_cls, self.seen_cls + curr_task_cls_num))    
            self.seen_cls += curr_task_cls_num  
            self.bias_layers.append(BiasLayer().to(device))

            train_ds, val_ds, _ = dataset['train'][inc_i], dataset['val'][inc_i], dataset['test'][inc_i]
            train_ds = self.combine_dataset_with_exemplars(train_ds, 'train')

            train_data = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            acc_test_ds = cosntruct_accumulative_ds(dataset['test'], inc_i) 
            test_data = DataLoader(acc_test_ds, batch_size=batch_size, shuffle=False)

            self.store_in_mem(max_size, self.seen_cls, train_ds=dataset['train'][inc_i], eval_ds=dataset['val'][inc_i], eval_ratio=.1)

            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            bias_optimizer = optim.SGD(self.bias_layers[inc_i].parameters(), lr=lr, momentum=0.9)
            
            print("seen cls number : ", self.seen_cls)

            print("test:", self.get_memory_ds('val'))
            val_mem_ds = self.get_memory_ds('val')   

            print(val_mem_ds.data.shape)
            val_bias_data = DataLoader(val_mem_ds, batch_size=100, shuffle=True, drop_last=False)

            test_acc = []

            for epoch in range(epoches):
                print("---")
                print("Epoch", epoch)

                cur_lr = self.get_lr(optimizer)
                print("Current Learning Rate : ", cur_lr)
                self.model.train()
                for _ in range(len(self.bias_layers)):
                    self.bias_layers[_].eval()
                if inc_i > 0:
                    self.stage1_distill(train_data, criterion, optimizer)
                else:
                    self.stage1(train_data, criterion, optimizer)
                acc = self.test(test_data)
            if inc_i > 0:
                for epoch in range(epoches):
                    # bias_scheduler.step()
                    self.model.eval()
                    for _ in range(len(self.bias_layers)):
                        self.bias_layers[_].train()
                    self.stage2(val_bias_data, criterion, bias_optimizer)
                    if epoch % 50 == 0:
                        acc = self.test(test_data)
                        test_acc.append(acc)
                        
            for i, layer in enumerate(self.bias_layers):
                layer.printParam(i)

            self.previous_model = deepcopy(self.model)
            acc = self.test(test_data)
            test_acc.append(acc)
            self.acc_lst.append(max(test_acc)*100)
            print(self.acc_lst)

    def bias_forward(self, input):

        outs = []   
        
        for i in range(len(self.task_cls)):
            input_tmp = input[:, self.task_cls[i][0]:self.task_cls[i][1]]   
            outs.append(self.bias_layers[i](input_tmp)) 

        outs = torch.cat(outs, dim=1)   
        
        return outs

    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(train_data):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p, label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2

        cur_task_cls = self.task_cls[-1][1] - self.task_cls[-1][0]

        alpha = (self.seen_cls - cur_task_cls)/ self.seen_cls
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(train_data):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p = self.previous_model(image)
                pre_p = self.bias_forward(pre_p)
                pre_p = F.softmax(pre_p[:, :self.seen_cls-cur_task_cls]/T, dim=1)

            logp = F.log_softmax(p[:,:self.seen_cls-cur_task_cls]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p, label)
            loss = loss_soft_target * alpha + (1-alpha) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))

    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(train_data):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage2(self, val_bias_data, criterion, optimizer):
        print("Evaluating ... ")
        losses = []
        for i, (image, label) in enumerate(val_bias_data):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))
