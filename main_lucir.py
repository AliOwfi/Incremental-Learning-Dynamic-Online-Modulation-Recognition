import copy
import math
import torch
import warnings
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from models import CNN1DClassifier
from torch.utils.data import DataLoader
from data_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def eval_dl(model, dl, verbose=True): 
    model.eval()
    n_correct = 0
    n_total = 0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        
        y_hats = model(x)
        y_hat = [ ]
        for tmp in y_hats:
            y_hat.append(tmp)
        
        y_hat = torch.cat(y_hat, dim=1) 
        y_hat = torch.argmax(y_hat, dim=1)

        n_correct += torch.sum(y_hat == y).item()
        n_total += y.shape[0]

    if verbose:
        print(f'Accuracy: {n_correct / n_total * 100}')

    return n_correct / n_total * 100


class LUCIRAppr():

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=5., lamb_mr=1., dist=0.5, K=2,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False):


        # super(LUCIRAppr, self).__init__()

        self.lamb = lamb
        self.lamb_mr = lamb_mr
        self.dist = dist
        self.K = K
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda

        self.lamda = self.lamb
        self.ref_model = None
        self.model = model
        self.device = device    
        self.nepochs = nepochs  
        self.lr = lr    
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.wu_nepochs = wu_nepochs
        self.wu_lr_factor = wu_lr_factor
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset

        self.memory = {}
        self.memory['train'] = {'x':[], 'y':[]} 


        self.warmup_loss = self.warmup_luci_loss

        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        # have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        # if not have_exemplars:
            # warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

    def reduce_exemplar_nums(self, m, key):
        for cls_id in range(len(self.memory[key]['x'])):
            self.memory[key]['x'][cls_id] = self.memory[key]['x'][cls_id][:m]
            self.memory[key]['y'][cls_id] = self.memory[key]['y'][cls_id][:m]
            
        

    def store_in_mem(self, max_sz, cur_cls_num, train_ds):
        sample_per_cls = max_sz // cur_cls_num  

        x_train, y_train = train_ds.data, train_ds.targets  
    

        self.reduce_exemplar_nums(sample_per_cls, 'train')

        cls_prev = len(self.memory['train']['x'])

        for cls_id in range(cls_prev, cur_cls_num):
            train_x_cls, train_y_cls = x_train[y_train == cls_id], y_train[y_train == cls_id]   

            self.memory['train']['x'].append(train_x_cls[:sample_per_cls]) 
            self.memory['train']['y'].append(train_y_cls[:sample_per_cls])
        

        assert len(self.memory['train']['x']) == cur_cls_num
        

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


    def get_optimizer(self):
        """Returns the optimizer"""
        if self.less_forget:
            # Don't update heads when Less-Forgetting constraint is activated (from original code)
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def pre_train_process(self, t):
        """Runs before training all epochs of the task (before the train session)"""
        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if self.model.model.__class__.__name__ == 'ResNet':
                old_block = self.model.model.layer3[-1]
                self.model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            
        # Changes the new head to a CosineLinear
        self.model.heads[-1] = CosineLinear(self.model.heads[-1].in_features, self.model.heads[-1].out_features)
        self.model.to(self.device)
        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            self.model.heads[-1].sigma = self.model.heads[-2].sigma
            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.less_forget:
                for h in self.model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                self.model.heads[-1].sigma.requires_grad = True
            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lamda = self.lamb * math.sqrt(sum([h.out_features for h in self.model.heads[:-1]])
                                                   / self.model.heads[-1].out_features)
        
        self.optimizer = self.get_optimizer()   

        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        # super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        # super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self):
        """Runs after training all the epochs of the task (after the train session)"""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward current model
            outputs, features = self.model(images, return_features=True)
            # Forward previous model
            ref_outputs = None
            ref_features = None
            if t > 0:
                ref_outputs, ref_features = self.ref_model(images, return_features=True)
            loss = self.criterion(t, outputs, targets, ref_outputs, features, ref_features)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None):
        """Returns the loss value"""
        if ref_outputs is None or ref_features is None:
            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)
            # Eq. 1: regular cross entropy
            loss = nn.CrossEntropyLoss(None)(outputs, targets)
        else:
        
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                loss_dist = nn.CosineEmbeddingLoss()(features, ref_features.detach(),
                                                     torch.ones(targets.shape[0]).to(self.device)) * self.lamda
            else:
                # Scores before scale, [-1, 1]
                ref_outputs = torch.cat([ro['wosigma'] for ro in ref_outputs], dim=1).detach()
                old_scores = torch.cat([o['wosigma'] for o in outputs[:-1]], dim=1)
                num_old_classes = ref_outputs.shape[1]
                

                # Eq. 5: Modified distillation loss for cosine normalization
                loss_dist = nn.MSELoss()(old_scores, ref_outputs) * self.lamda * num_old_classes

            loss_mr = torch.zeros(1).to(self.device)
            if self.margin_ranking:
                # Scores before scale, [-1, 1]
                outputs_wos = torch.cat([o['wosigma'] for o in outputs], dim=1)
                num_old_classes = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]
                

                # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                # The index of hard samples, i.e., samples from old classes
                hard_index = targets < num_old_classes
                hard_num = hard_index.sum()
                

                if hard_num > 0:
                    # Get "ground truth" scores
                    gt_scores = outputs_wos.gather(1, targets.unsqueeze(1))[hard_index]
                    gt_scores = gt_scores.repeat(1, self.K)

                    # Get top-K scores on novel classes
                    max_novel_scores = outputs_wos[hard_index, num_old_classes:].topk(self.K, dim=1)[0]
                    # print(gt_scores.shape, max_novel_scores.shape)

                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    # Eq. 8: margin ranking loss
                    loss_mr = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1),
                                                                     max_novel_scores.view(-1, 1),
                                                                     torch.ones(hard_num * self.K, 1).to(self.device))
                    loss_mr *= self.lamb_mr

            # Eq. 1: regular cross entropy
            loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)
            
            # Eq. 9: integrated objective
        
            loss = loss_dist + loss_ce + loss_mr
        return loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)


# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
    
        if self.sigma is not None:
            out_s = self.sigma * out
        else:
            out_s = out
        if self.training:
            return {'wsigma': out_s, 'wosigma': out}
        else:
            return out_s


# This class implements a ResNet Basic Block without the final ReLu in the forward
# class BasicBlockNoRelu(nn.Module):
#     expansion = 1

#     def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
#         super(BasicBlockNoRelu, self).__init__()
#         self.conv1 = conv1
#         self.bn1 = bn1
#         self.relu = relu
#         self.conv2 = conv2
#         self.bn2 = bn2
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         # Removed final ReLU
#         return out
    

class LUCIRModel(nn.Module):
    def __init__(self, feat_ext, out_size) -> None:
        super().__init__()
        self.model = feat_ext  
        self.out_size = out_size
        self.heads = nn.ModuleList()    

    def add_head(self, num_outputs):
        self.heads.append(CosineLinear(self.out_size, num_outputs))    

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def forward(self, x, return_features=False):
        x = self.model(x)
        y = []
        for head in self.heads:
            y.append(head(x))

        if return_features:
            return y, x
        else:
            return y
        


def train_lucir(scenario_name, task_num, n_epochs, lamb, lamb_mr, exemp_num,
              seed=0, dataset_name='split_modulation', lr=1e-3, bs=16):
    

    ds_dict, task_order, im_sz, class_num, emb_fact, total_cls = get_dataset_specs_class_inc(seed=seed, task_num=task_num, 
                                                                                rnd_order=False, dataset=dataset_name,
                                                                                eval_ratio=None, order=None) 
    
    
    feat_ext = CNN1DClassifier(n_way=total_cls, indclude_head=False, last_relu=False, bias=False).to(device) 
    model = LUCIRModel(feat_ext, out_size=128).to(device)   
    appr = LUCIRAppr(model, device, n_epochs, lr, lamb=lamb, lamb_mr=lamb_mr, fix_bn=True, momentum=0.9)

    acc_mat = np.zeros((task_num, task_num))    
    known_task = 0  

    for t_id in range(task_num):
        ds_train = ds_dict['train'][t_id]
        ds_tst = ds_dict['test'][t_id]   
        if t_id > 0:
            comb_ds = appr.combine_dataset_with_exemplars(ds_train, 'train')    
        else:
            comb_ds = ds_train 

        dl_tst = torch.utils.data.DataLoader(ds_tst, batch_size=bs, shuffle=False)   
        dl_train = torch.utils.data.DataLoader(comb_ds, batch_size=bs, shuffle=True)   

        new_cls_num = len(np.unique(ds_train.targets))  
        known_task += new_cls_num            

        model.add_head(new_cls_num) 
        appr.pre_train_process(t_id)

        for epoch in range(n_epochs):
            appr.train_epoch(t_id, dl_train)
            acc = eval_dl(model, dl_tst, verbose=False)
            print(f"Epoch: {epoch}, acc: {acc}")

        appr.store_in_mem(exemp_num, known_task, ds_train)  
        appr.post_train_process()   

        for t_tst_id in range(t_id+1):
            ds_tst = ds_dict['test'][t_tst_id]  
            dl_tst = torch.utils.data.DataLoader(ds_tst, batch_size=bs, shuffle=False)   
            acc = eval_dl(model, dl_tst, verbose=False)
            acc_mat[t_id, t_tst_id] = acc
        
        with np.printoptions(precision=2, suppress=True):
            print(acc_mat)    

        acc_ds = combine_acc_datasets(ds_dict['test'], t_id)
        acc_dl = torch.utils.data.DataLoader(acc_ds, batch_size=bs, shuffle=False)  
        acc_comb = eval_dl(model, acc_dl, verbose=False)    
        print(f"Accumulative acc: {acc_comb}")



train_lucir('lucir_modulation', task_num=5, n_epochs=10, seed=0, lamb=1., lamb_mr=1., exemp_num=2000,   
           dataset_name='split_modulation', lr=1e-3, bs=16)