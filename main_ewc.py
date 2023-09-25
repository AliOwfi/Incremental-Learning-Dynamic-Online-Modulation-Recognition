from models import *    
import torch
from data_utils import *  
import torch.nn.functional as F 
  


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

class EWCModel(torch.nn.Module):    
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model    
        
        self.memory = {}
        self.memory['train'] = {'x':[], 'y':[]} 
        
    def reduce_exemplar_nums(self, m, key):
        for cls_id in range(len(self.memory[key]['x'])):
            self.memory[key]['x'][cls_id] = self.memory[key]['x'][cls_id][:m]
            self.memory[key]['y'][cls_id] = self.memory[key]['y'][cls_id][:m]

    def sample_exemplars(self, bs, key): 
        x_tensors, y_tensors = [], []   
        for cls_id in range(len(self.memory[key]['x'])):    
            x_cls, y_cls = self.memory[key]['x'][cls_id], self.memory[key]['y'][cls_id]   
            
            x_tensors.append(x_cls)
            y_tensors.append(y_cls) 
        
        x_tensors = torch.cat(x_tensors, dim=0) 
        y_tensors = torch.cat(y_tensors, dim=0) 

        rnd_idx = torch.randperm(len(x_tensors))[:bs]   
        x_tensors, y_tensors = x_tensors[rnd_idx], y_tensors[rnd_idx]   
        return x_tensors, y_tensors 


            

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
        

    def forward(self, x):
        x = self.model(x)
        return x    


def register_ewc_params(model, dl, known_tasks, return_fishers=False):

    norm_fact = len(dl)
    model.eval()
    
    for b_ind, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        preds = model(x)[:, :known_tasks]   

        loss = F.nll_loss(F.log_softmax(preds, dim=1), y)

        model.zero_grad()
        loss.backward()

        tmp_fisher = []

        for p_ind, (n, p) in enumerate(model.named_parameters()):
    
            if hasattr(model, f"fisher_{n.replace('.', '_')}"):
                current_fisher = getattr(model, f"fisher_{n.replace('.', '_')}")
            else:
                current_fisher = 0

            new_fisher = current_fisher + p.grad.detach() ** 2 / norm_fact 
            
            model.register_buffer(f"fisher_{n.replace('.', '_')}", new_fisher)
            tmp_fisher.append(p.grad.detach() ** 2 / norm_fact )

    for p_ind, (n, p) in enumerate(model.named_parameters()):
        
        
        model.register_buffer(f"mean_{n.replace('.', '_')}", p.data.clone())

    model.zero_grad()
    if return_fishers:
        return tmp_fisher
    

def register_blank_ewc_params(model):
    
    for p_ind, (n, p) in enumerate(model.named_parameters()):
    
        model.register_buffer(f"fisher_{n.replace('.', '_')}", torch.zeros_like(p))
        model.register_buffer(f"mean_{n.replace('.', '_')}", torch.zeros_like(p))
        
    model.zero_grad()
    


def compute_ewc_loss(model):
    loss = 0
    for n, p in model.named_parameters():
        
        loss += (getattr(model, f"fisher_{n.replace('.', '_')}") * \
            (p - getattr(model, f"mean_{n.replace('.', '_')}")).pow(2)).sum()

    return loss / 2.


def eval_dl(model, dl, known_tasks, verbose=True): 
    model.eval()
    n_correct = 0
    n_total = 0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)[:, :known_tasks]   
        y_hat = torch.argmax(y_hat, dim=1)  
        n_correct += torch.sum(y_hat == y).item()
        n_total += y.shape[0]

    if verbose:
        print(f'Accuracy: {n_correct / n_total * 100}')

    return n_correct / n_total * 100


def train_ewc(scenario_name, task_num, n_epochs, w_ewc, 
              seed=0, dataset_name='split_modulation', lr=1e-3, bs=16):

    ds_dict, task_order, im_sz, class_num, emb_fact, total_cls = get_dataset_specs_class_inc(seed=seed, task_num=task_num, 
                                                                                rnd_order=False, dataset=dataset_name,
                                                                                eval_ratio=None, order=None) 
    
    
    feat_ext = CNN1DClassifier(n_way=total_cls, indclude_head=True).to(device) 
    model = EWCModel(feat_ext).to(device)   

    criterion = torch.nn.CrossEntropyLoss() 

    known_task = 0
    acc_mat = np.zeros((task_num, task_num))    
    
    for t_id in range(task_num):
        ds_train = ds_dict['train'][t_id]
        ds_tst = ds_dict['test'][t_id]   
        dl_tst = torch.utils.data.DataLoader(ds_tst, batch_size=bs, shuffle=False)   
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=True)   

        new_cls_num = len(np.unique(ds_train.targets))  
        known_task += new_cls_num            

        optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

        for epoch in range(n_epochs):
            model.train()   
            for b_idx, (x, y) in enumerate(dl_train):
                x, y = x.to(device), y.to(device)   

                logits = model(x)[:, :known_task]
                loss_ce = criterion(logits, y) 

                loss_ewc = torch.tensor(0.).to(device)    
                if t_id > 0:    
                    loss_ewc = compute_ewc_loss(model)
                    
                loss = loss_ce + w_ewc * loss_ewc

                optimizer.zero_grad() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)   
                optimizer.step()    
            
            acc = eval_dl(model, dl_tst, verbose=False, known_tasks=known_task)
            print(f"Epoch: {epoch}, loss ce: {loss_ce.item()}, loss ewc: {loss_ewc.item()}, acc: {acc}")
            

        # sample_per_cls = exemp_num // known_task    
        # model.reduce_exemplar_nums(sample_per_cls, 'train') 
        # model.store_in_mem(exemp_num, known_task, ds_train )
        register_ewc_params(model, dl_train, known_tasks=known_task)    


        for t_tst_id in range(t_id+1):
            ds_tst = ds_dict['test'][t_tst_id]  
            dl_tst = torch.utils.data.DataLoader(ds_tst, batch_size=bs, shuffle=False)   
            acc = eval_dl(model, dl_tst, verbose=False, known_tasks=known_task)
            acc_mat[t_id, t_tst_id] = acc
        
        with np.printoptions(precision=2, suppress=True):
            print(acc_mat)    

        acc_ds = combine_acc_datasets(ds_dict['test'], t_id)
        acc_dl = torch.utils.data.DataLoader(acc_ds, batch_size=bs, shuffle=False)  
        acc_comb = eval_dl(model, acc_dl, verbose=False, known_tasks=known_task)    
        print(f"Accumulative acc: {acc_comb}")


train_ewc('ewc_modulation', task_num=5, n_epochs=2, w_ewc=1000, seed=0,
           dataset_name='split_modulation', lr=1e-3, bs=16)


        