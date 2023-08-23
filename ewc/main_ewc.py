
import torch 
import torch.nn.functional as F 
import numpy as np  
from models import *
from data_utils import *
from torch.utils.data import DataLoader
import pickle as pkl    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def register_ewc_params(model, dl, return_fishers=False, mh=False, task_id=-1):

    norm_fact = len(dl)
    model.eval()
    
    for b_ind, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        if mh == False:
            preds = model(x)
        else:
            preds = model(x)[task_id]

        loss = F.nll_loss(F.log_softmax(preds, dim=1), y)

        model.zero_grad()
        loss.backward()

        tmp_fisher = []

        for p_ind, (n, p) in enumerate(model.named_parameters()):
            if 'heads' in n:
                continue

            if hasattr(model, f"fisher_{n.replace('.', '_')}"):
                current_fisher = getattr(model, f"fisher_{n.replace('.', '_')}")
            else:
                current_fisher = 0

            new_fisher = current_fisher + p.grad.detach() ** 2 / norm_fact 
            
            model.register_buffer(f"fisher_{n.replace('.', '_')}", new_fisher)
            tmp_fisher.append(p.grad.detach() ** 2 / norm_fact )

    for p_ind, (n, p) in enumerate(model.named_parameters()):
        if 'heads' in n:
            continue
        
        model.register_buffer(f"mean_{n.replace('.', '_')}", p.data.clone())

    model.zero_grad()
    if return_fishers:
        return tmp_fisher
    

def register_blank_ewc_params(model):
    
    for p_ind, (n, p) in enumerate(model.named_parameters()):
        if 'heads' in n:
            continue

        model.register_buffer(f"fisher_{n.replace('.', '_')}", torch.zeros_like(p))
        model.register_buffer(f"mean_{n.replace('.', '_')}", torch.zeros_like(p))
        
    model.zero_grad()
    


def compute_ewc_loss(model):
    loss = 0
    for n, p in model.named_parameters():
        if 'heads' in n:
                continue
        
        loss += (getattr(model, f"fisher_{n.replace('.', '_')}") * \
            (p - getattr(model, f"mean_{n.replace('.', '_')}")).pow(2)).sum()

    return loss / 2.



def eval_dl(model, dl, verbose=True): 
    model.eval()
    n_correct = 0
    n_total = 0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)
        y_hat = torch.argmax(y_hat, dim=1)
        n_correct += torch.sum(y_hat == y).item()
        n_total += y.shape[0]

    if verbose:
        print(f'Accuracy: {n_correct / n_total * 100}')

    return n_correct / n_total * 100



def train_ewc(scenario_name, seed=0):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    bs = 16
    task_num = 5
    n_epochs = 10
    lr = 1e-3
    w_ewc = 1000

    
    model = MLP([784, 256, 256, 10]).to(device)  
    model.train()     


    loss_fn = torch.nn.CrossEntropyLoss()

    acc_mat = np.zeros((task_num, task_num))    

    save_dict = {}  
    save_dict['scenario'] = scenario_name
    save_dict['model_name'] = model.__class__.__name__
    save_dict['task_num'] = task_num    
    save_dict['w_ewc'] = w_ewc
    save_dict['seed'] = seed    

    cont_method_args = {'w_ewc': w_ewc} 
    save_dict['cont_method_args'] = cont_method_args    

    ds_dict = get_cil_mnist(None)   

    loss_ewc = torch.tensor(0)  

    for task_ind in range(task_num): 
        ds_train = ds_dict['train'][task_ind]
        ds_tst = ds_dict['test'][task_ind]
        dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)    
        dl_tst = DataLoader(ds_tst, batch_size=bs, shuffle=True)

        optim = torch.optim.Adam(model.parameters(), lr=lr) 

        loss_ = []
        for epoch in range(n_epochs):
            model.train()
            for i, (x, y) in enumerate(dl_train):
                x, y = x.to(device), y.to(device)                
                
                y_hat = model(x)

                loss_curr_task = loss_fn(y_hat, y)

                if task_ind > 0:
                    loss_ewc = compute_ewc_loss(model)

                loss = loss_curr_task + w_ewc * loss_ewc

                optim.zero_grad()
                loss.backward()
                
                optim.step()

                loss_.append(loss.item())
            
            acc = eval_dl(model, dl_tst, verbose=False)    
            print(f'task {task_ind} epoch {epoch} loss: {loss.item()} acc: {acc} loss_ewc: {loss_ewc.item()}')

        register_ewc_params(model, dl_train, mh=False)
        save_dict['model'] = model.state_dict()

        for task_ind_tst in range(task_ind+1):
            ds_tst = ds_dict['test'][task_ind_tst]
            dl_tst = DataLoader(ds_tst, batch_size=bs, shuffle=True)
            acc = eval_dl(model, dl_tst, verbose=False)
            acc_mat[task_ind, task_ind_tst] = acc

        with np.printoptions(precision=2, suppress=True):
            print(acc_mat)

    avg_acc = np.mean(np.mean(acc_mat[-1]))
    bwt = np.mean((acc_mat[-1] - np.diag(acc_mat))[:-1])

    print(f'avg acc: {avg_acc} bwt: {bwt}')

    save_dict['acc_mat'] = acc_mat
    save_dict['avg_acc'] = avg_acc
    save_dict['bwt'] = bwt
    save_dict['model'] = model.state_dict()
    save_dict['optim'] = optim.state_dict()
    
    
    pkl.dump(save_dict, open(f'{scenario_name}.pkl', 'wb'))


if __name__ == '__main__':
    scenario_name = 'ewc_mnist_cil'
    train_ewc(scenario_name, seed=0)      
