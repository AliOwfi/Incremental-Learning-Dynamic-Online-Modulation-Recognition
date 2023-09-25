import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from il_algorithms.bic import BICTrainer
from utils import *
from dataloader import *
from utils import eval_dl
from ewc import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_cl(method, task_order, snrs: list, n_epochs, bs, lr, seed, save=False):
    set_seed(seed)
    dict_ds = get_cil_datasets(classes_order=task_order, snrs=snrs)

    total_cls = task_order.shape[0]*task_order.shape[1]
    task_num = task_order.shape[0]

    print("Dataset created")

    if method == 'icarl':
        exemp_num = 2000
        acc_mat, accumulative_acc_lst = train_icarl(dict_ds, task_order, n_epochs=n_epochs, model_type='cnn1d',
                                                    lr=lr, optim_name='sgd', bs=bs, emb_dim=128, exemp_num=exemp_num)
    elif method == 'bic':
        exemp_num = 2000

        train_bic(ds_dict=dict_ds, total_cls=total_cls, task_num=task_num, exemp_num=exemp_num, lr=lr,
                  n_epochs=n_epochs, bs=bs)

    elif method == 'conventional':
        class_num = 2

        model = CNN1DClassifier(n_way=20).to(device)
        acc_mat, accumulative_acc_lst = train_conventional(model, dict_ds, batch_size=bs, epochs=n_epochs, lr=lr)
    else:
        raise Exception("CL method not known")
    if save:
        save_data_dict = {'acc_mat': acc_mat,
                          'accumulative_acc_lst': accumulative_acc_lst,
                          'task_order': task_order,
                          'snrs': snrs,
                          'lr': lr,
                          'bs': bs,
                          'exemp_num': exemp_num,
                          'method': method
                          }
        file_name = f'{method}_{snrs}_epoch{n_epochs}_bs{bs}_seed{seed}_{task_order}.pkl'
        with open(f'results/{file_name}', 'wb') as fp:
            pickle.dump(save_data_dict, fp)


def train_conventional(model, dict_ds, batch_size, epochs=10, lr=0.0001):
    task_num = len(dict_ds['train'])
    criterion = nn.CrossEntropyLoss()

    acc_mat = np.zeros((task_num, task_num))

    for task_ind in range(task_num):
        dl_train = DataLoader(dict_ds['train'][task_ind], batch_size=batch_size, shuffle=True)
        dl_tst = DataLoader(dict_ds['test'][task_ind], batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_ = []

        for epoch in range(epochs):
            model.train()
            for batch in dl_train:
                batch['sig'], batch['modulation'] = batch['sig'].to(device), batch['modulation'].to(device)

                optimizer.zero_grad()
                pred = model(batch['sig'])
                loss = criterion(pred, batch['modulation'])
                loss.backward()
                optimizer.step()
                loss_.append(loss.item())

            acc = eval_dl(model, dl_tst, verbose=False)
            print(f'task {task_ind} epoch {epoch} loss: {loss.item()} acc: {acc}')

        for task_ind_tst in range(task_ind + 1):
            dl_tst = DataLoader(dict_ds['test'][task_ind_tst], batch_size=batch_size, shuffle=True)
            acc = eval_dl(model, dl_tst, verbose=False)
            acc_mat[task_ind, task_ind_tst] = acc

            print(f"train_task {task_ind} test_task {task_ind_tst} acc {acc}")

        with np.printoptions(precision=2, suppress=True):
            print(acc_mat)

    avg_acc = np.mean(np.mean(acc_mat[-1]))
    print(f'avg acc: {avg_acc} ')


def train_ewc(model, dict_ds, batch_size, epochs=10, lr=0.0001):
    w_ewc = 1000

    task_num = len(dict_ds['train'])
    criterion = nn.CrossEntropyLoss()

    acc_mat = np.zeros((task_num, task_num))


    loss_ewc = torch.tensor(0)

    for task_ind in range(task_num):
        ds_train = dict_ds['train'][task_ind]
        ds_tst = dict_ds['test'][task_ind]
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dl_tst = DataLoader(ds_tst, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        loss_ = []
        for epoch in range(epochs):
            model.train()
            for batch in dl_train:
                batch['sig'], batch['modulation'] = batch['sig'].to(device), batch['modulation'].to(device)

                pred = model(batch['sig'])

                loss_curr_task = criterion(pred, batch['modulation'])

                if task_ind > 0:
                    loss_ewc = compute_ewc_loss(model)

                loss = loss_curr_task + w_ewc * loss_ewc

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                loss_.append(loss.item())

            acc = eval_dl(model, dl_tst, verbose=False)
            print(f'task {task_ind} epoch {epoch} loss: {loss.item()} acc: {acc} loss_ewc: {loss_ewc.item()}')

        register_ewc_params(model, dl_train, mh=False)
        #         save_dict['model'] = model.state_dict()

        for task_ind_tst in range(task_ind + 1):
            ds_tst = dict_ds['test'][task_ind_tst]
            dl_tst = DataLoader(ds_tst, batch_size=batch_size, shuffle=True)
            acc = eval_dl(model, dl_tst, verbose=False)
            acc_mat[task_ind, task_ind_tst] = acc

            print(f"train_task {task_ind} test_task {task_ind_tst} acc {acc}")

        with np.printoptions(precision=2, suppress=True):
            print(acc_mat)

    avg_acc = np.mean(np.mean(acc_mat[-1]))
    bwt = np.mean((acc_mat[-1] - np.diag(acc_mat))[:-1])

    print(f'avg acc: {avg_acc}')


def train_bic(ds_dict, total_cls, task_num, exemp_num, lr, n_epochs, bs):

    trainer = BICTrainer(total_cls, ds_dict=ds_dict, task_num=task_num)

    trainer.train(bs, n_epochs, lr, exemp_num)

