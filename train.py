import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

from bic.trainer import Trainer
from utils import *
from data_utils import *
import pickle as pkl

from utils import eval_dl
from ewc import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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




# def train_bic(scenario_name, task_num, rnd_order, dataset_name, exemp_num, lr, n_epochs, seed=0):
def train_bic(ds_dict, total_cls, task_num, exemp_num, lr, n_epochs):
    batch_size = 128

    # ds_dict, task_order, im_sz, class_num, emb_fact, total_cls = get_dataset_specs_class_inc(seed=1,
    #                                                                                          task_num=10,
    #                                                                                          rnd_order=False,
    #                                                                                          dataset='split_modulation',
    #                                                                                          eval_ratio=0.1, order=None, snrs=[20])

    # save_dict = {}
    # save_dict['scenario'] = scenario_name
    # save_dict['model_type'] = 'resnet'
    # save_dict['dataset'] = 'split_cifar100'
    # save_dict['optim_name'] = 'sgd'
    # save_dict['class_num'] = class_num
    # save_dict['bs'] = batch_size
    # save_dict['lr'] = 1.
    # save_dict['n_epochs'] = n_epochs
    # save_dict['task_num'] = task_num
    # save_dict['task_order'] = task_order
    # save_dict['seed'] = seed
    # save_dict['emb_fact'] = emb_fact

    trainer = Trainer(total_cls, ds_dict=ds_dict, task_num=task_num)

    trainer.train(batch_size, n_epochs, lr, exemp_num)

    # save_dict['model'] = trainer.model.state_dict()
    # save_name = generate_save_name(save_dict)
    # pkl.dump(save_dict, open(f'{save_name}.pkl', 'wb'))
