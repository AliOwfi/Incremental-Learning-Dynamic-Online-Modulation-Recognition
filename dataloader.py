import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py


class SignalDataset(Dataset):
    """
        {'sig': array([[-0.66901106,  0.6191937 ],
            [ 0.17118636, -0.08626155],
            [ 0.4972762 ,  0.6710627 ],
            ...,
            [ 1.1782482 ,  0.05399285],
            [ 1.9242011 ,  0.9850999 ],
            [ 0.09299215,  0.38277483]], dtype=float32),
         'modulation': 23,
         'snr': -20}
    """

    def __init__(self, snrs: list, modulation_ids: list, file_path):
        self.modulation_ids = modulation_ids
        self.snrs = snrs

        f = h5py.File(file_path)

        self.list_signals = []
        self.data = []
        self.targets = []

        for modulation_id in modulation_ids:
            for snr in snrs:
                for signal_index in range(modulation_id * 106496 + int((snr + 20) / 2) * 4096,
                                          modulation_id * 106496 + (int((snr + 20) / 2) + 1) * 4096):
                    if np.argmax(f['Y'][signal_index]) == modulation_id and f['Z'][signal_index] == snr:
                        self.list_signals.append({'sig': torch.tensor(f['X'][signal_index].T),
                                                  'modulation': torch.tensor(np.argmax(f['Y'][signal_index])),
                                                  'snr': f['Z'][signal_index][0]})

                        self.data.append(torch.tensor(f['X'][signal_index].T))
                        self.targets.append(torch.tensor(np.argmax(f['Y'][signal_index])))

    def __len__(self):
        return len(self.list_signals)

    def __getitem__(self, idx):
        return self.list_signals[idx]


class CustomTenDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform=None):
        self.data = data_tensor
        self.targets = target_tensor

        assert len(self.data) == len(self.targets), "Data and target tensors must have the same length."

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, target


def ds_random_split(ds, test_ratio=.1, val_ratio=.01):
    data_x, data_y = ds.data, ds.targets
    test_num = int(len(data_x) * test_ratio)
    val_num = int(len(data_x) * val_ratio)
    prms = torch.randperm(len(data_x))

    # test_x, test_y = torch.stack(data_x[:test_num]), torch.stack(data_y[:test_num])
    # trian_x, train_y = torch.stack(data_x[test_num:]), torch.stack(data_y[test_num:])

    train_x = torch.stack([data_x[i] for i in prms[test_num+val_num:]])
    train_y = torch.stack([data_y[i] for i in prms[test_num+val_num:]])
    
    test_x = torch.stack([data_x[i] for i in prms[:test_num]])
    test_y = torch.stack([data_y[i] for i in prms[:test_num]])

    val_x = torch.stack([data_x[i] for i in prms[test_num:test_num+val_num]])
    val_y = torch.stack([data_y[i] for i in prms[test_num:test_num+val_num]])

    new_ds_train = CustomTenDataset(train_x, train_y)
    new_ds_test = CustomTenDataset(test_x, test_y)
    new_ds_val = CustomTenDataset(val_x, val_y)

    return new_ds_train, new_ds_test, new_ds_val


def get_cil_datasets(classes_order=np.arange(4).reshape(2, 2), snrs=[20], save_data=False):
    file_path = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"

    ds_dict = {'train': [], 'test': [], 'val': []}

    for classes_id in classes_order:
        dataset = SignalDataset(snrs=snrs, modulation_ids=classes_id,
                                file_path=file_path)

        train_dataset, test_dataset, val_dataset = ds_random_split(dataset)

        ds_dict['train'].append(train_dataset)
        ds_dict['test'].append(test_dataset)
        ds_dict['val'].append(val_dataset)

    if save_data:
        for i in range(len(ds_dict['train'])):
            np.savez(f"data_{i}.npz", train_x=ds_dict['train'][i].data, train_y=ds_dict['train'][i].data
                     , test_x=ds_dict['test'][i].data, test_y=ds_dict['test'][i].data)

    return ds_dict


if __name__ == "__main__":

    file_path = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
    dataloaders = []
    snrs = [20, 22]

    for modulation_id in range(24):
        dataset = SignalDataset(snrs=snrs, modulation_ids=[modulation_id],
                                file_path=file_path)
        dataloaders.append(DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0))