import numpy as np
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

        for modulation_id in modulation_ids:
            for snr in snrs:
                for signal_index in range(modulation_id * 106496 + int((snr + 20) / 2) * 4096,
                                          modulation_id * 106496 + (int((snr + 20) / 2) + 1) * 4096):
                    if np.argmax(f['Y'][signal_index]) == modulation_id and f['Z'][signal_index] == snr:
                        self.list_signals.append({'sig': f['X'][signal_index],
                                                  'modulation': np.argmax(f['Y'][signal_index]),
                                                  'snr': f['Z'][signal_index][0]})

    def __len__(self):
        return len(self.list_signals)

    def __getitem__(self, idx):
        return self.list_signals[idx]


if __name__ == "__main__":

    file_path = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
    dataloaders = []
    snrs = [20, 22]

    for modulation_id in range(24):
        dataset = SignalDataset(snrs=snrs, modulation_ids=[modulation_id],
                                file_path=file_path)
        dataloaders.append(DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0))