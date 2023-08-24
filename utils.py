import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_dl(model, dl, verbose=True):
    model.eval()
    n_correct = 0
    n_total = 0
    for batch in dl:
        batch['sig'], batch['modulation'] = batch['sig'].to(device), batch['modulation'].to(device)

        pred = model(batch['sig'])
        pred = torch.argmax(pred, dim=1)
        n_correct += torch.sum(pred == batch['modulation']).item()
        n_total += batch['modulation'].shape[0]

    if verbose:
        print(f'Accuracy: {n_correct / n_total * 100}')

    return n_correct / n_total * 100
