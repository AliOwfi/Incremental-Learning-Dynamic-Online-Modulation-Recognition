import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_ewc_loss(model):
    loss = 0
    for n, p in model.named_parameters():
        if 'heads' in n:
            continue

        loss += (getattr(model, f"fisher_{n.replace('.', '_')}") * \
                 (p - getattr(model, f"mean_{n.replace('.', '_')}")).pow(2)).sum()

    return loss / 2.


def register_ewc_params(model, dl, return_fishers=False, mh=False, task_id=-1):
    norm_fact = len(dl)
    model.eval()

    for batch in dl:
        batch['sig'], batch['modulation'] = batch['sig'].to(device), batch['modulation'].to(device)

        if mh == False:
            preds = model(batch['sig'])
        else:
            preds = model(batch['sig'])[task_id]

        loss = F.nll_loss(F.log_softmax(preds, dim=1), batch['modulation'])

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
            tmp_fisher.append(p.grad.detach() ** 2 / norm_fact)

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

