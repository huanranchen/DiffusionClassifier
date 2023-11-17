import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Iterable, List
from tqdm import tqdm


@torch.no_grad()
def test_acc(
    model: nn.Module,
    loader: DataLoader or Iterable,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=False,
):
    total_loss, total_acc, denominator = 0, 0, 0
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device).eval()
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        logit = model(x)
        if logit.shape != y.shape:
            total_loss += criterion(logit, y).item() * y.shape[0]
            _, pre = torch.max(logit, dim=1)
        else:
            pre = logit
        if verbose:
            print(logit, "\n", pre, y, "\n" + "-" * 50)
        total_acc += torch.sum((pre == y)).item()
        denominator += y.shape[0]

    test_loss = total_loss / denominator
    test_accuracy = total_acc / denominator
    print(f"loss = {test_loss}, acc = {test_accuracy}")
    return test_loss, test_accuracy


@torch.no_grad()
def test_multimodel_acc(loader: DataLoader, target_models: List[nn.Module]) -> List[float]:
    transfer_accs = [0] * len(target_models)
    denominator = 0
    for x, y in loader:
        denominator += x.shape[0]
        for i, model in enumerate(target_models):
            pre = model(x)  # N, D
            pre = torch.max(pre, dim=1)[1]  # N
            transfer_accs[i] += torch.sum(pre == y).item()

    transfer_accs = [i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print("-" * 100)
        print(model.__class__, 1 - transfer_accs[i])
        print("-" * 100)
    return transfer_accs


@torch.no_grad()
def test_multimodel_acc_one_image(x: torch.tensor, y: torch.tensor, target_models: List[nn.Module]) -> List[float]:
    transfer_accs = [0] * len(target_models)
    denominator = 0
    denominator += x.shape[0]
    for i, model in enumerate(target_models):
        pre = model(x)  # N, D
        pre = torch.max(pre, dim=1)[1]  # N
        transfer_accs[i] += torch.sum(pre == y).item()

    transfer_accs = [i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print("-" * 100)
        print(model.__class__, 1 - transfer_accs[i])
        print("-" * 100)
    return transfer_accs


@torch.no_grad()
def test_range_predictor_acc(
    model: nn.Module,
    loader: DataLoader or Iterable,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=False,
):
    total_acc, denominator, total_length = 0, 0, 0
    model.to(device).eval().requires_grad_(False)
    for x, y in tqdm(loader):
        assert x.shape[0] == y.shape[0] == 1, "batch size should be 1"
        x, y = x.to(device), y.cpu().item()
        candidate = model(x)
        if verbose:
            print(candidate, y)
        total_acc += (y in candidate)
        denominator += 1
        total_length += len(candidate)

    test_accuracy, total_length = total_acc / denominator, total_length / denominator
    print(f"range prediction acc = {test_accuracy}, average length = {total_length}")
    return test_accuracy
