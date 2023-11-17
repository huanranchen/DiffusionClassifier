import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Callable, Tuple
from tqdm import tqdm
from attacks import AdversarialInputAttacker
from .utils import cosine_similarity, list_mean
from torch import multiprocessing
from data import save_to_someset


def test_transfer_attack_acc(
    attacker: Callable,
    loader: DataLoader,
    target_models: List[nn.Module],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[float]:
    transfer_accs = [0] * len(target_models)
    denominator = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        x = attacker(x, y)
        with torch.no_grad():
            denominator += x.shape[0]
            for i, model in enumerate(target_models):
                pre = model(x)  # N, D
                if pre.shape != y.shape:
                    _, pre = torch.max(pre, dim=1)
                transfer_accs[i] += torch.sum(pre == y).item()

    transfer_accs = [1 - i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print("-" * 100)
        if hasattr(model, "model"):
            print(model.__class__, model.model.__class__, transfer_accs[i])
        else:
            print(model.__class__, transfer_accs[i])
        print("-" * 100)
    return transfer_accs


def test_robustness(
    attacker: Callable,
    loader: DataLoader,
    target_models: List[nn.Module],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    adversaries_save_folder: str = None,
) -> List[float]:
    robustness = [0] * len(target_models)
    denominator = 0
    adversaries = []
    gts = []
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        x = attacker(x, y)
        adversaries += list(x.clone().split(1, dim=0))
        gts += [i.item() for i in list(y.split(1, dim=0))]
        with torch.no_grad():
            denominator += x.shape[0]
            for i, model in enumerate(target_models):
                pre = model(x)  # N, D
                if pre.shape != y.shape:
                    _, pre = torch.max(pre, dim=1)
                robustness[i] += torch.sum(pre == y).item()

    robustness = [i / denominator for i in robustness]
    # print
    for i, model in enumerate(target_models):
        print("-" * 100)
        if hasattr(model, "model"):
            print(model.__class__, model.model.__class__, robustness[i])
        else:
            print(model.__class__, robustness[i])
        print("-" * 100)

    # save adversaries
    if adversaries_save_folder is not None:
        save_to_someset(adversaries, gts, adversaries_save_folder)
    return robustness


def test_transfer_attack_acc_and_cosine_similarity(
    attacker: AdversarialInputAttacker,
    loader: DataLoader,
    target_models: List[nn.Module],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[List[float], float, float, float]:
    criterion = nn.CrossEntropyLoss()
    train_models: List[nn.Module] = attacker.models
    transfer_accs = [0] * len(target_models)
    denominator = 0
    train_train_cosine_similarities, train_test_cosine_similarities, test_test_cosine_similarities = [], [], []
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        x = attacker(x, y)
        with torch.no_grad():
            denominator += x.shape[0]
            for i, model in enumerate(target_models):
                pre = model(x)  # N, D
                pre = torch.max(pre, dim=1)[1]  # N
                transfer_accs[i] += torch.sum(pre == y).item()
        # calculate cosine simiarity
        train_grads, test_grads = [], []
        x.requires_grad = True
        for m in train_models:
            loss: torch.tensor = criterion(m(x.to(m.device)), y.to(m.device))
            loss.backward()
            train_grads.append(x.grad)
            x.grad = None
        for m in target_models:
            loss: torch.tensor = criterion(m(x), y)
            loss.backward()
            test_grads.append(x.grad)
            x.grad = None
        x.requires_grad = False
        train_grads, test_grads = torch.stack(train_grads), torch.stack(test_grads)
        train_train_cosine_similarities.append(cosine_similarity(train_grads, train_grads))
        train_test_cosine_similarities.append(cosine_similarity(train_grads, test_grads))
        test_test_cosine_similarities.append(cosine_similarity(test_grads, test_grads))

    transfer_accs = [1 - i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print("-" * 100)
        print(model.__class__, transfer_accs[i])
        print("-" * 100)
    train_train_cosine_similarities = list_mean(train_train_cosine_similarities)
    train_test_cosine_similarities = list_mean(train_test_cosine_similarities)
    test_test_cosine_similarities = list_mean(test_test_cosine_similarities)
    print("-" * 100)
    print("train_train_cosine_similarities", train_train_cosine_similarities)
    print("train_test_cosine_similarities", train_test_cosine_similarities)
    print("test_test_cosine_similarities", test_test_cosine_similarities)
    print("-" * 100)
    return transfer_accs, train_train_cosine_similarities, train_test_cosine_similarities, test_test_cosine_similarities


def test_transfer_attack_acc_with_batch(
    get_attacker: Callable,
    batch_x: torch.tensor,
    batch_y: torch.tensor,
    get_target_models: Callable,
    batch_size: int = 1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[float]:
    attacker = get_attacker()
    target_models = get_target_models()
    transfer_accs = [0] * len(target_models)
    denominator = 0
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    xs = list(torch.split(batch_x, batch_size, dim=0))
    ys = list(torch.split(batch_y, batch_size, dim=0))
    attacker.to(device)

    for model in target_models:
        model.to(device)
    for x, y in tqdm(zip(xs, ys)):
        x = attacker(x, y)
        with torch.no_grad():
            denominator += x.shape[0]
            for i, model in enumerate(target_models):
                pre = model(x)  # N, D
                pre = torch.max(pre, dim=1)[1]  # N
                transfer_accs[i] += torch.sum(pre == y).item()

    transfer_accs = [1 - i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print("-" * 100)
        print(model.__class__, transfer_accs[i])
        print("-" * 100)
    return transfer_accs


def test_transfer_attack_acc_distributed(
    get_attacker: Callable,
    loader: DataLoader,
    get_target_models: Callable,
    batch_size: int = 1,
    num_gpu: int = torch.cuda.device_count(),
):
    def list_mean(x: list) -> float:
        return sum(x) / len(x)

    print(f"available gpu num {num_gpu}")
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    xs, ys = torch.cat(xs, dim=0), torch.cat(ys, dim=0)
    xs, ys = list(torch.split(xs, xs.shape[0] // num_gpu, dim=0)), list(torch.split(ys, ys.shape[0] // num_gpu, dim=0))
    pool = multiprocessing.Pool(processes=num_gpu)
    results = [
        pool.apply_async(
            func=test_transfer_attack_acc_with_batch,
            args=(get_attacker, xs[i], ys[i], get_target_models),
            kwds=({"batch_size": batch_size, "device": torch.device(f"cuda:{num_gpu - i - 1}")}),
        )
        for i in range(num_gpu)
    ]
    pool.close()
    pool.join()
    # print(results)
    # results = [list_mean([results[target_model_id][j] for j in range(len(results))])
    #            for target_model_id in range(len(results[0]))]
    # for i, model in enumerate(target_models):
    #     print('-' * 100)
    # print(model.__class__, model.model.__class__, results[i])
    # print('-' * 100)
    return results
