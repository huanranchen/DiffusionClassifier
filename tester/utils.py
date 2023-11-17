import torch
import os
from typing import List


def cosine_similarity(x: List):
    '''
    input a list of tensor with same shape. return the mean cosine_similarity
    '''
    x = torch.stack(x)
    N = x.shape[0]
    x = x.reshape(N, -1)

    norm = torch.norm(x, p=2, dim=1)
    x /= norm.reshape(-1, 1)  # N, D
    similarity = x @ x.T  # N, N
    # only the tri-upper part. Note that the diagonal means which one is diagonal.
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).to(torch.bool)
    similarity = similarity[mask]
    return torch.mean(similarity).item()


def list_mean(x: list) -> float:
    return sum(x) / len(x)


def update_log(x: str or List[str], path: str = './log.txt'):
    with open(path, 'a') as file:
        if type(x) is str:
            file.write(x if x.endswith('\n') else x + '\n')
        elif type(x) is list:
            for now in x:
                file.write(now if now.endswith('\n') else now + '\n')
