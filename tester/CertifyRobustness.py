import torch
from torch.utils.data import DataLoader
from typing import Iterable, Dict, Tuple, List
from tqdm import tqdm
from defenses.RandomizedSmoothing import Smooth
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

__WRONG_PREDICTION__ = -1


def robust_radius_given_correct_num(nA: int = 990, n: int = 1000, alpha: float = 0.001, sigma: float = 0.25) -> float:
    pABar = proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0]
    radius = sigma * norm.ppf(pABar)
    return radius


def radii_discretion(radii: List[float], epsilons: Iterable = (0, 0.25, 0.5, 0.75, 1)):
    radii_tensor = torch.tensor(radii)
    denominator = len(radii)
    result = dict()
    for eps in epsilons:
        print("-" * 100)
        result[eps] = torch.sum(radii_tensor >= eps).item() / denominator
        print(f"certified robustness at {eps} is {result[eps]}")
        print("-" * 100)
    return result


def nA_and_n_to_radii(nAs: List, ns: List, *args, **kwargs):
    radii = []
    for nA, n in zip(nAs, ns):
        radii.append(robust_radius_given_correct_num(nA, n, *args, **kwargs))
    return radii


@torch.no_grad()
def certify_robustness(
    model: Smooth,
    loader: DataLoader or Iterable,
    epsilons: Iterable = (0, 0.25, 0.5, 0.75, 1),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    *args,
    **kwargs,
) -> Tuple[Dict, List, List, List]:
    model.base_classifier.to(device).eval()
    radii, nAs, ns = [], [], []
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        label, radius, nA, n = model.certify(x.squeeze(), *args, **kwargs)
        radii.append(radius if label == y.item() else __WRONG_PREDICTION__)
        nAs.append(nA if label == y.item() else 0)
        ns.append(n)
    result = radii_discretion(radii, epsilons)
    return result, nAs, ns, radii
