import torch
from models.unets import get_edm_cifar_uncond, get_edm_cifar_cond
from data import get_CIFAR10_test
from tester import test_acc, test_apgd_dlr_acc
from tester.utils import cosine_similarity
from defenses.PurificationDefenses.DiffPure.DiffPure.DiffusionLikelihoodMaximizer import (
    diffusion_likelihood_maximizer_defense,
)
from defenses.PurificationDefenses.DiffPure import EDMEulerIntegralDC
from defenses.PurificationDefenses.DiffPure.LikelihoodMaximization import EDMEulerIntegralLM
from models import WideResNet_70_16_dropout
from tqdm import tqdm


loader = get_CIFAR10_test(batch_size=1)
x, y = next(iter(loader))
x, y = x.cuda(), y.cuda()
uncond_edm_net = get_edm_cifar_uncond()
uncond_edm_net.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
uncond_edm_dc = EDMEulerIntegralDC(unet=uncond_edm_net)
lm = EDMEulerIntegralLM(uncond_edm_dc)

# cond_edm_net = get_edm_cifar_cond()
# cond_edm_net.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_cond.pt"))
# dc = EDMEulerIntegralDC(unet=cond_edm_net)

defensed = diffusion_likelihood_maximizer_defense(WideResNet_70_16_dropout(), lm.optimize_back)

num_grad = 10

# Our diffusion classifier
criterion = torch.nn.CrossEntropyLoss()
grad = []
for i in range(10):
    grad.clear()
    for _ in tqdm(range(num_grad)):
        x.requires_grad_(True)
        loss = defensed(x).squeeze()[i]
        loss.backward()
        grad.append(x.grad.clone())
        x.requires_grad_(False)
        x.grad = None

    print("class ", i, " cosine similarity: ", cosine_similarity(grad))
