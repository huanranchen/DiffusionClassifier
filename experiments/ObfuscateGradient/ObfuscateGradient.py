import torch
from data import get_CIFAR10_test
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import random
from defenses import RobustDiffusionClassifier, DiffusionClassifier, DiffusionPure
from tester.utils import cosine_similarity

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=1)
device = torch.device('cuda')

diffpure = DiffusionClassifier()
diffpure.unet.load_state_dict(torch.load('./unet_condition_old.pt'))
diffpure.eval().requires_grad_(False).to(device)

x, y = next(iter(loader))
x, y = x.cuda(), y.cuda()
criterion = torch.nn.CrossEntropyLoss()

num_grad = 10

# Our diffusion classifier
grad = []
for i in range(10):
    grad.clear()
    for _ in tqdm(range(num_grad)):
        x.requires_grad_(True)
        diffpure.partial(x, i)
        grad.append(x.grad.clone())
        x.requires_grad_(False)
        x.grad = None

    print('class ', i, ' cosine similarity: ', cosine_similarity(grad))

# robust diffusion classifier
diffpure = RobustDiffusionClassifier(bpda=True, likelihood_maximization=True, diffpure=False)
diffpure.unet.load_state_dict(torch.load('./unet_condition_old.pt'))
diffpure.eval().requires_grad_(False).cuda()
grad = []
for i in range(10):
    grad.clear()
    for _ in tqdm(range(num_grad)):
        x.requires_grad_(True)
        loss = diffpure(x).squeeze()[i]
        loss.backward()
        grad.append(x.grad.clone())
        x.requires_grad_(False)
        x.grad = None

    print('class ', i, ' cosine similarity: ', cosine_similarity(grad))

# diffpure
grad.clear()
del diffpure
criterion = torch.nn.CrossEntropyLoss()
diffpure = DiffusionPure(grad_checkpoint=True)
for _ in tqdm(range(num_grad)):
    x.requires_grad_(True)
    pre = diffpure(x)
    loss = pre[0, 0]
    loss.backward()
    grad.append(x.grad.clone())
    x.requires_grad_(False)
    x.grad = None

print(cosine_similarity(grad))

# resnet
from models import resnet50

grad.clear()
diffpure = resnet50(pretrained=True).cuda().eval().requires_grad_(False)
x = x.view(1, 3, 32, 32)
for _ in tqdm(range(num_grad)):
    x.requires_grad_(True)
    pre = diffpure(x)
    loss = pre[0, 0]
    loss.backward()
    grad.append(x.grad.clone())
    x.requires_grad_(False)
    x.grad = None

print(cosine_similarity(grad))
