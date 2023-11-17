# Diffusion Classifier

---

## Install
### Environment
Please create a new conda environment and run:
```bash
pip install requirements.txt
```


### Data
**CIFAR10** will be downloaded automatically    
For **PACS** dataset, please refer to ./data/PACS.py for install    
For **NIPS17** dataset, you can run \
```bash
kaggle datasets download -d google-brain/nips-2017-adversarial-learning-development-set
```
and then put it into ./resources/NIPS17

### Model Checkpoints
All the models in our paper are from torchvision, robustbench, timm library, and the checkpoints will be downloaded automatically.

We also encapsulate some models and defenses in *"./models"* and *"./defenses"*. If you want to attack them, you can download their checkpoints by yourself

---

## Usage

### Code Framework

> attacks: Some attack algorithms. Including VMI, VMI-CW, CW, SAM, etc.      
> data: loader of CIFAR, NIPS17, PACS    
> defenses: Some defenses algorithm    
> experiments: Example codes    
> models: Some pretrained models   
> optimizer: scheduler and optimizer   
> tester: some functions to test accuracy and attack success rate   
> utils: Utilities. Like draw landscape, get time, HRNet, etc.     


### Basic functions

```
tester.test_transfer_attack_acc(attacker:AdversarialInputBase, loader:DataLoader, target_models: List[nn.Module]) \
```

This function aims to get the attack success rate on loader against target models



```
attacker = xxxAttacker(train_models: List[nn.Module])
```
You can initialize attacker like this.

### Examples
Here is an example of testing attack success rate on NIPS17 loader.
```python
from models import resnet18, Wong2020Fast, Engstrom2019Robustness, BaseNormModel, Identity
from attacks import MI_CommonWeakness
attacker = MI_CommonWeakness([
    BaseNormModel(resnet18(pretrained=True)), # model that requires normalization
    Identity(Wong2020Fast(pretrained=True)) # model that do not need normalization
])

from tester import test_transfer_attack_acc
from data import get_NIPS17_loader
test_transfer_attack_acc(attacker, 
                         get_NIPS17_loader(), 
                         [
                             Identity(Wong2020Fast(pretrained=True)), # white box attack
                             Identity(Engstrom2019Robustness(pretrained=True)), # transfer attack
                          ]
                         )
```


For more example codes, please visit *'./experiments'* folder. There are some example codes using our framework to attack and draw landscapes. I believe you can quickly get familiar with our framework via these example codes.


### HRNet
HRNet is a function that aims to reduce memory cost when crafting adversarial examples.

**We haven't implemented the convolution of HRNet. Up to now, HRNet can only help to reduce about 30% of memory cost**

#### Usage
```python
from models import resnet18
from utils import change

model = resnet18()
model = change(model)
```


---

## Citation
Please cite us:
```
@article{chen2023rethinking,
  title={Rethinking Model Ensemble in Transfer-based Adversarial Attacks},
  author={Chen, Huanran and Zhang, Yichi and Dong, Yinpeng and Zhu, Jun},
  journal={arXiv preprint arXiv:2303.09105},
  year={2023}
}
```

If you have any question, you can contact us by:   
Email: huanran_chen@outlook.com, huanranchen@bit.edu.cn    
