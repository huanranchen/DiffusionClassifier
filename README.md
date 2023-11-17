# Diffusion Classifier

---

## Install

Please refer to:    [**BasicReadMe**](https://github.com/huanranchen/AdversarialAttack/blob/main/README.md)

### Model Checkpoints


There are two version of our implementation. One is based on VP-SDE proposed by Song et al. (2021). This version is not recommended, and has been depreciated. For this, you need to download:

**CIFAR10 unconditional diffusion model for DiffPure**:      
https://drive.google.com/file/d/1zfblaZd64Aye-FMpus85U5PLrpWKusO4/view            
Put it into ./resources/checkpoints/DiffPure/32x32_diffusion.pth

**CIFAR10 WideResNet-70-16-dropout~(discriminative classifier used in DiffPure)**:        
https://github.com/NVlabs/DiffPure, "Data and pre-trained models" part.       
Put it into ./resources/checkpoints/models/WideResNet_70_16_dropout.pt

**Conditional diffusion model for diffusion classifier**           
We will share our checkpoints soon. Now you can train it by yourself.      

**ImageNet unconditional diffusion model for DiffPure**:      
https://drive.google.com/file/d/1zfblaZd64Aye-FMpus85U5PLrpWKusO4/view               
Put it into ./resources/checkpoints/DiffPure/256x256_diffusion_uncond.pt



Another is based on Karras et al. (2022), a.k.a. EDM. We strongly recommend you to use this implementation. For this, you need to download:

EDM checkpoints: https://drive.google.com/drive/folders/1mQoH6WbnfItphYKehWVniZmq9iixRn8L?usp=sharing

Put it into ./resources/checkpoints/models/EDM/


**CIFAR10 WideResNet-70-16-dropout~(discriminative classifier used in DiffPure)**:        
https://github.com/NVlabs/DiffPure, "Data and pre-trained models" part.       
Put it into ./resources/checkpoints/models/WideResNet_70_16_dropout.pt



---


## Experiments

All experiments codes are in *'./experiments/'*.

**DiffAttack.py**:  Attack DiffPure.         

**DiffusionClassifierN+T**: Multihead diffusion with likelihood maximization.

**DiffusionClassifierTK**: Off-the-shelf diffusion.

**DiffusionClassifierN+TK**: Off-the-shelf diffusion with likelihood maximization.

**DiffusionAsClassifier**: Depreciated. Test robustness of diffusion classifier (VP-SDE) under AutoAttack+BPDA/Lagrange/DirectDifferentiate

**DiffusionMaximizer**: Likelihood maximizer. A new diffusion purification method we proposed. See Sec 3.4 in our paper for detail. Could be combined with discriminative classifier.

**DirectAttack**: Direct differentiate through likelihood maximization.

**ObfuscatedGradient**: Measure the cosine similarity between the gradient of diffusion classifier and DiffPure. See Sec 4.4 in our paper for detail.

**OptimalDiffusionClassifier**: See Sec 3.3 for detail.

**stadv**: Measure the robustness under STAdv attack.

**cifar100**: Experiment on cifar100





---

## Citation
Please cite us:
```
@article{chen2023robust,
  title={Robust Classification via a Single Diffusion Model},
  author={Chen, Huanran and Dong, Yinpeng and Wang, Zhengyi and Yang, Xiao and Duan, Chengqi and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2305.15241},
  year={2023}
}
```

If you have any question, you can contact us by:  

Github issue.

Email: huanran_chen@outlook.com (Recommended), huanranchen@bit.edu.cn (Not recommended, this email will be banned 1 years later)




