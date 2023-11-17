from robustbench import load_model
from defenses.Transformations.TTE import TTEAugWrapper
from data import get_CIFAR10_test
from tester import test_apgd_dlr_acc, test_transfer_attack_acc
from attacks import StAdvAttack

loader = get_CIFAR10_test(batch_size=1)
loader = [item for i, item in enumerate(loader) if i < 256]
models = [load_model(model_name="Wang2023Better_WRN-70-16", dataset="cifar10", threat_model="Linf"),
          load_model(model_name="Wang2023Better_WRN-70-16", dataset="cifar10", threat_model="L2"),
          ]
for model in models:
    wraped_model = TTEAugWrapper(model, flip=True, flip_crop=True, n_crops=1)
    test_apgd_dlr_acc(wraped_model, loader=loader, bs=1, eps=8 / 255, norm='Linf')
    test_apgd_dlr_acc(wraped_model, loader=loader, bs=1, eps=0.5, norm='L2')

    # attacker = StAdvAttack([wraped_model], num_iterations=100, bound=0.05)
    # test_transfer_attack_acc(attacker, loader, [wraped_model])
