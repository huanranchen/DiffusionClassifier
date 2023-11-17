from .tf_inception_v3 import IncV3KitModel
from .tf_inception_v4 import IncV4KitModel
from .tf_inc_res_v2 import IncResV2KitModel
from .tf_ens3_adv_inc_v3 import IncV3Ens3AdvKitModel
from .tf_ens4_adv_inc_v3 import IncV3Ens4AdvKitModel
from .tf_ens_adv_inc_res_v2 import IncResV2EnsKitModel
from .tf_resnet_v2_152 import Res152KitModel

"""
please download checkpoints from 
https://github.com/ylhz/tf_to_pytorch_model
"""


def tf_inc_v3(path='./resources/checkpoints/tf_models/tf2torch_inception_v3.npy'):
    model = IncV3KitModel(path)
    return model


def tf_inc_v4(path='./resources/checkpoints/tf_models/tf2torch_inception_v4.npy'):
    model = IncV4KitModel(path)
    return model


def tf_incres_v2(path='./resources/checkpoints/tf_models/tf2torch_inc_res_v2.npy'):
    model = IncResV2KitModel(path)
    return model


def tf_inc_v3_ens3(path='./resources/checkpoints/tf_models/tf2torch_ens3_adv_inc_v3.npy'):
    model = IncV3Ens3AdvKitModel(path)
    return model


def tf_inc_v3_ens4(path='./resources/checkpoints/tf_models/tf2torch_ens4_adv_inc_v3.npy'):
    model = IncV3Ens4AdvKitModel(path)
    return model


def tf_incres_v2_ens(path='./resources/checkpoints/tf_models/tf2torch_ens_adv_inc_res_v2.npy'):
    model = IncResV2EnsKitModel(path)
    return model


def tf_res152(path='./resources/checkpoints/tf_models/tf2torch_resnet_v2_152.npy'):
    model = Res152KitModel(path)
    return model
