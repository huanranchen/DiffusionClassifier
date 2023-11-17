from .models.ncsnpp import NCSNpp
import yaml
import argparse

"""
export PATH=/usr/local/cuda/bin:$PATH for solving bug like:
https://github.com/Dao-AILab/flash-attention/issues/157
"""

__all__ = ['get_NCSNPP']

import ml_collections
import torch
from .models import layerspp
from .models.ncsnpp_multihead import NCSNppMultiHead


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_iters = 1300001
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 1024
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'CIFAR10'
    data.image_size = 32
    data.random_flip = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_diffpure_model_cifar_config():
    # config = {
    #     "sigma_min": 0.01,
    #     "sigma_max": 50,
    #     "num_scales": 1000,
    #     "beta_min": 0.1,
    #     "beta_max": 20.,
    #     "dropout": 0.1,
    #     "name": 'ncsnpp',
    #     "scale_by_sigma": False,
    #     "ema_rate": 0.9999,
    #     "normalization": 'GroupNorm',
    #     "nonlinearity": 'swish',
    #     "nf": 128,
    #     "ch_mult": [1, 2, 2, 2],  # (1, 2, 2, 2)
    #     "num_res_blocks": 8,
    #     "attn_resolutions": [16],  # (16,)
    #     "resamp_with_conv": True,
    #     "conditional": True,
    #     "fir": False,
    #     "fir_kernel": [1, 3, 3, 1],
    #     "skip_rescale": True,
    #     "resblock_type": 'biggan',
    #     "progressive": 'none',
    #     "progressive_input": 'none',
    #     "progressive_combine": 'sum',
    #     "attention_type": 'ddpm',
    #     "init_scale": 0.,
    #     "embedding_type": 'positional',
    #     "fourier_scale": 16,
    #     "conv_size": 3,
    # }
    with open('./models/unets/score_sde/cifar10.yml') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return config


# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with DDPM."""


def get_ddpmpp_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vpsde'
    training.continuous = False
    training.reduce_mean = True

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'none'

    # data
    data = config.data
    data.centered = True

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.0
    model.embedding_type = 'positional'
    model.conv_size = 3

    return config


def get_ncsnpp_deep_continuous_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vpsde'
    training.continuous = True
    training.n_iters = 950001
    training.reduce_mean = True

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # data
    data = config.data
    data.centered = True

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.fourier_scale = 16
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 8
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.embedding_type = 'positional'
    model.init_scale = 0.0
    model.conv_size = 3
    return config


def get_NCSNPP(grad_checkpoint=False, **kwargs):
    layerspp.gradient_checkpoint = grad_checkpoint
    config = get_diffpure_model_cifar_config()
    model = NCSNpp(config, **kwargs)
    return model


def get_NCSNPP_cached(grad_checkpoint=False, **kwargs):
    layerspp.gradient_checkpoint = grad_checkpoint
    config = get_diffpure_model_cifar_config()
    model = NCSNppMultiHead(config, **kwargs)
    return model
