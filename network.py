import numpy as np
import pandas as pd
from typing import List, Union, DefaultDict
import logging
import torch
from torch import nn


class ExponentialActivation(nn.Module):

    def __init__(self):
        super(ExponentialActivation, self).__init__()

    def forward(self, x):
        x = torch.exp(x)
        return x


class LinearActivation(nn.Module):

    activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(),
                'exp': ExponentialActivation()}
    initializers = {'xavier': nn.init.xavier_uniform_, 'zeros': nn.init.zeros_, 'normal': nn.init.normal_}

    def __init__(
            self,
            input_dim,
            out_dim,
            batchnorm=False,
            activation='relu',
            weight_init=None,
            weight_init_params: DefaultDict = {},
            bias_init=None,
            bias_init_params: DefaultDict = {}
    ):
        super(LinearActivation, self).__init__()
        self.batchnorm_layer = None
        self.linear = nn.Linear(input_dim, out_dim)

        if weight_init is not None:
            self.initializers[weight_init](self.linear.weight, **weight_init_params)
        if bias_init is not None:
            self.initializers[bias_init](self.linear.bias, **bias_init_params)

        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(num_features=out_dim)
        self.act_layer = self.activations[activation]

    def forward(self, x):
        x = self.linear(x)
        if self.batchnorm_layer:
            x = self.batchnorm_layer(x)
        x = self.act_layer(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(
            self,
            input_d,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            batchnorm: bool = True,
            activation: str = 'relu',
            weight_initializer=None,
            weight_init_params: DefaultDict = {},
            bias_initializer=None,
            bias_init_params: DefaultDict = {}
    ):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_d.shape[1]
        self.batchnorm = batchnorm

        encode_layers = []
        if len(encoder_layers_dim) > 0:
            for i in range(len(encoder_layers_dim)):
                if i == 0:
                    encode_layers.append(LinearActivation(self.input_dim, encoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
                else:
                    encode_layers.append(LinearActivation(encoder_layers_dim[i - 1], encoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
            latent_layer_input_dim = encoder_layers_dim[-1]
        else:
            latent_layer_input_dim = self.input_dim
        self.encode = nn.Sequential(*encode_layers)

        self.latent_layer = LinearActivation(latent_layer_input_dim, latent_layer_out_dim,
                                             batchnorm=self.batchnorm, activation=activation,
                                             weight_init=weight_initializer, weight_init_params=weight_init_params,
                                             bias_init=bias_initializer, bias_init_params=bias_init_params)

        decode_layers = []
        if len(decoder_layers_dim) > 0:
            for i in range(len(decoder_layers_dim)):
                if i == 0:
                    decode_layers.append(LinearActivation(latent_layer_out_dim, decoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
                else:
                    decode_layers.append(LinearActivation(decoder_layers_dim[i - 1], decoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
            output_layer_input_dim = decoder_layers_dim[-1]
        else:
            output_layer_input_dim = latent_layer_out_dim
        self.decode = nn.Sequential(*decode_layers)

        self.output_layer = LinearActivation(output_layer_input_dim, self.input_dim,
                                             activation=activation,
                                             weight_init=weight_initializer, weight_init_params=weight_init_params,
                                             bias_init=bias_initializer, bias_init_params=bias_init_params)

    def forward(self, x):
        encoded = self.encode(x)
        latent = self.latent_layer(encoded)
        decoded = self.decode(latent)
        output = self.output_layer(decoded)

        return latent, output
