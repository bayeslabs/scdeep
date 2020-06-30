import numpy as np
import pandas as pd
from typing import List, Union, DefaultDict, Dict
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
                   'exp': ExponentialActivation(), 'softplus': nn.Softplus()}
    initializers = {'xavier': nn.init.xavier_uniform_, 'zeros': nn.init.zeros_, 'normal': nn.init.normal_}

    def __init__(
            self,
            input_dim,
            out_dim,
            batchnorm=False,
            activation='relu',
            dropout=0.,
            weight_init=None,
            weight_init_params: Dict = None,
            bias_init=None,
            bias_init_params: Dict = None
    ):
        super(LinearActivation, self).__init__()
        if weight_init_params is None:
            weight_init_params = {}
        if bias_init_params is None:
            bias_init_params = {}
        self.batchnorm_layer = None
        self.act_layer = None
        self.dropout_layer = None
        self.linear = nn.Linear(input_dim, out_dim)

        if weight_init is not None:
            self.initializers[weight_init](self.linear.weight, **weight_init_params)
        if bias_init is not None:
            self.initializers[bias_init](self.linear.bias, **bias_init_params)

        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(num_features=out_dim)
        if activation:
            self.act_layer = self.activations[activation]
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        if self.batchnorm_layer:
            x = self.batchnorm_layer(x)
        if self.act_layer:
            x = self.act_layer(x)
        if self.dropout_layer:
            x = self.dropout_layer(x)
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
            dropout: float = 0.,
            weight_initializer=None,
            weight_init_params: Dict = None,
            bias_initializer=None,
            bias_init_params: Dict = None
    ):
        super(AutoEncoder, self).__init__()
        if weight_init_params is None:
            weight_init_params = {}
        if bias_init_params is None:
            bias_init_params = {}
        self.input_dim = input_d.shape[1]
        self.batchnorm = batchnorm

        encode_layers = []
        if len(encoder_layers_dim) > 0:
            for i in range(len(encoder_layers_dim)):
                if i == 0:
                    encode_layers.append(LinearActivation(self.input_dim, encoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
                else:
                    encode_layers.append(LinearActivation(encoder_layers_dim[i - 1], encoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
            self.latent_layer_input_dim = encoder_layers_dim[-1]
        else:
            self.latent_layer_input_dim = self.input_dim
        self.encode = nn.Sequential(*encode_layers)

        self.latent_layer = LinearActivation(self.latent_layer_input_dim, latent_layer_out_dim,
                                             batchnorm=self.batchnorm, activation=activation,
                                             dropout=dropout,
                                             weight_init=weight_initializer, weight_init_params=weight_init_params,
                                             bias_init=bias_initializer, bias_init_params=bias_init_params)

        decode_layers = []
        if len(decoder_layers_dim) > 0:
            for i in range(len(decoder_layers_dim)):
                if i == 0:
                    decode_layers.append(LinearActivation(latent_layer_out_dim, decoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
                else:
                    decode_layers.append(LinearActivation(decoder_layers_dim[i - 1], decoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
            self.output_layer_input_dim = decoder_layers_dim[-1]
        else:
            self.output_layer_input_dim = latent_layer_out_dim
        self.decode = nn.Sequential(*decode_layers)

        self.output_layer = LinearActivation(self.output_layer_input_dim, self.input_dim,
                                             activation=activation,
                                             weight_init=weight_initializer, weight_init_params=weight_init_params,
                                             bias_init=bias_initializer, bias_init_params=bias_init_params)

    def forward(self, x):
        encoded = self.encode(x)
        latent = self.latent_layer(encoded)
        decoded = self.decode(latent)
        output = self.output_layer(decoded)

        return latent, output


class NBAutoEncoder(AutoEncoder):
    def __init__(
            self,
            input_d,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            batchnorm: bool = True,
            activation: str = 'relu',
            weight_initializer='xavier',
            **kwargs
    ):
        self.batchnorm = batchnorm
        super(NBAutoEncoder, self).__init__(input_d, encoder_layers_dim=encoder_layers_dim,
                                            decoder_layers_dim=decoder_layers_dim, latent_layer_out_dim=latent_layer_out_dim,
                                            batchnorm=batchnorm, activation=activation,
                                            weight_initializer=weight_initializer, **kwargs)

        # self.latent_layer = LinearActivation(self.latent_layer_input_dim, latent_layer_out_dim,
        #                                      batchnorm=self.batchnorm, activation=None, weight_init=weight_initializer)
        # the self.output_layer will be equivalent to the mean layer
        self.output_layer = LinearActivation(self.output_layer_input_dim, self.input_dim,
                                             activation='exp', weight_init=weight_initializer)

        self.theta_layer = LinearActivation(self.output_layer_input_dim, self.input_dim,
                                            activation='softplus', weight_init=weight_initializer)

    def forward(self, x, size_factors):
        encoded = self.encode(x)
        latent_output = self.latent_layer(encoded)
        decoded = self.decode(latent_output)
        mean = self.output_layer(decoded)
        mean = mean * size_factors.reshape((-1, 1))
        theta = self.theta_layer(decoded)
        return latent_output, [mean, theta]


class ZINBAutoEncoder(AutoEncoder):
    def __init__(
            self,
            input_d,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            batchnorm: bool = True,
            activation: str = 'relu',
            weight_initializer='xavier',
            **kwargs
    ):
        self.batchnorm = batchnorm
        super(ZINBAutoEncoder, self).__init__(input_d, encoder_layers_dim=encoder_layers_dim,
                                              decoder_layers_dim=decoder_layers_dim, latent_layer_out_dim=latent_layer_out_dim,
                                              batchnorm=batchnorm, activation=activation,
                                              weight_initializer=weight_initializer, **kwargs)

        # self.latent_layer = LinearActivation(self.latent_layer_input_dim, latent_layer_out_dim,
        #                                      batchnorm=self.batchnorm, activation=None, weight_init=weight_initializer)
        # the self.output_layer will be equivalent to the mean layer
        self.output_layer = LinearActivation(self.output_layer_input_dim, self.input_dim,
                                             activation='exp', weight_init=weight_initializer)

        self.theta_layer = LinearActivation(self.output_layer_input_dim, self.input_dim,
                                            activation='softplus', weight_init=weight_initializer)

        self.pi_layer = LinearActivation(self.output_layer_input_dim, self.input_dim,
                                         activation='sigmoid', weight_init=weight_initializer)

    def forward(self, x, size_factors):
        encoded = self.encode(x)
        latent_output = self.latent_layer(encoded)
        decoded = self.decode(latent_output)
        mean = self.output_layer(decoded)
        mean = mean * size_factors.reshape((-1, 1))
        theta = self.theta_layer(decoded)
        pi = self.pi_layer(decoded)
        return latent_output, [mean, theta, pi]
