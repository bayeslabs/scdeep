import numpy as np
import pandas as pd
from typing import List, Union
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse
from torch import distributions

from trainer import Trainer
from dataset import GeneExpressionDataset, normalize
from network import AutoEncoder, LinearActivation
from utils import nan2inf



class DCA(AutoEncoder):
    def __init__(
            self,
            input_d,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            batchnorm: bool = True,
            activation: str = 'relu',
            weight_init='xavier',
            loss_type: str = 'zinb_loss'
    ):
        self.batchnorm = batchnorm
        self.loss_type = loss_type
        self.pi_layer = None
        super(DCA, self).__init__(input_d, encoder_layers_dim, decoder_layers_dim, latent_layer_out_dim, batchnorm,
                                  activation, weight_init)
        # the self.output_layer will be equivalent to the mean layer
        if len(decoder_layers_dim) > 0:
            output_layer_input_dim = decoder_layers_dim[-1]
        else:
            output_layer_input_dim = latent_layer_out_dim
        self.output_layer = LinearActivation(output_layer_input_dim, self.input_dim,
                                             activation='exp', weight_init=weight_init)
        self.theta_layer = LinearActivation(output_layer_input_dim, self.input_dim,
                                            activation='exp', weight_init=weight_init)

        if self.loss_type == 'zinb_loss':
            self.pi_layer = LinearActivation(output_layer_input_dim, self.input_dim,
                                             activation='sigmoid', weight_init=weight_init)

    def forward(self, x, size_factors):
        encoded = self.encode(x)
        latent_output = self.latent_layer(encoded)
        decoded = self.decode(latent_output)
        mean = self.output_layer(decoded)
        mean = mean * size_factors.reshape((-1, 1))
        theta = self.theta_layer(decoded)
        if self.pi_layer:
            pi = self.pi_layer(decoded)
            return latent_output, [mean, theta, pi]
        return latent_output, [mean, theta]


class DCATrainer(Trainer):
    def __init__(
            self,
            model: DCA,
            gene_dataset: GeneExpressionDataset,
            use_mask: bool = True,
            scale_factor=1.0,
            ridge_lambda=0.0,
            l1_coeff=0.,
            l2_coeff=0.,
            **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)

        self.model = self.model
        self.use_mask = use_mask
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff

        (train_set, test_set, val_set) = self.train_test_validation()
        self.register_posterior(train_set, test_set, val_set)

        self.loss_type = self.model.loss_type
        if self.loss_type == 'zinb_loss':
            self.ridge_lambda = ridge_lambda
            self.loss = self.zinb_loss
        else:
            self.loss = self.nb_loss
        self.latent_outputs = []

    def nb_loss(self, y_true, output, mean=True):
        y_pred, theta = output

        y_true = y_true.type(torch.FloatTensor)
        y_pred = y_pred.type(torch.FloatTensor) * self.scale_factor

        theta = torch.min(theta, torch.zeros_like(theta)+1e6)
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                    y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))

        loss = t1 + t2

        loss = nan2inf(loss)

        if mean:
            loss = torch.mean(loss)

        return loss

    def zinb_loss(self, y_true, output, mean=True):
        y_pred, theta, pi = output

        nb_case = self.nb_loss(y_true, [y_pred, theta], mean=False) - torch.log(1.0 - pi + self.eps)

        y_true = y_true.type(torch.FloatTensor)
        y_pred = y_pred.type(torch.FloatTensor) * self.scale_factor
        theta = torch.min(theta, torch.zeros_like(theta)+1e6)

        zero_nb = torch.pow(theta / (theta + y_pred + self.eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + self.eps)
        result = torch.where((y_true < 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(pi)
        result += ridge

        result = nan2inf(result)

        if mean:
            result = torch.mean(result)
        return result

    def on_training_begin(self):
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

    def model_output(self, data_tensor):
        data_tensors, indices = data_tensor

        size_factors_layer = torch.tensor(self.gene_dataset.size_factor[indices, :])
        size_factors_layer = size_factors_layer.cuda() if self.use_cuda else size_factors_layer

        latent_output, output = self.model(data_tensors, size_factors_layer)
        self.latent_outputs.append(latent_output)
        return output, data_tensors

    def on_training_loop(self, data):
        output, data_tensors = self.model_output(data)
        self.optimizer.zero_grad()

        self.current_loss = loss = self.loss(data_tensors, output)
        l1_reg = torch.tensor(0.)
        l2_reg = torch.tensor(0.)
        for param in self.model.parameters():
            l1_reg += torch.norm(param, p=1)
            l2_reg += (torch.norm(param, p=2) ** 2)
        self.current_loss = loss = loss + self.l1_coeff*l1_reg + self.l2_coeff*l2_reg

        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5)
        self.optimizer.step()

    def predict(self, input_d, size_factor, subset=1):
        latent_output, output = self.model(torch.tensor(input_d), torch.tensor(size_factor))
        latent_output = latent_output.detach().numpy()
        output = [l.detach().numpy() for l in output]
        return latent_output, output





