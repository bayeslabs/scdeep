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
from dataset import GeneExpressionDataset
from utils import truncated_normal, parameterized_truncated_normal
from network import AutoEncoder, LinearActivation


class scScope(AutoEncoder):
    def __init__(
            self,
            input_d,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            exp_batch_input,
            t: int = 2
    ):
        self.t = t
        self.input_dim = input_d.shape[1]

        super(scScope, self).__init__(input_d, encoder_layers_dim, decoder_layers_dim, latent_layer_out_dim,
                                      activation='relu', weight_initializer='normal', weight_init_params={'std': 0.1},
                                      bias_initializer='zeros')

        if len(exp_batch_input) > 0:
            num_batch = exp_batch_input.shape[1]
        else:
            num_batch = 1

        self.batch_effect_layer = nn.Linear(num_batch, self.input_dim, bias=False)
        nn.init.zeros_(self.batch_effect_layer.weight)

    def forward(self, x, exp_batch_input):
        self.input = x
        latent_features_list = []
        output_list = []
        self.batch_effect_removal_layer = self.batch_effect_layer(exp_batch_input)
        for i in range(self.t):
            if i == 0:
                x = F.relu(x - self.batch_effect_removal_layer)
            else:
                if i == 1:

                    impute_layer1 = LinearActivation(self.input_dim, 64, activation='relu',
                                                     weight_init='normal', weight_init_params={'std':0.1},
                                                     bias_init='zeros')

                    impute_layer2 = nn.Linear(64, self.input_dim)
                    nn.init.zeros_(impute_layer2.bias)
                    nn.init.normal_(impute_layer2.weight, mean=0.0, std=0.1)

                    self.imputation_model = nn.Sequential(impute_layer1, impute_layer2)
                    self.imputation_model = self.imputation_model.float()

                imputed = self.imputation_model(output)
                imputed = torch.mul(1 - torch.sign(self.input), imputed)
                x = F.relu(imputed + self.input - self.batch_effect_removal_layer)
            x = self.encode(x)
            latent_features = self.latent_layer(x)
            x = self.decode(latent_features)
            output = self.output_layer(x)
            output_list.append(output)
            latent_features_list.append(latent_features)

        return output_list, latent_features_list, self.batch_effect_removal_layer


class scScopeTrainer(Trainer):

    def __init__(
            self,
            model: scScope,
            gene_dataset: GeneExpressionDataset,
            train_size: Union[int, float] = 0.8,
            test_size: Union[int, float] = 0.2,
            use_mask: bool = True,
            **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)

        self.model = self.model
        self.train_size = train_size
        self.test_size = test_size
        self.use_mask = use_mask

        (train_set, test_set, val_set) = self.train_test_validation(train_size=train_size, test_size=test_size)
        self.register_posterior(train_set, test_set, val_set)

    def on_training_begin(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def on_training_loop(self, data_tensors):
        data, indices = data_tensors
        batch_data = torch.tensor(self.gene_dataset.batch_indices[indices, :])
        if self.gene_dataset.num_batches > 1:
            one_hot_batches = np.zeros((batch_data.shape[0], self.gene_dataset.num_batches))
            one_hot_batches[:,(batch_data.reshape((1, -1)) - 1)] = 1.0
        else:
            one_hot_batches = np.ones_like(batch_data)
        one_hot_batches = torch.tensor(one_hot_batches)
        one_hot_batches = one_hot_batches.cuda() if self.use_cuda else one_hot_batches

        output_list, latent_list, batch_effect_removal_layer = self.model(data, one_hot_batches.float())
        self.current_loss = loss = self.loss(output_list, data, use_mask=self.use_mask,
                                             batch_effect_removal_layer=batch_effect_removal_layer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def loss(self, output_layer_list, input_d, use_mask, batch_effect_removal_layer):
        input_d_corrected = input_d - batch_effect_removal_layer
        if use_mask:
            val_mask = torch.sign(input_d_corrected)
        else:
            val_mask = torch.sign(input_d_corrected + 1)

        for i in range(len(output_layer_list)):
            out_layer = output_layer_list[i]
            if i == 0:
                loss_value = torch.norm(torch.mul(val_mask, out_layer) - torch.mul(val_mask, input_d_corrected)) / \
                             torch.norm(torch.mul(val_mask, input_d))
            else:
                loss_value += torch.norm(torch.mul(val_mask, out_layer) - torch.mul(val_mask, input_d_corrected)) / \
                              torch.norm(torch.mul(val_mask, input_d))

        return loss_value

    def predict(self, input_d, batch_data):

        if len(np.unique(batch_data)) > 1:
            one_hot_batches = np.zeros((batch_data.shape[0], len(np.unique(batch_data))))
            one_hot_batches[:, (batch_data.reshape((1, -1)) - 1)] = 1
        else:
            one_hot_batches = np.ones_like(batch_data)
        input_d = torch.tensor(input_d)
        one_hot_batches = torch.tensor(one_hot_batches)
        input_d = input_d.cuda() if self.use_cuda else input_d
        one_hot_batches = one_hot_batches.cuda() if self.use_cuda else one_hot_batches
        output_layer, latent_layer, batch_removal_layer = self.model(input_d,
                                                                     one_hot_batches.float())

        latent_layer_numpy = [l.detach().numpy() for l in latent_layer]

        return latent_layer_numpy

