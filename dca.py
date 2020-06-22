import numpy as np
import pandas as pd
from typing import List, Union
import logging
import torch
from torch import nn
import torch.optim as optim


from trainer import Trainer
from dataset import GeneExpressionDataset, normalize
from network import AutoEncoder, NBAutoEncoder, ZINBAutoEncoder
from losses import nb_loss, zinb_loss


class DCATrainer(Trainer):
    def __init__(
            self,
            model: Union[NBAutoEncoder, ZINBAutoEncoder],
            gene_dataset: GeneExpressionDataset,
            use_mask: bool = True,
            scale_factor=1.0,
            ridge_lambda=0.0,
            l1_coeff=0.,
            l2_coeff=0.,
            **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)

        self.use_mask = use_mask
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.ridge_lambda = None

        (train_set, test_set, val_set) = self.train_test_validation()
        self.register_posterior(train_set, test_set, val_set)

        if type(self.model).__name__ == 'ZINBAutoEncoder':
            self.ridge_lambda = ridge_lambda
            print("HUA HAI")
            self.loss = zinb_loss
        else:
            self.loss = nb_loss
        self.latent_output = None

    def on_training_begin(self):
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def model_output(self, data_tensor):
        data_tensors, indices = data_tensor

        size_factors_layer = torch.tensor(self.gene_dataset.size_factor[indices, :])
        size_factors_layer = size_factors_layer.cuda() if self.use_cuda else size_factors_layer

        latent_output, output = self.model(data_tensors, size_factors_layer)
        self.latent_output = latent_output
        return output, data_tensors

    def on_training_loop(self, data):
        output, data_tensors = self.model_output(data)
        self.optimizer.zero_grad()

        self.current_loss = loss = self.loss(data_tensors, output, eps=self.eps, scale_factor=self.scale_factor,
                                             ridge_lambda=self.ridge_lambda)
        l1_reg = torch.tensor(0.)
        l2_reg = torch.tensor(0.)
        for param in self.model.parameters():
            l1_reg += torch.norm(param, p=1)
            l2_reg += (torch.norm(param, p=2) ** 2)
        self.current_loss = loss = loss + self.l1_coeff*l1_reg + self.l2_coeff*l2_reg

        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5)
        self.optimizer.step()

    def predict(self, input_d, size_factor):
        latent_output, output = self.model(torch.tensor(input_d), torch.tensor(size_factor))
        latent_output = latent_output.detach().numpy()
        output = [l.detach().numpy() for l in output]
        return latent_output, output





