import numpy as np
from typing import List
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


from scdeep.trainer import Trainer
from scdeep.dataset import GeneExpressionDataset
from scdeep.network import AutoEncoder, LinearActivation


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
                                      bias_initializer='zeros', batchnorm=False)

        if len(exp_batch_input) > 0:
            num_batch = exp_batch_input.shape[1]
        else:
            num_batch = 1

        self.batch_effect_layer = nn.Linear(num_batch, self.input_dim, bias=False)
        nn.init.zeros_(self.batch_effect_layer.weight)

        impute_layer1 = LinearActivation(self.input_dim, 64, activation='relu',
                                         weight_init='normal', weight_init_params={'std': 0.1},
                                         bias_init='zeros')

        impute_layer2 = LinearActivation(64, self.input_dim, activation=None,
                                         weight_init='normal', weight_init_params={'std': 0.1},
                                         bias_init='zeros')

        # impute_layer2 = nn.Linear(64, self.input_dim)
        # nn.init.zeros_(impute_layer2.bias)
        # nn.init.normal_(impute_layer2.weight, mean=0.0, std=0.1)

        self.imputation_model = nn.Sequential(impute_layer1, impute_layer2)

    def forward(self, x, exp_batch_input):
        self.input = x
        latent_features_list = []
        output_list = []
        batch_effect_removal_layer = self.batch_effect_layer(exp_batch_input)
        for i in range(self.t):
            if i == 0:
                x = F.relu(x - batch_effect_removal_layer)
            else:
                imputed = self.imputation_model(output)
                imputed = torch.mul(1 - torch.sign(self.input), imputed)
                x = F.relu(imputed + self.input - batch_effect_removal_layer)
            encoded = self.encode(x)
            latent_features = self.latent_layer(encoded)
            decoded = self.decode(latent_features)
            output = self.output_layer(decoded)
            output_list.append(output)
            latent_features_list.append(latent_features)

        return output_list, latent_features_list, batch_effect_removal_layer


class scScopeTrainer(Trainer):

    def __init__(
            self,
            model: scScope,
            gene_dataset: GeneExpressionDataset,
            use_mask: bool = True,
            **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)

        self.use_mask = use_mask

        if self.gene_dataset.num_batches > 1:
            self.one_hot_batches = np.zeros((self.gene_dataset.nb_cells, self.gene_dataset.num_batches))
            self.one_hot_batches[:, (self.gene_dataset.batch_indices.reshape((1, -1)) - 1)] = 1.0
        else:
            self.one_hot_batches = np.zeros_like(self.gene_dataset.batch_indices)
        self.one_hot_batches = torch.tensor(self.one_hot_batches, dtype=torch.float32, device=self.device)
        # self.one_hot_batches = self.one_hot_batches.cuda() if self.use_cuda else self.one_hot_batches

        (train_set, test_set, val_set) = self.train_test_validation()
        self.register_posterior(train_set, test_set, val_set)

    def on_training_begin(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def model_output(self, data_tensor):

        data, indices = data_tensor
        batch_data = self.one_hot_batches[indices.long(), :]

        output_list, latent_list, batch_effect_removal_layer = self.model(data, batch_data)
        return output_list, latent_list, batch_effect_removal_layer, data

    def on_training_loop(self, data_tensor):
        output_list, latent_list, batch_effect_removal_layer, data = self.model_output(data_tensor)
        self.current_loss = loss = self.loss(output_list, data, use_mask=self.use_mask,
                                             batch_effect_removal_layer=batch_effect_removal_layer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def on_validation(self, data_tensor, loss):
        output_list, latent_list, batch_effect_removal_layer, data = self.model_output(data_tensor)
        loss.append(self.loss(output_list, data, use_mask=self.use_mask,
                              batch_effect_removal_layer=batch_effect_removal_layer).item())
        return loss

    def loss(self, output_layer_list, input_d, use_mask, batch_effect_removal_layer):
        # input_d_corrected = input_d - batch_effect_removal_layer
        # if use_mask:
        #     val_mask = torch.sign(input_d_corrected)
        # else:
        #     val_mask = torch.sign(input_d_corrected + 1)

        # for i in range(len(output_layer_list)):
        #     out_layer = output_layer_list[i]
        #     if i == 0:
        #         loss_value = torch.norm(torch.mul(val_mask, (out_layer - input_d_corrected))) / \
        #                      torch.norm(torch.mul(val_mask, input_d))
        #     else:
        #         loss_value += torch.norm(torch.mul(val_mask, (out_layer - input_d_corrected))) / \
        #                       torch.norm(torch.mul(val_mask, input_d))
        # return loss_value
        input_d_corrected = input_d - batch_effect_removal_layer
        val_mask = torch.sign(input_d_corrected)
        for i in range(len(output_layer_list)):
            out_layer = output_layer_list[i]
            if i == 0:
                loss_value = (torch.norm(torch.mul(val_mask, out_layer - input_d_corrected)))
            else:
                loss_value = loss_value + (torch.norm(torch.mul(val_mask, out_layer - input_d_corrected)))
        return loss_value

    @torch.no_grad()
    def predict(self, input_d, batch_data):

        if len(np.unique(batch_data)) > 1:
            one_hot_batches = np.zeros((batch_data.shape[0], len(np.unique(batch_data))))
            one_hot_batches[:, (batch_data.reshape((1, -1)) - 1)] = 1
        else:
            one_hot_batches = np.zeros_like(batch_data)
        input_d = torch.tensor(input_d, device=self.device)
        one_hot_batches = torch.tensor(one_hot_batches, dtype=torch.float32, device=self.device)

        output_layer, latent_layer, batch_removal_layer = self.model(input_d,
                                                                     one_hot_batches)

        output_layer_numpy = [o.cpu().detach().numpy() for o in output_layer]
        latent_layer_numpy = [l.cpu().detach().numpy() for l in latent_layer]
        batch_removal_layer_numpy = batch_removal_layer.cpu().detach().numpy()

        return output_layer_numpy, latent_layer_numpy, batch_removal_layer_numpy

